#!/usr/bin/env python3
"""Apply greedy OTU-style clustering using trained NanoPred fast/full models."""

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import pandas as pd
from Bio import SeqIO

from src.data_creation import compute_metrics, compute_pair_features, rna_to_dna
from train_model import _get_classifier_scores, expand_features


SUPPORTED_FASTQ_EXTENSIONS = (".fastq", ".fq", ".fastq.gz", ".fq.gz")


def _run_cutadapt(
    r1_path: str,
    r2_path: str,
    out_r1: str,
    out_r2: str,
    primer5: Optional[str],
    primer3: Optional[str],
) -> None:
    """Trim primers from a paired-end FASTQ file pair using cutadapt.

    Uses relaxed settings suitable for Nanopore reads (high error rate, Q10
    quality): 20 % error tolerance.  Reads where no primer is found are kept
    unchanged rather than discarded (no ``--discard-untrimmed``).

    In paired-end mode ``-g`` trims the forward primer from the 5' end of R1
    and ``-G`` trims the reverse primer from the 5' end of R2, matching the
    library prep orientation.

    Args:
        r1_path:  Path to the R1 input FASTQ (plain or gzip).
        r2_path:  Path to the R2 input FASTQ (plain or gzip).
        out_r1:   Path for the trimmed R1 output FASTQ.
        out_r2:   Path for the trimmed R2 output FASTQ.
        primer5:  Forward primer sequence used as ``-g`` for R1 (or None).
        primer3:  Reverse primer sequence used as ``-G`` for R2 (or None).

    Raises:
        RuntimeError: If cutadapt is not installed or exits with an error.
    """
    cmd = ["cutadapt"]
    if primer5:
        cmd.extend(["-g", primer5])
    if primer3:
        cmd.extend(["-G", primer3])
    # -e 0.2: allow up to 20 % mismatches — tolerant for Nanopore error rates.
    # Default cutadapt behaviour keeps untrimmed reads, which is the relaxed
    # policy we want (never discard a read just because primers are absent).
    cmd.extend(["-e", "0.2", "-o", out_r1, "-p", out_r2, r1_path, r2_path])

    try:
        result = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            "cutadapt is not installed or not on PATH. "
            "Install it with: pip install cutadapt"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"cutadapt failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )


# Matches filenames of the form <sample>_R1[_NNN].<ext> (case-insensitive).
# Groups: 1=sample prefix, 2=_R1 literal, 3=optional _NNN suffix, 4=extension.
_R1_RE = re.compile(r"^(.+?)(_R1)(_\d+)?(\.(fastq|fq)(\.gz)?)$", re.IGNORECASE)


def discover_fastq_pairs(input_dir: str) -> List[Tuple[str, str, str]]:
    """Return ``(r1_path, r2_path, sample_name)`` tuples for every R1/R2 pair.

    Files must be named ``<sample>_R1.<ext>`` / ``<sample>_R2.<ext>`` (or
    ``<sample>_R1_NNN.<ext>`` / ``<sample>_R2_NNN.<ext>`` for Illumina-style
    run-number suffixes).  Supported extensions: ``fastq``, ``fq``,
    ``fastq.gz``, ``fq.gz`` (case-insensitive).

    Raises:
        FileNotFoundError: If *input_dir* does not exist or a matching R2 file
            cannot be found for a discovered R1 file.
        ValueError: If no R1/R2 pairs are found in *input_dir*.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pairs: List[Tuple[str, str, str]] = []
    for name in sorted(os.listdir(input_dir)):
        m = _R1_RE.match(name)
        if m is None:
            continue
        prefix = m.group(1)        # sample name
        r1_suffix = m.group(2)     # "_R1" (preserves original case)
        num_suffix = m.group(3) or ""  # e.g. "_001" or ""
        ext = m.group(4)           # e.g. ".fastq.gz"

        r2_name = prefix + r1_suffix.replace("1", "2") + num_suffix + ext
        r1_path = os.path.join(input_dir, name)
        r2_path = os.path.join(input_dir, r2_name)

        if not os.path.isfile(r2_path):
            raise FileNotFoundError(
                f"Found R1 file {name!r} but matching R2 file {r2_name!r} "
                f"does not exist in {input_dir}"
            )

        pairs.append((r1_path, r2_path, prefix))

    if not pairs:
        raise ValueError(
            f"No R1/R2 FASTQ pairs found in {input_dir}. "
            "Files must be named <sample>_R1.<ext> / <sample>_R2.<ext> "
            f"with supported extensions: {', '.join(SUPPORTED_FASTQ_EXTENSIONS)}"
        )
    return pairs


def _open_fastq(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_fastq_records(path: str):
    with _open_fastq(path) as handle:
        for record in SeqIO.parse(handle, "fastq"):
            seq = rna_to_dna(str(record.seq))
            quality = record.letter_annotations.get("phred_quality", [])
            yield seq, quality


def _quality_mean(scores: Sequence[int]) -> float:
    return float(sum(scores)) / len(scores) if scores else 0.0


def dereplicate_fastq_files(
    fastq_pairs: Sequence[Tuple[str, str, str]],
    primer5: Optional[str] = None,
    primer3: Optional[str] = None,
) -> Tuple[List[str], Dict[str, dict]]:
    """Dereplicate reads from paired-end FASTQ files, optionally trimming primers.

    For each ``(r1_path, r2_path, sample_name)`` tuple in *fastq_pairs*:

    * If primers are supplied, runs cutadapt on the pair to produce trimmed
      copies in a temporary directory.
    * Reads both (trimmed) R1 and R2 files into a per-sample sequence table,
      counting occurrences and keeping the highest-mean-quality representative.
    * Merges per-sample tables into a global dereplication table.

    Args:
        fastq_pairs: Sequence of ``(r1_path, r2_path, sample_name)`` tuples as
            returned by :func:`discover_fastq_pairs`.
        primer5: Forward primer sequence for cutadapt ``-g`` (or ``None``).
        primer3: Reverse primer sequence for cutadapt ``-G`` (or ``None``).

    Returns:
        A tuple ``(sample_names, global_table)`` where *sample_names* is the
        ordered list of sample identifiers and *global_table* maps each unique
        sequence to a dict with keys ``sequence``, ``sequence_id``,
        ``total_abundance``, ``sample_counts``, ``representative_quality``,
        and ``representative_quality_mean``.
    """
    sample_names: List[str] = []
    global_table: Dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for r1_path, r2_path, sample_name in fastq_pairs:
            sample_names.append(sample_name)
            sample_table: Dict[str, dict] = {}

            if primer5 or primer3:
                out_r1 = os.path.join(tmpdir, sample_name + "_R1.trimmed.fastq")
                out_r2 = os.path.join(tmpdir, sample_name + "_R2.trimmed.fastq")
                _run_cutadapt(r1_path, r2_path, out_r1, out_r2, primer5, primer3)
                read_paths = [out_r1, out_r2]
            else:
                read_paths = [r1_path, r2_path]

            for read_path in read_paths:
                for seq, quality in _iter_fastq_records(read_path):
                    q_mean = _quality_mean(quality)
                    row = sample_table.get(seq)
                    if row is None:
                        sample_table[seq] = {
                            "count": 1,
                            "quality": list(quality),
                            "quality_mean": q_mean,
                        }
                    else:
                        row["count"] += 1
                        if q_mean > row["quality_mean"]:
                            row["quality"] = list(quality)
                            row["quality_mean"] = q_mean

            for seq, sample_row in sample_table.items():
                global_row = global_table.get(seq)
                if global_row is None:
                    global_table[seq] = {
                        "sequence": seq,
                        "total_abundance": int(sample_row["count"]),
                        "sample_counts": {sample_name: int(sample_row["count"])},
                        "representative_quality": list(sample_row["quality"]),
                        "representative_quality_mean": float(sample_row["quality_mean"]),
                    }
                else:
                    global_row["total_abundance"] += int(sample_row["count"])
                    global_row["sample_counts"][sample_name] = int(sample_row["count"])
                    if float(sample_row["quality_mean"]) > global_row["representative_quality_mean"]:
                        global_row["representative_quality"] = list(sample_row["quality"])
                        global_row["representative_quality_mean"] = float(sample_row["quality_mean"])

    # sequence_id is computed once after all samples are merged to avoid
    # redundant SHA-256 work during the per-sample accumulation loop.
    for seq, row in global_table.items():
        row["sequence_id"] = hashlib.sha256(seq.encode("utf-8")).hexdigest()

    return sample_names, global_table


def _load_json_if_exists(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_models_and_metadata(model_dir: str):
    fast_model_path = os.path.join(model_dir, "fast_model.pkl")
    full_model_path = os.path.join(model_dir, "full_model.pkl")
    if not os.path.exists(fast_model_path):
        raise FileNotFoundError(f"Missing model artifact: {fast_model_path}")
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Missing model artifact: {full_model_path}")

    fast_model = joblib.load(fast_model_path)
    full_model = joblib.load(full_model_path)

    fast_meta = _load_json_if_exists(os.path.join(model_dir, "fast_model_metadata.json"))
    full_meta = _load_json_if_exists(os.path.join(model_dir, "full_model_metadata.json"))
    return fast_model, full_model, fast_meta, full_meta


def resolve_selected_features(model, metadata: Optional[dict], model_name: str) -> List[str]:
    if metadata and isinstance(metadata.get("selected_features"), list) and metadata["selected_features"]:
        return list(metadata["selected_features"])
    if hasattr(model, "feature_names_in_") and len(getattr(model, "feature_names_in_")) > 0:
        return [str(v) for v in model.feature_names_in_]
    raise ValueError(
        f"Unable to determine selected features for {model_name}. "
        "Provide *_metadata.json with 'selected_features' or a model exposing feature_names_in_."
    )


def resolve_fast_threshold(fast_model, fast_metadata: Optional[dict]) -> float:
    if fast_metadata is not None and "threshold" in fast_metadata:
        return float(fast_metadata["threshold"])
    if hasattr(fast_model, "predict_proba"):
        return 0.5
    if hasattr(fast_model, "decision_function"):
        return 0.0
    return 0.5


def build_model_input(pair_features: dict, selected_features: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame([pair_features])
    expanded = expand_features(base)
    missing = [feature for feature in selected_features if feature not in expanded.columns]
    if missing:
        suffix = f" (showing first 10 of {len(missing)} total)" if len(missing) > 10 else ""
        shown = ", ".join(missing[:10])
        available = ", ".join(expanded.columns.tolist())
        raise ValueError(
            f"Missing expected model features after expansion: {shown}{suffix}. Available features: {available}"
        )
    return expanded[list(selected_features)]


def greedy_cluster(
    global_table: Dict[str, dict],
    fast_model,
    full_model,
    fast_features: Sequence[str],
    full_features: Sequence[str],
    fast_threshold: float,
    percent_identity: float,
) -> List[dict]:
    rows = sorted(
        global_table.values(),
        key=lambda row: (-int(row["total_abundance"]), row["sequence"]),
    )

    metric_cache: Dict[str, dict] = {}
    centroids: List[str] = []
    centroid_to_otu: Dict[str, str] = {}
    clustered_rows: List[dict] = []

    for row in rows:
        sequence = row["sequence"]
        if sequence not in metric_cache:
            metric_cache[sequence] = compute_metrics(sequence, row["representative_quality"])
        candidate_metrics = metric_cache[sequence]

        assigned = False
        for centroid_seq in centroids:
            if centroid_seq not in metric_cache:
                centroid_row = global_table[centroid_seq]
                metric_cache[centroid_seq] = compute_metrics(
                    centroid_seq,
                    centroid_row["representative_quality"],
                )
            centroid_metrics = metric_cache[centroid_seq]

            pair_features = compute_pair_features(candidate_metrics, centroid_metrics)
            X_fast = build_model_input(pair_features, fast_features)
            fast_score = float(_get_classifier_scores(fast_model, X_fast)[0])
            if fast_score < fast_threshold:
                continue

            X_full = build_model_input(pair_features, full_features)
            predicted_identity = float(full_model.predict(X_full)[0])
            if predicted_identity >= percent_identity:
                clustered_rows.append(
                    {
                        "sequence_id": row["sequence_id"],
                        "sequence": sequence,
                        "otu_id": centroid_to_otu[centroid_seq],
                        "centroid_sequence_id": global_table[centroid_seq]["sequence_id"],
                        "predicted_percent_identity": predicted_identity,
                        "assignment_type": "fast+full",
                        "is_centroid": False,
                        "total_abundance": int(row["total_abundance"]),
                        "sample_counts": row["sample_counts"],
                    }
                )
                assigned = True
                break

        if assigned:
            continue

        centroids.append(sequence)
        otu_id = f"OTU_{len(centroids)}"
        centroid_to_otu[sequence] = otu_id
        clustered_rows.append(
            {
                "sequence_id": row["sequence_id"],
                "sequence": sequence,
                "otu_id": otu_id,
                "centroid_sequence_id": row["sequence_id"],
                "predicted_percent_identity": None,
                "assignment_type": "new_centroid",
                "is_centroid": True,
                "total_abundance": int(row["total_abundance"]),
                "sample_counts": row["sample_counts"],
            }
        )

    return clustered_rows


def build_assignment_table(rows: Sequence[dict], sample_names: Sequence[str]) -> pd.DataFrame:
    out_rows = []
    for row in rows:
        base = {
            "sequence_id": row["sequence_id"],
            "sequence": row["sequence"],
            "otu_id": row["otu_id"],
            "centroid_sequence_id": row["centroid_sequence_id"],
            "predicted_percent_identity": row["predicted_percent_identity"],
            "assignment_type": row["assignment_type"],
            "assignment_status": "new_centroid" if row["is_centroid"] else "assigned_to_existing_otu",
            "is_centroid": row["is_centroid"],
            "total_abundance": row["total_abundance"],
            "sample_counts": json.dumps(
                {sample_name: int(row["sample_counts"].get(sample_name, 0)) for sample_name in sample_names},
                sort_keys=True,
            ),
        }
        for sample_name in sample_names:
            base[f"count_{sample_name}"] = int(row["sample_counts"].get(sample_name, 0))
        out_rows.append(base)

    return pd.DataFrame(out_rows)


def build_otu_table(rows: Sequence[dict], sample_names: Sequence[str]) -> pd.DataFrame:
    otu_order: List[str] = []
    otu_counts: Dict[str, Dict[str, int]] = {}

    for row in rows:
        otu_id = row["otu_id"]
        if otu_id not in otu_counts:
            otu_order.append(otu_id)
            otu_counts[otu_id] = {sample_name: 0 for sample_name in sample_names}
        for sample_name in sample_names:
            otu_counts[otu_id][sample_name] += int(row["sample_counts"].get(sample_name, 0))

    otu_df = pd.DataFrame.from_dict(otu_counts, orient="index")
    otu_df = otu_df.reindex(index=otu_order, columns=list(sample_names), fill_value=0)
    otu_df.index.name = "otu_id"
    return otu_df


def write_output(rows: Sequence[dict], sample_names: Sequence[str], output_path: str) -> Tuple[str, str]:
    assignments_path = output_path
    otu_table_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "OTU_table.csv")

    assignment_df = build_assignment_table(rows, sample_names)
    assignment_df.to_csv(assignments_path, sep="\t", index=False)

    otu_df = build_otu_table(rows, sample_names)
    otu_df.to_csv(otu_table_path)
    return assignments_path, otu_table_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy OTU clustering with NanoPred fast/full models."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing model artifacts.")
    parser.add_argument("--input-dir", required=True, help="Directory containing FASTQ files.")
    parser.add_argument("--output", required=True, help="Output TSV/CSV clustering path.")
    parser.add_argument(
        "--percent-identity",
        type=float,
        required=True,
        help="Assignment threshold for full-model predicted percent identity.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress details.")
    parser.add_argument(
        "--primer5",
        default=None,
        help="Forward primer sequence (cutadapt -g, applied to R1 before dereplication).",
    )
    parser.add_argument(
        "--primer3",
        default=None,
        help="Reverse primer sequence (cutadapt -G, applied to R2 before dereplication).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fast_model, full_model, fast_meta, full_meta = load_models_and_metadata(args.model_dir)
    fast_features = resolve_selected_features(fast_model, fast_meta, "fast model")
    full_features = resolve_selected_features(full_model, full_meta, "full model")
    fast_threshold = resolve_fast_threshold(fast_model, fast_meta)

    fastq_pairs = discover_fastq_pairs(args.input_dir)
    sample_names, global_table = dereplicate_fastq_files(
        fastq_pairs,
        primer5=args.primer5,
        primer3=args.primer3,
    )

    if args.verbose:
        print(f"Loaded {len(fastq_pairs)} FASTQ pair(s):")
        for r1, r2, sname in fastq_pairs:
            print(f"  - {sname}: {os.path.basename(r1)} / {os.path.basename(r2)}")
        if args.primer5 or args.primer3:
            primer_parts = []
            if args.primer5:
                primer_parts.append(f"-g {args.primer5!r}")
            if args.primer3:
                primer_parts.append(f"-G {args.primer3!r}")
            print(f"Primer trimming enabled (cutadapt): {' '.join(primer_parts)}")
        print(f"Global dereplicated sequences: {len(global_table)}")
        print(f"Using fast-model threshold: {fast_threshold:.6f}")

    clustered_rows = greedy_cluster(
        global_table=global_table,
        fast_model=fast_model,
        full_model=full_model,
        fast_features=fast_features,
        full_features=full_features,
        fast_threshold=fast_threshold,
        percent_identity=float(args.percent_identity),
    )
    assignments_path, otu_table_path = write_output(clustered_rows, sample_names, args.output)

    if args.verbose:
        n_centroids = sum(1 for row in clustered_rows if row["is_centroid"])
        print(f"Cluster assignments written to: {assignments_path}")
        print(f"OTU table written to: {otu_table_path}")
        print(f"Total unique sequences: {len(clustered_rows)}")
        print(f"Centroids/OTUs: {n_centroids}")


if __name__ == "__main__":
    main()
