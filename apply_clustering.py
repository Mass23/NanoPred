#!/usr/bin/env python3
"""Apply greedy OTU-style clustering using trained NanoPred fast/full models."""

import argparse
import gzip
import hashlib
import json
import os
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
    input_path: str,
    output_path: str,
    primer5: Optional[str],
    primer3: Optional[str],
) -> str:
    """Trim primers from a FASTQ file using cutadapt.

    Uses relaxed settings suitable for Nanopore reads (high error rate, Q10
    quality): 20 % error tolerance and RC-strand detection.  Reads where no
    primer is found are kept unchanged rather than discarded.

    Args:
        input_path:  Path to the input FASTQ (plain or gzip).
        output_path: Path for the trimmed output FASTQ.
        primer5:     5' primer sequence (or None to skip).
        primer3:     3' primer sequence (or None to skip).

    Returns:
        *output_path* on success.

    Raises:
        RuntimeError: If cutadapt is not installed or exits with an error.
    """
    cmd = ["cutadapt"]
    if primer5:
        cmd.extend(["-g", primer5])
    if primer3:
        cmd.extend(["-a", primer3])
    # --rc: also search the reverse complement of the adapter sequences so reads
    #       from either strand of the amplicon are trimmed correctly.
    # -e 0.2: allow up to 20 % mismatches — tolerant for Nanopore error rates.
    # Default cutadapt behaviour keeps untrimmed reads, which is the relaxed
    # policy we want (never discard a read just because primers are absent).
    cmd.extend(["--rc", "-e", "0.2", "-o", output_path, input_path])

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
    return output_path


def discover_fastq_files(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, name)
        if os.path.isfile(full_path) and name.lower().endswith(SUPPORTED_FASTQ_EXTENSIONS):
            files.append(full_path)

    if not files:
        raise ValueError(
            f"No FASTQ files found in {input_dir}. Expected extensions: {', '.join(SUPPORTED_FASTQ_EXTENSIONS)}"
        )
    return files


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
    fastq_files: Sequence[str],
    primer5: Optional[str] = None,
    primer3: Optional[str] = None,
) -> Tuple[List[str], Dict[str, dict]]:
    sample_names: List[str] = []
    global_table: Dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for path in fastq_files:
            sample_name = os.path.basename(path)
            sample_names.append(sample_name)
            sample_table: Dict[str, dict] = {}

            if primer5 or primer3:
                trimmed_path = os.path.join(tmpdir, sample_name + ".trimmed.fastq")
                _run_cutadapt(path, trimmed_path, primer5, primer3)
                read_path = trimmed_path
            else:
                read_path = path

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
        help="5' primer sequence for trimming before dereplication (optional).",
    )
    parser.add_argument(
        "--primer3",
        default=None,
        help="3' primer sequence for trimming before dereplication (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fast_model, full_model, fast_meta, full_meta = load_models_and_metadata(args.model_dir)
    fast_features = resolve_selected_features(fast_model, fast_meta, "fast model")
    full_features = resolve_selected_features(full_model, full_meta, "full model")
    fast_threshold = resolve_fast_threshold(fast_model, fast_meta)

    fastq_files = discover_fastq_files(args.input_dir)
    sample_names, global_table = dereplicate_fastq_files(
        fastq_files,
        primer5=args.primer5,
        primer3=args.primer3,
    )

    if args.verbose:
        print(f"Loaded {len(fastq_files)} FASTQ file(s):")
        for path in fastq_files:
            print(f"  - {path}")
        if args.primer5 or args.primer3:
            print(f"Primer trimming enabled: primer5={args.primer5!r}, primer3={args.primer3!r}")
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
