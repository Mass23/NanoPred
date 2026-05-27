#!/usr/bin/env python3
"""Apply greedy OTU-style clustering using trained NanoPred fast/full models."""

import argparse
import gzip
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import pandas as pd
from Bio import SeqIO

from src.data_creation import compute_metrics, compute_pair_features, rna_to_dna
from train_model import _get_classifier_scores, expand_features


SUPPORTED_FASTQ_EXTENSIONS = (".fastq", ".fq", ".fastq.gz", ".fq.gz")


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


def dereplicate_fastq_files(fastq_files: Sequence[str]) -> Tuple[List[str], Dict[str, dict]]:
    sample_names: List[str] = []
    global_table: Dict[str, dict] = {}

    for path in fastq_files:
        sample_name = os.path.basename(path)
        sample_names.append(sample_name)
        sample_table: Dict[str, dict] = {}

        for seq, quality in _iter_fastq_records(path):
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
        otu_id = f"OTU_{len(centroids):06d}"
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


def write_output(rows: Sequence[dict], sample_names: Sequence[str], output_path: str) -> None:
    out_rows = []
    for row in rows:
        base = {
            "sequence_id": row["sequence_id"],
            "sequence": row["sequence"],
            "otu_id": row["otu_id"],
            "centroid_sequence_id": row["centroid_sequence_id"],
            "predicted_percent_identity": row["predicted_percent_identity"],
            "assignment_type": row["assignment_type"],
            "is_centroid": row["is_centroid"],
            "total_abundance": row["total_abundance"],
        }
        for sample_name in sample_names:
            base[f"count_{sample_name}"] = int(row["sample_counts"].get(sample_name, 0))
        out_rows.append(base)

    df = pd.DataFrame(out_rows)
    sep = "\t" if output_path.lower().endswith(".tsv") else ","
    df.to_csv(output_path, sep=sep, index=False)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fast_model, full_model, fast_meta, full_meta = load_models_and_metadata(args.model_dir)
    fast_features = resolve_selected_features(fast_model, fast_meta, "fast model")
    full_features = resolve_selected_features(full_model, full_meta, "full model")
    fast_threshold = resolve_fast_threshold(fast_model, fast_meta)

    fastq_files = discover_fastq_files(args.input_dir)
    sample_names, global_table = dereplicate_fastq_files(fastq_files)

    if args.verbose:
        print(f"Loaded {len(fastq_files)} FASTQ file(s):")
        for path in fastq_files:
            print(f"  - {path}")
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
    write_output(clustered_rows, sample_names, args.output)

    if args.verbose:
        n_centroids = sum(1 for row in clustered_rows if row["is_centroid"])
        print(f"Clusters written to: {args.output}")
        print(f"Total unique sequences: {len(clustered_rows)}")
        print(f"Centroids/OTUs: {n_centroids}")


if __name__ == "__main__":
    main()
