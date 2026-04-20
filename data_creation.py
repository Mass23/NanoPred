"""
data_creation.py - CLI entry point for generating the NanoPred training dataset.

Usage:
    python data_creation.py -f sequences.fasta -o all_pairs_data.csv \
        [-n 500000] [-p1 PRIMER5] [-p2 PRIMER3] [--seed 42] [--chunk-size 1000]

    Multiple FASTA files can be provided as a comma-separated list:
    python data_creation.py -f file1.fasta,file2.fasta,file3.fasta \
        -o all_pairs_data.csv -n 500000

    To run sharded generation in parallel and then merge:
        for i in $(seq 0 23); do
            python data_creation.py -f seqs.fasta -n 500000 \\
                --num-shards 24 --shard-id $i &
        done
        wait
        python data_creation.py --merge -o all_pairs_data.csv --num-shards 24
"""

import argparse
from src.data_creation import generate_dataset, merge_shards


def main():
    parser = argparse.ArgumentParser(
        description="Generate a large training dataset of sequence pair metrics."
    )
    parser.add_argument(
        "-f", "--fasta",
        required=False,
        default=None,
        help=(
            "Path(s) to input FASTA file(s) containing reference sequences. "
            "Multiple files can be provided as a comma-separated list "
            "(e.g. file1.fasta,file2.fasta,file3.fasta)."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default="all_pairs_data.csv",
        help="Output CSV file path (default: all_pairs_data.csv).",
    )
    parser.add_argument(
        "-n", "--num-pairs",
        type=int,
        default=500_000,
        help="Number of sequence pairs to generate (default: 500000).",
    )
    parser.add_argument(
        "-p1", "--primer5",
        default="",
        help="Forward (5') primer sequence for trimming (optional).",
    )
    parser.add_argument(
        "-p2", "--primer3",
        default="",
        help="Reverse (3') primer sequence for trimming (optional).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of pairs to process per batch (default: 1000).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help=(
            "Index of this shard (0-indexed). Use with --num-shards to split "
            "dataset generation across multiple processes. Default: 0."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Total number of shards. When > 1, each shard writes to a separate "
            "file (e.g. all_pairs_data.part0.csv). Default: 1 (no sharding)."
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help=(
            "Merge all shard part files into the final output CSV (shuffled). "
            "Requires --num-shards > 1. Shard files are deleted after merging "
            "unless --keep-shards is also given."
        ),
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        default=False,
        help=(
            "When --merge is used, keep the individual shard part files after "
            "merging instead of deleting them (default: delete)."
        ),
    )
    args = parser.parse_args()

    fasta_paths = [p.strip() for p in args.fasta.split(',') if p.strip()] if args.fasta else []
    if not fasta_paths:
        parser.error("-f/--fasta is required when not using --merge")

    generate_dataset(
        fasta_paths=fasta_paths,
        num_pairs=args.num_pairs,
        output_csv=args.output,
        primer5=args.primer5,
        primer3=args.primer3,
        seed=args.seed,
        chunk_size=args.chunk_size,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    if args.merge:
            if args.num_shards <= 1:
                parser.error("--merge requires --num-shards > 1")
            merge_shards(
                output_csv=args.output,
                num_shards=args.num_shards,
                seed=args.seed,
                keep_shards=args.keep_shards,
            )
            return

if __name__ == "__main__":
    main()
