"""
data_creation.py - CLI entry point for generating the NanoPred training dataset.

Usage:
    python data_creation.py -f sequences.fasta -o all_pairs_data.csv \
        [-n 500000] [-p1 PRIMER5] [-p2 PRIMER3] [--seed 42] [--chunk-size 1000]

    Multiple FASTA files can be provided as a comma-separated list:
    python data_creation.py -f file1.fasta,file2.fasta,file3.fasta \
        -o all_pairs_data.csv -n 500000
"""

import argparse
from src.data_creation import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate a large training dataset of sequence pair metrics."
    )
    parser.add_argument(
        "-f", "--fasta",
        required=True,
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
    args = parser.parse_args()

    fasta_paths = [p.strip() for p in args.fasta.split(',') if p.strip()]

    generate_dataset(
        fasta_paths=fasta_paths,
        num_pairs=args.num_pairs,
        output_csv=args.output,
        primer5=args.primer5,
        primer3=args.primer3,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
