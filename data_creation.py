"""
data_creation.py - CLI entry point for generating the NanoPred training dataset.

Usage:
    python data_creation.py -f sequences.fasta -o all_pairs_data.csv \
        [-n 500000] [-p1 PRIMER5] [-p2 PRIMER3] [--seed 42] [--chunk-size 1000]

    Multiple FASTA files can be provided as a comma-separated list:
    python data_creation.py -f file1.fasta,file2.fasta,file3.fasta \
        -o all_pairs_data.csv -n 500000

    To spawn all shards in parallel and then merge automatically:
    python data_creation.py -f file1.fasta,file2.fasta -o all_pairs_data.csv \
        -n 2000000 --num-shards 24 --merge
"""

import argparse
import os
import subprocess
import sys

from src.data_creation import generate_dataset, merge_shards


def _expected_rows(num_pairs: int, num_shards: int, shard_id: int) -> int:
    """Return the expected number of data rows for shard *shard_id*."""
    base = num_pairs // num_shards
    extra = num_pairs % num_shards
    return base + (1 if shard_id < extra else 0)


def _validate_shard_row_counts(output_csv: str, num_pairs: int, num_shards: int) -> None:
    """Verify every shard part CSV exists and contains the expected row count.

    Raises ``ValueError`` if any shard file is missing or has a row count
    that does not match the expected value.  The merge step should only
    proceed when this function returns without raising.
    """
    base, ext = os.path.splitext(output_csv)
    errors = []
    for shard_id in range(num_shards):
        shard_path = f"{base}.part{shard_id}{ext}"
        expected = _expected_rows(num_pairs, num_shards, shard_id)
        if not os.path.exists(shard_path):
            errors.append(f"  shard {shard_id}: file missing ({shard_path})")
            continue
        with open(shard_path) as f:
            actual = sum(1 for _ in f) - 1  # subtract header line
        if actual != expected:
            errors.append(
                f"  shard {shard_id}: expected {expected} rows, got {actual} ({shard_path})"
            )
    if errors:
        raise ValueError(
            "Row-count validation failed; aborting merge.\n" + "\n".join(errors)
        )


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
            "Index of this shard (0-indexed). Used internally by child "
            "processes when running in sharded mode. Default: 0."
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
            "When set together with --num-shards > 1, spawn all shards as "
            "child processes, wait for them to finish, validate row counts, "
            "and then merge the part files into the final output CSV. "
            "Shard files are deleted after merging unless --keep-shards is given."
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

    # ------------------------------------------------------------------
    # Orchestrator mode: spawn all shards then validate and merge.
    # ------------------------------------------------------------------
    if args.merge and args.num_shards > 1:
        if not args.fasta:
            parser.error("-f/--fasta is required when using --merge")

        # Build the base child command from the current sys.argv, stripping
        # --merge so children only generate their part (no recursion).
        base_argv = [a for a in sys.argv[1:] if a != "--merge"]

        print(
            f"[orchestrator] Spawning {args.num_shards} shard processes "
            f"(num-pairs={args.num_pairs}) ..."
        )
        procs = []
        try:
            for shard_id in range(args.num_shards):
                cmd = [sys.executable, sys.argv[0]] + base_argv + [
                    "--shard-id", str(shard_id),
                ]
                # Inherit stdout/stderr so output is visible directly.
                procs.append(subprocess.Popen(cmd))

            # Wait for all children.
            failed = []
            for shard_id, proc in enumerate(procs):
                rc = proc.wait()
                if rc != 0:
                    failed.append((shard_id, rc))
        except Exception:
            # Terminate any still-running children before re-raising.
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            raise

        if failed:
            details = ", ".join(f"shard {s} (rc={r})" for s, r in failed)
            raise RuntimeError(
                f"{len(failed)} shard process(es) failed: {details}"
            )

        # Validate row counts before merging.
        _validate_shard_row_counts(args.output, args.num_pairs, args.num_shards)

        # Merge validated part files.
        merge_shards(
            output_csv=args.output,
            num_shards=args.num_shards,
            seed=args.seed,
            keep_shards=args.keep_shards,
        )

        # Final sanity check: merged file should have exactly num_pairs rows.
        with open(args.output) as f:
            merged_rows = sum(1 for _ in f) - 1
        if merged_rows != args.num_pairs:
            raise ValueError(
                f"Merged CSV has {merged_rows} rows but expected {args.num_pairs}."
            )
        print(f"[orchestrator] Done. {merged_rows} rows written to {args.output}.")
        return

    # ------------------------------------------------------------------
    # Worker / single-run mode: generate one shard (or the full dataset).
    # ------------------------------------------------------------------
    fasta_paths = [p.strip() for p in args.fasta.split(",") if p.strip()] if args.fasta else []
    if not fasta_paths:
        parser.error("-f/--fasta is required")

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


if __name__ == "__main__":
    main()
