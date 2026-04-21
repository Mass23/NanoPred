"""
data_creation.py - CLI entry point for generating the NanoPred training dataset.

Usage:
    python data_creation.py -f sequences.fasta -o all_pairs_data.csv \
        [-n 500000] [-p1 PRIMER5] [-p2 PRIMER3] [--seed 42] [--chunk-size 1000]

    Multiple FASTA files can be provided as a comma-separated list:
    python data_creation.py -f file1.fasta,file2.fasta,file3.fasta \
        -o all_pairs_data.csv -n 500000

    To run all 24 shards automatically and then merge (recommended):
        python data_creation.py -f seqs.fasta -n 500000 \
            --num-shards 24 --run-sharded -o all_pairs_data.csv

    To run a single shard manually (and merge later):
        python data_creation.py -f seqs.fasta -n 500000 \
            --num-shards 24 --shard-id 3 -o all_pairs_data.csv

    To merge previously generated shard files without re-generating:
        python data_creation.py --merge-only -o all_pairs_data.csv --num-shards 24
"""

import argparse
import subprocess
import sys

from src.data_creation import generate_dataset, merge_shards


def _build_shard_cmd(base_argv: list, shard_id: int, num_shards: int) -> list:
    """Build the subprocess command for one shard from the parent argv.

    Strips flags that must not be forwarded to child shards (--run-sharded,
    --merge-only, --merge) and any existing --shard-id / --num-shards values,
    then appends the per-shard values.
    """
    skip_flags = {'--run-sharded', '--merge-only', '--merge'}
    skip_with_value = {'--shard-id', '--num-shards'}

    filtered = []
    i = 0
    while i < len(base_argv):
        arg = base_argv[i]
        if arg in skip_flags:
            i += 1
        elif arg in skip_with_value:
            i += 2  # skip flag and its value
        else:
            filtered.append(arg)
            i += 1

    return (
        [sys.executable, __file__]
        + filtered
        + ['--shard-id', str(shard_id), '--num-shards', str(num_shards)]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a large training dataset of sequence pair metrics.\n\n"
            "Quick start (parallel, 24 shards):\n"
            "  python data_creation.py -f seqs.fasta -n 500000 \\\n"
            "      --num-shards 24 --run-sharded -o all_pairs_data.csv\n\n"
            "Merge existing shard files only:\n"
            "  python data_creation.py --merge-only -o all_pairs_data.csv --num-shards 24"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f", "--fasta",
        required=False,
        default=None,
        help=(
            "Path(s) to input FASTA file(s) containing reference sequences. "
            "Multiple files can be provided as a comma-separated list "
            "(e.g. file1.fasta,file2.fasta,file3.fasta). "
            "Not required when using --merge-only."
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
        "--run-sharded",
        action="store_true",
        default=False,
        help=(
            "Spawn --num-shards subprocesses of this script (shard-id 0 .. "
            "num-shards-1), wait for all to complete, then merge the part "
            "files into the final output CSV. Requires --num-shards > 1 and "
            "-f/--fasta. Per-shard stdout/stderr is written to "
            "<output>.shard_<id>.log files."
        ),
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        default=False,
        help=(
            "Skip data generation and only merge existing shard part files "
            "into the final output CSV. Requires --num-shards > 1."
        ),
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        default=False,
        help=(
            "Keep the individual shard part files after merging instead of "
            "deleting them (default: delete after merge)."
        ),
    )
    # Legacy alias kept for backward compatibility
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # Unify legacy flags
    merge_only = args.merge_only or args.merge
    keep_parts = args.keep_parts or args.keep_shards

    # --- Merge-only mode ---
    if merge_only:
        if args.num_shards <= 1:
            parser.error("--merge-only requires --num-shards > 1")
        merge_shards(
            output_csv=args.output,
            num_shards=args.num_shards,
            seed=args.seed,
            keep_shards=keep_parts,
        )
        return

    # All other modes require --fasta
    fasta_paths = [p.strip() for p in args.fasta.split(',') if p.strip()] if args.fasta else []
    if not fasta_paths:
        parser.error("-f/--fasta is required when not using --merge-only")

    # --- Run-sharded mode: spawn subprocesses, wait, then merge ---
    if args.run_sharded:
        if args.num_shards <= 1:
            parser.error("--run-sharded requires --num-shards > 1")

        base, ext = __import__('os').path.splitext(args.output)
        procs = []
        log_files = []
        base_argv = sys.argv[1:]  # everything after the script name
        for shard_id in range(args.num_shards):
            cmd = _build_shard_cmd(base_argv, shard_id=shard_id, num_shards=args.num_shards)
            log_path = f'{base}.shard_{shard_id}.log'
            log_files.append(log_path)
            log_fh = open(log_path, 'w')
            procs.append((shard_id, subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh), log_fh))

        print(f"Launched {args.num_shards} shard processes. Waiting for completion ...")
        failed = []
        for shard_id, proc, log_fh in procs:
            ret = proc.wait()
            log_fh.close()
            if ret != 0:
                failed.append(shard_id)

        if failed:
            print(
                f"WARNING: {len(failed)} shard(s) exited with non-zero status: "
                f"{failed}. Check the corresponding log files for details.",
                file=sys.stderr,
            )

        print("All shards done. Merging ...")
        merge_shards(
            output_csv=args.output,
            num_shards=args.num_shards,
            seed=args.seed,
            keep_shards=keep_parts,
        )
        return

    # --- Normal mode: generate a single shard (or the whole dataset) ---
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
