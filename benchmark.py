#!/usr/bin/env python3
"""
benchmark.py – NanoPred sequence-analysis performance benchmark.

Measures execution time for the following operations on ~1000 sequences:

  1. Loading FASTA via Biopython
  2. Primer trimming via cutPrimers
  3. GC content calculation
  4. DNA → binary transformation + hashing
  5. Getting sequence length
  6. Computing K-mer hashes (k = 3, 5, 7; hash sizes = 64, 128, 256 bits)
  7. Quality stats (mean, median, q25, q75)
  8. Quality values hashing (hash sizes = 64, 128, 256 bits)

Usage
-----
  # Generate synthetic data and run all benchmarks:
  python benchmark.py

  # Use existing FASTA/FASTQ files:
  python benchmark.py --fasta sequences.fasta --fastq sequences.fastq

  # Adjust number of sequences and warmup/timing repetitions:
  python benchmark.py --n-sequences 500 --warmup 3 --repeats 10

  # Save results to JSON:
  python benchmark.py --output results.json
"""

import argparse
import json
import os
import random
import string
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# ---------------------------------------------------------------------------
# Local metric functions
# ---------------------------------------------------------------------------
# We import from src.metrics, but make the import path robust for both
# "python benchmark.py" (repo root) and "python -m benchmark" invocations.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.metrics import (  # noqa: E402
    load_fasta,
    load_fastq,
    gc_content,
    dna_binary_hash,
    sequence_length,
    kmer_hashes,
    quality_stats,
    quality_hash,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_BASES = "ACGT"
_TYPICAL_PRIMERS = ("GTGYCAGCMGCCGCGGTAA", "GGACTACNVGGGTWTCTAAT")  # 515F / 806R


def _random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(_BASES) for _ in range(length))


def _random_quality(length: int, rng: random.Random, lo: int = 5, hi: int = 40) -> List[int]:
    return [rng.randint(lo, hi) for _ in range(length)]


def generate_test_records(
    n: int = 1000,
    min_len: int = 200,
    max_len: int = 600,
    seed: int = 42,
    with_primers: bool = True,
) -> List[SeqRecord]:
    """
    Generate *n* synthetic FASTQ SeqRecords with random DNA sequences and
    Phred quality scores.  When *with_primers* is True the forward and reverse
    primer sequences are prepended/appended to every sequence so that primer-
    trimming benchmarks have realistic input.
    """
    rng = random.Random(seed)
    records: List[SeqRecord] = []
    fwd, rev = _TYPICAL_PRIMERS

    for i in range(n):
        insert_len = rng.randint(min_len, max_len)
        insert = _random_dna(insert_len, rng)

        if with_primers:
            raw_seq = fwd + insert + rev
        else:
            raw_seq = insert

        quals = _random_quality(len(raw_seq), rng)
        rec = SeqRecord(
            Seq(raw_seq),
            id=f"seq_{i:05d}",
            description="synthetic",
        )
        rec.letter_annotations["phred_quality"] = quals
        records.append(rec)

    return records


def _write_fasta(records: List[SeqRecord], path: str) -> None:
    with open(path, "w") as fh:
        SeqIO.write(records, fh, "fasta")


def _write_fastq(records: List[SeqRecord], path: str) -> None:
    with open(path, "w") as fh:
        SeqIO.write(records, fh, "fastq")


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class BenchmarkResult:
    """Container for timing results of a single benchmark."""

    def __init__(self, name: str, times: List[float], n_items: int) -> None:
        self.name = name
        self.times = times  # per-run wall-clock times (seconds)
        self.n_items = n_items  # items processed per run

    # -- Aggregate statistics ------------------------------------------------

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean_run(self) -> float:
        return self.total / len(self.times)

    @property
    def min_run(self) -> float:
        return min(self.times)

    @property
    def max_run(self) -> float:
        return max(self.times)

    @property
    def std_run(self) -> float:
        if len(self.times) < 2:
            return 0.0
        mean = self.mean_run
        variance = sum((t - mean) ** 2 for t in self.times) / (len(self.times) - 1)
        return variance ** 0.5

    @property
    def per_item_mean(self) -> float:
        """Average time per individual item (sequence) across all runs."""
        return self.mean_run / self.n_items if self.n_items else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_items": self.n_items,
            "n_runs": len(self.times),
            "total_s": round(self.total, 6),
            "mean_run_s": round(self.mean_run, 6),
            "min_run_s": round(self.min_run, 6),
            "max_run_s": round(self.max_run, 6),
            "std_run_s": round(self.std_run, 6),
            "per_item_us": round(self.per_item_mean * 1e6, 3),
        }


def _time_function(
    fn: Callable,
    warmup: int = 2,
    repeats: int = 5,
) -> Tuple[List[float], Any]:
    """
    Execute *fn()* **warmup** times (results discarded) then **repeats** times
    while recording wall-clock duration.  Returns ``(times, last_result)``.
    """
    for _ in range(warmup):
        fn()

    times: List[float] = []
    last_result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_result = fn()
        times.append(time.perf_counter() - t0)

    return times, last_result


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_load_fasta(
    fasta_path: str,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    records: List[SeqRecord] = []

    def run() -> List[SeqRecord]:
        nonlocal records
        records = load_fasta(fasta_path)
        return records

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult("load_fasta", times, len(records))


def bench_load_fastq(
    fastq_path: str,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    records: List[SeqRecord] = []

    def run() -> List[SeqRecord]:
        nonlocal records
        records = load_fastq(fastq_path)
        return records

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult("load_fastq", times, len(records))


def bench_trim_primers(
    records: List[SeqRecord],
    primer5: str,
    primer3: str,
    fmt: str,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    """Benchmark primer trimming via cutPrimers (skipped if not installed)."""
    from src.metrics import trim_primers  # local import to allow skip on ImportError

    def run() -> List[SeqRecord]:
        return trim_primers(records, primer5, primer3, fmt=fmt)

    try:
        times, trimmed = _time_function(run, warmup=warmup, repeats=repeats)
        return BenchmarkResult("trim_primers", times, len(records))
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"  [SKIP] trim_primers – cutPrimers not available: {exc}", file=sys.stderr)
        return BenchmarkResult("trim_primers [SKIPPED]", [float("nan")], len(records))


def bench_gc_content(
    sequences: List[str],
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    def run() -> List[float]:
        return [gc_content(s) for s in sequences]

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult("gc_content", times, len(sequences))


def bench_dna_binary_hash(
    sequences: List[str],
    hash_bits: int,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    def run() -> List[bytes]:
        return [dna_binary_hash(s, hash_bits) for s in sequences]

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult(f"dna_binary_hash_{hash_bits}bit", times, len(sequences))


def bench_sequence_length(
    sequences: List[str],
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    def run() -> List[int]:
        return [sequence_length(s) for s in sequences]

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult("sequence_length", times, len(sequences))


def bench_kmer_hashes(
    sequences: List[str],
    k: int,
    hash_bits: int,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    def run() -> List[List[bytes]]:
        return [kmer_hashes(s, k, hash_bits) for s in sequences]

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult(f"kmer_hashes_k{k}_{hash_bits}bit", times, len(sequences))


def bench_quality_stats(
    quality_lists: List[List[int]],
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    def run() -> List[Dict[str, float]]:
        return [quality_stats(q) for q in quality_lists]

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult("quality_stats", times, len(quality_lists))


def bench_quality_hash(
    quality_lists: List[List[int]],
    hash_bits: int,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    def run() -> List[bytes]:
        return [quality_hash(q, hash_bits) for q in quality_lists]

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult(f"quality_hash_{hash_bits}bit", times, len(quality_lists))


def bench_combined_pipeline(
    sequences: List[str],
    quality_lists: List[List[int]],
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    """Benchmark a combined pipeline: GC + length + k-mer(5, 128) + quality stats."""

    def run() -> List[Dict[str, Any]]:
        results = []
        for seq, quals in zip(sequences, quality_lists):
            results.append(
                {
                    "length": sequence_length(seq),
                    "gc": gc_content(seq),
                    "kmers_k5_128": kmer_hashes(seq, k=5, hash_bits=128),
                    "dna_hash": dna_binary_hash(seq, 256),
                    "quality": quality_stats(quals),
                    "qual_hash": quality_hash(quals, 128),
                }
            )
        return results

    times, _ = _time_function(run, warmup=warmup, repeats=repeats)
    return BenchmarkResult("combined_pipeline", times, len(sequences))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_table(results: List[BenchmarkResult]) -> str:
    headers = [
        "Benchmark", "N seqs", "Runs",
        "Mean (s)", "Min (s)", "Max (s)", "Std (s)", "Per-seq (µs)",
    ]
    rows = []
    for r in results:
        d = r.to_dict()
        rows.append([
            d["name"],
            str(d["n_items"]),
            str(d["n_runs"]),
            f"{d['mean_run_s']:.6f}",
            f"{d['min_run_s']:.6f}",
            f"{d['max_run_s']:.6f}",
            f"{d['std_run_s']:.6f}",
            f"{d['per_item_us']:.3f}",
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    lines = [sep, fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*row))
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmarks(
    fasta_path: Optional[str] = None,
    fastq_path: Optional[str] = None,
    n_sequences: int = 1000,
    warmup: int = 2,
    repeats: int = 5,
    output: Optional[str] = None,
    skip_primers: bool = False,
    primer5: str = _TYPICAL_PRIMERS[0],
    primer3: str = _TYPICAL_PRIMERS[1],
) -> List[BenchmarkResult]:
    """
    Run the full benchmark suite and optionally write results to *output* (JSON).

    Returns the list of :class:`BenchmarkResult` objects.
    """
    results: List[BenchmarkResult] = []
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_obj.name

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    print(f"Preparing test data ({n_sequences} sequences) …")

    if fasta_path and fastq_path:
        # Load from provided files
        fasta_records = load_fasta(fasta_path)[:n_sequences]
        fastq_records = load_fastq(fastq_path)[:n_sequences]
        print(f"  Loaded {len(fasta_records)} FASTA and {len(fastq_records)} FASTQ records.")
    else:
        # Generate synthetic data
        fastq_records = generate_test_records(n=n_sequences, with_primers=True)
        # Strip quality for FASTA records (re-use same sequences)
        fasta_records = [
            SeqRecord(rec.seq, id=rec.id, description=rec.description)
            for rec in fastq_records
        ]
        fasta_path = os.path.join(tmpdir, "test.fasta")
        fastq_path = os.path.join(tmpdir, "test.fastq")
        _write_fasta(fasta_records, fasta_path)
        _write_fastq(fastq_records, fastq_path)
        print(f"  Generated {n_sequences} synthetic sequences.")

    sequences: List[str] = [str(r.seq) for r in fasta_records]
    quality_lists: List[List[int]] = [
        r.letter_annotations.get("phred_quality", []) for r in fastq_records
    ]

    # ------------------------------------------------------------------
    # 1. Load FASTA
    # ------------------------------------------------------------------
    print("\n[1/8] Loading FASTA …")
    results.append(bench_load_fasta(fasta_path, warmup=warmup, repeats=repeats))

    # ------------------------------------------------------------------
    # 2. Load FASTQ
    # ------------------------------------------------------------------
    print("[2/8] Loading FASTQ …")
    results.append(bench_load_fastq(fastq_path, warmup=warmup, repeats=repeats))

    # ------------------------------------------------------------------
    # 3. Primer trimming
    # ------------------------------------------------------------------
    if skip_primers:
        print("[3/8] Primer trimming … [SKIPPED via --skip-primers]")
    else:
        print("[3/8] Primer trimming …")
        results.append(
            bench_trim_primers(
                fastq_records,
                primer5=primer5,
                primer3=primer3,
                fmt="fastq",
                warmup=1,
                repeats=max(1, repeats // 2),
            )
        )

    # ------------------------------------------------------------------
    # 4. GC content
    # ------------------------------------------------------------------
    print("[4/8] GC content …")
    results.append(bench_gc_content(sequences, warmup=warmup, repeats=repeats))

    # ------------------------------------------------------------------
    # 5. DNA → binary hash (64 / 128 / 256 bits)
    # ------------------------------------------------------------------
    print("[5/8] DNA binary hash …")
    for bits in (64, 128, 256):
        results.append(bench_dna_binary_hash(sequences, hash_bits=bits, warmup=warmup, repeats=repeats))

    # ------------------------------------------------------------------
    # 6. Sequence length
    # ------------------------------------------------------------------
    print("[6/8] Sequence length …")
    results.append(bench_sequence_length(sequences, warmup=warmup, repeats=repeats))

    # ------------------------------------------------------------------
    # 7. K-mer hashes (k = 3, 5, 7; hash sizes = 64, 128, 256)
    # ------------------------------------------------------------------
    print("[7/8] K-mer hashes …")
    for k in (3, 5, 7):
        for bits in (64, 128, 256):
            results.append(bench_kmer_hashes(sequences, k=k, hash_bits=bits, warmup=warmup, repeats=repeats))

    # ------------------------------------------------------------------
    # 8. Quality stats + quality hash
    # ------------------------------------------------------------------
    print("[8/8] Quality stats and quality hashing …")
    results.append(bench_quality_stats(quality_lists, warmup=warmup, repeats=repeats))
    for bits in (64, 128, 256):
        results.append(bench_quality_hash(quality_lists, hash_bits=bits, warmup=warmup, repeats=repeats))

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------
    print("\n[+] Combined pipeline …")
    results.append(
        bench_combined_pipeline(sequences, quality_lists, warmup=warmup, repeats=repeats)
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(_format_table(results))

    if output:
        data = {
            "config": {
                "n_sequences": n_sequences,
                "warmup_runs": warmup,
                "timed_runs": repeats,
            },
            "results": [r.to_dict() for r in results],
        }
        with open(output, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nResults saved to {output}")

    tmpdir_obj.cleanup()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NanoPred sequence-analysis performance benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta",
        metavar="PATH",
        help="Path to input FASTA file (generated synthetically if omitted).",
    )
    parser.add_argument(
        "--fastq",
        metavar="PATH",
        help="Path to input FASTQ file (generated synthetically if omitted).",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=1000,
        metavar="N",
        help="Number of sequences to benchmark (used when generating synthetic data).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        metavar="N",
        help="Number of warmup runs (discarded) before timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        metavar="N",
        help="Number of timed repetitions per benchmark.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write results as JSON to this file.",
    )
    parser.add_argument(
        "--skip-primers",
        action="store_true",
        help="Skip primer-trimming benchmark (requires cutPrimers to be installed).",
    )
    parser.add_argument(
        "--primer5",
        default=_TYPICAL_PRIMERS[0],
        metavar="SEQ",
        help="Forward primer sequence for trimming benchmark.",
    )
    parser.add_argument(
        "--primer3",
        default=_TYPICAL_PRIMERS[1],
        metavar="SEQ",
        help="Reverse primer sequence for trimming benchmark.",
    )

    args = parser.parse_args()

    # Validate: if one of --fasta / --fastq is given, both must be given
    if bool(args.fasta) ^ bool(args.fastq):
        parser.error("--fasta and --fastq must be provided together.")

    run_benchmarks(
        fasta_path=args.fasta,
        fastq_path=args.fastq,
        n_sequences=args.n_sequences,
        warmup=args.warmup,
        repeats=args.repeats,
        output=args.output,
        skip_primers=args.skip_primers,
        primer5=args.primer5,
        primer3=args.primer3,
    )


if __name__ == "__main__":
    main()
