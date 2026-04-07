"""
benchmark.py - Benchmark sequence analysis metrics on realistic sequencing data.

Loads a FASTA file, simulates sequencing errors based on randomly drawn Phred
quality scores (uniform Q10–Q50), converts to FASTQ format, and times each
metric computation on the resulting error-containing reads.

Usage:
    python benchmark.py -f <input.fasta>
"""

import argparse
import hashlib
import math
import random
import statistics
import time
from collections import defaultdict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


# ---------------------------------------------------------------------------
# Error simulation helpers
# ---------------------------------------------------------------------------

BASES = ["A", "T", "G", "C"]


def phred_to_error_prob(q: float) -> float:
    """Convert a Phred quality score to an error probability.

    P(error) = 10^(-Q/10)
    """
    return 10 ** (-q / 10)


def simulate_errors(sequence: str, avg_phred: float) -> str:
    """Introduce substitution errors into *sequence* at each position
    independently with probability derived from *avg_phred*.

    Replacement bases are chosen uniformly from the three bases that differ
    from the original base.
    """
    p_error = phred_to_error_prob(avg_phred)
    seq_list = list(sequence.upper())
    for i, base in enumerate(seq_list):
        if random.random() < p_error:
            alt_bases = [b for b in BASES if b != base]
            seq_list[i] = random.choice(alt_bases)
    return "".join(seq_list)


def generate_quality_string(avg_phred: float, length: int) -> str:
    """Generate a per-position FASTQ quality string of the given *length*.

    Each per-position Phred score is drawn from a small Gaussian centred on
    *avg_phred* (σ = 2), clamped to [2, 40], then encoded as chr(Q + 33).
    """
    chars = []
    for _ in range(length):
        q = int(round(random.gauss(avg_phred, 2)))
        q = max(2, min(40, q))
        chars.append(chr(q + 33))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def compute_gc_content(sequence: str) -> float:
    """Return the GC fraction (0–1) of *sequence*."""
    if not sequence:
        return 0.0
    gc = sum(1 for b in sequence.upper() if b in ("G", "C"))
    return gc / len(sequence)


def dna_to_binary(sequence: str) -> list[int]:
    """Map each nucleotide to a binary value: G/C → 1, A/T → 0, other → 0."""
    mapping = {"G": 1, "C": 1, "A": 0, "T": 0}
    return [mapping.get(b, 0) for b in sequence.upper()]


def hash_data(data: bytes, hash_size: int) -> str:
    """Return a hex digest of *data* using SHAKE-128 truncated to *hash_size* bits."""
    h = hashlib.shake_128(data)
    return h.hexdigest(hash_size // 8)


def compute_dna_binary_hash(sequence: str, hash_size: int) -> str:
    """Return a hash of the binary-encoded representation of *sequence*."""
    binary = dna_to_binary(sequence)
    data = bytes(binary)
    return hash_data(data, hash_size)


def generate_kmers(sequence: str, k: int) -> list[str]:
    """Return all overlapping k-mers of length *k* from *sequence*."""
    return [sequence[i: i + k] for i in range(len(sequence) - k + 1)]


def compute_kmer_hash(sequence: str, k: int, hash_size: int) -> str:
    """Concatenate all k-mers and return a hash of the result."""
    kmers = generate_kmers(sequence, k)
    data = "".join(kmers).encode()
    return hash_data(data, hash_size)


def compute_quality_stats(quality_string: str) -> dict:
    """Return mean, median, q25, and q75 of the per-position Phred scores."""
    scores = [ord(c) - 33 for c in quality_string]
    mean = sum(scores) / len(scores)
    median = statistics.median(scores)
    q25, q75 = statistics.quantiles(scores, n=4)[0], statistics.quantiles(scores, n=4)[2]
    return {"mean": mean, "median": median, "q25": q25, "q75": q75}


def compute_quality_hash(quality_string: str, hash_size: int) -> str:
    """Return a hash of the raw quality string bytes."""
    return hash_data(quality_string.encode(), hash_size)


# ---------------------------------------------------------------------------
# Benchmarking utilities
# ---------------------------------------------------------------------------

class Timer:
    """Accumulate elapsed time across multiple calls."""

    def __init__(self):
        self.total = 0.0
        self.count = 0
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        self.total += time.perf_counter() - self._start
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


def print_results(timers: dict[str, Timer], n_reads: int) -> None:
    """Print a summary table of benchmark results."""
    col_w = 40
    print()
    print("=" * 70)
    print(f"{'Benchmark Results':^70}")
    print(f"  Reads benchmarked: {n_reads}")
    print("=" * 70)
    print(f"{'Metric':<{col_w}} {'Total (s)':>12} {'Mean (ms/read)':>16}")
    print("-" * 70)
    for name, timer in timers.items():
        print(
            f"{name:<{col_w}} {timer.total:>12.4f} {timer.mean * 1000:>16.4f}"
        )
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main benchmark routine
# ---------------------------------------------------------------------------

def run_benchmark(fasta_path: str) -> None:
    """Load *fasta_path*, simulate errors, and benchmark all metrics."""

    # --- Load FASTA ---
    t_load = Timer()
    t_load.start()
    records = list(SeqIO.parse(fasta_path, "fasta"))
    t_load.stop()
    n_reads = len(records)
    print(f"Loaded {n_reads} sequences from '{fasta_path}'.")

    if n_reads == 0:
        print("No sequences found. Exiting.")
        return

    # --- Assign Phred scores and simulate errors ---
    print("Simulating errors and generating quality strings …")
    error_sequences = []
    quality_strings = []
    for record in records:
        avg_q = random.uniform(10, 50)
        seq_str = str(record.seq)
        err_seq = simulate_errors(seq_str, avg_q)
        qual_str = generate_quality_string(avg_q, len(err_seq))
        error_sequences.append(err_seq)
        quality_strings.append(qual_str)

    # --- Define timers for each metric ---
    timers: dict[str, Timer] = {
        "FASTA loading": t_load,
        "GC content": Timer(),
        "Sequence length": Timer(),
        "DNA binary hash (64-bit)": Timer(),
        "DNA binary hash (128-bit)": Timer(),
        "DNA binary hash (256-bit)": Timer(),
    }
    for k in (3, 5, 7):
        for bits in (64, 128, 256):
            timers[f"K-mer hash k={k} ({bits}-bit)"] = Timer()
    for bits in (64, 128, 256):
        timers[f"Quality hash ({bits}-bit)"] = Timer()
    timers["Quality stats"] = Timer()

    # --- Benchmark each metric ---
    for seq, qual in zip(error_sequences, quality_strings):
        # GC content
        timers["GC content"].start()
        _ = compute_gc_content(seq)
        timers["GC content"].stop()

        # Sequence length
        timers["Sequence length"].start()
        _ = len(seq)
        timers["Sequence length"].stop()

        # DNA binary hashing
        for bits in (64, 128, 256):
            timers[f"DNA binary hash ({bits}-bit)"].start()
            _ = compute_dna_binary_hash(seq, bits)
            timers[f"DNA binary hash ({bits}-bit)"].stop()

        # K-mer hashing
        for k in (3, 5, 7):
            for bits in (64, 128, 256):
                timers[f"K-mer hash k={k} ({bits}-bit)"].start()
                _ = compute_kmer_hash(seq, k, bits)
                timers[f"K-mer hash k={k} ({bits}-bit)"].stop()

        # Quality hashing
        for bits in (64, 128, 256):
            timers[f"Quality hash ({bits}-bit)"].start()
            _ = compute_quality_hash(qual, bits)
            timers[f"Quality hash ({bits}-bit)"].stop()

        # Quality stats
        timers["Quality stats"].start()
        _ = compute_quality_stats(qual)
        timers["Quality stats"].stop()

    print_results(timers, n_reads)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sequence analysis metrics on realistic sequencing data. "
            "Loads a FASTA file, simulates per-read sequencing errors based on "
            "a uniformly drawn average Phred score (Q10–Q50), and times each "
            "metric computation."
        )
    )
    parser.add_argument(
        "-f",
        "--fasta",
        required=True,
        help="Path to the input FASTA file.",
    )
    args = parser.parse_args()
    run_benchmark(args.fasta)


if __name__ == "__main__":
    main()
