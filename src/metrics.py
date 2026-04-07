"""
Metrics module for NanoPred.

Provides individual functions for each sequence analysis metric used in the
benchmarking pipeline. Every public function accepts plain Python types (str,
list, …) so callers are not forced to keep Biopython objects around.
"""

import hashlib
import os
import statistics
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_fasta(filepath: str) -> List[SeqRecord]:
    """Load all records from a FASTA file using Biopython."""
    with open(filepath, "r") as fh:
        return list(SeqIO.parse(fh, "fasta"))


def load_fastq(filepath: str) -> List[SeqRecord]:
    """Load all records from a FASTQ file using Biopython."""
    with open(filepath, "r") as fh:
        return list(SeqIO.parse(fh, "fastq"))


# ---------------------------------------------------------------------------
# 2. Primer trimming via cutPrimers
# ---------------------------------------------------------------------------

def _run_cutprimers(
    input_path: str,
    output_path: str,
    primer5: str,
    primer3: Optional[str],
    fmt: str,
    untrimmed_path: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Run the cutPrimers command-line tool and return (trimmed_path, untrimmed_path)."""
    cmd = [
        "python", "-m", "cutPrimers",
        f"--{'fastq' if fmt == 'fastq' else 'fasta'}", input_path,
        "--primer5", primer5,
        "--outputFile", output_path,
    ]
    if primer3:
        cmd += ["--primer3", primer3]
    if untrimmed_path:
        cmd += ["--outputUntrimmFile", untrimmed_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cutPrimers failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_path, untrimmed_path


def _reverse_complement_records(records: List[SeqRecord]) -> List[SeqRecord]:
    """Return a list of reverse-complemented SeqRecords."""
    rc = []
    for rec in records:
        rc.append(rec.reverse_complement(id=rec.id, description=rec.description))
    return rc


def _write_seqs(records: List[SeqRecord], path: str, fmt: str) -> None:
    with open(path, "w") as fh:
        SeqIO.write(records, fh, fmt)


def trim_primers(
    records: List[SeqRecord],
    primer5: str,
    primer3: Optional[str] = None,
    fmt: str = "fastq",
) -> List[SeqRecord]:
    """
    Trim primers from *records* using cutPrimers.

    Applies a bidirectional strategy:
    1. First pass trims in the original orientation.
    2. Sequences not trimmed in pass 1 are reverse-complemented and retried
       with swapped primers.

    Parameters
    ----------
    records:
        Sequence records (FASTA or FASTQ).
    primer5:
        Forward (5') primer sequence.
    primer3:
        Reverse (3') primer sequence (optional).
    fmt:
        ``"fasta"`` or ``"fastq"``.

    Returns
    -------
    List[SeqRecord]
        All successfully trimmed records.

    Raises
    ------
    RuntimeError
        If cutPrimers is not installed or exits with a non-zero status.
    FileNotFoundError
        If the cutPrimers module cannot be located.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input.{fmt}")
        _write_seqs(records, input_path, fmt)

        trimmed_path = os.path.join(tmpdir, f"trimmed.{fmt}")
        untrimmed_path = os.path.join(tmpdir, f"untrimmed.{fmt}")

        _run_cutprimers(input_path, trimmed_path, primer5, primer3, fmt, untrimmed_path)

        trimmed_records = (
            list(SeqIO.parse(trimmed_path, fmt))
            if os.path.exists(trimmed_path)
            else []
        )
        untrimmed_records = (
            list(SeqIO.parse(untrimmed_path, fmt))
            if os.path.exists(untrimmed_path)
            else []
        )

        if not untrimmed_records:
            return trimmed_records

        # Second pass: reverse-complement untrimmed and swap primer orientation
        rc_records = _reverse_complement_records(untrimmed_records)
        rc_input = os.path.join(tmpdir, f"rc_input.{fmt}")
        _write_seqs(rc_records, rc_input, fmt)

        rc_trimmed_path = os.path.join(tmpdir, f"rc_trimmed.{fmt}")
        _run_cutprimers(
            rc_input,
            rc_trimmed_path,
            primer3 if primer3 else "",
            primer5,
            fmt,
        )
        rc_trimmed_records = (
            list(SeqIO.parse(rc_trimmed_path, fmt))
            if os.path.exists(rc_trimmed_path)
            else []
        )

        return trimmed_records + rc_trimmed_records


# ---------------------------------------------------------------------------
# 3. GC content
# ---------------------------------------------------------------------------

def gc_content(seq: str) -> float:
    """Return the GC content of *seq* as a fraction in [0, 1]."""
    if not seq:
        return 0.0
    upper = seq.upper()
    return (upper.count("G") + upper.count("C")) / len(upper)


# ---------------------------------------------------------------------------
# 4. DNA → binary transformation + hashing
# ---------------------------------------------------------------------------

_DNA_BINARY: Dict[str, str] = {"A": "00", "T": "01", "G": "10", "C": "11"}


def dna_to_binary(seq: str) -> str:
    """Encode *seq* as a binary string (A=00, T=01, G=10, C=11).

    Bases not in {A, T, G, C} are encoded as ``"00"`` (treated as A).
    """
    upper = seq.upper()
    return "".join(_DNA_BINARY.get(b, "00") for b in upper)


def dna_binary_hash(seq: str, hash_bits: int = 64) -> bytes:
    """
    Convert *seq* to its binary representation and return a hash of *hash_bits* bits.

    Supported sizes: 64, 128, 256.
    """
    binary_str = dna_to_binary(seq)
    encoded = binary_str.encode()
    return _hash_bytes(encoded, hash_bits)


# ---------------------------------------------------------------------------
# 5. Sequence length
# ---------------------------------------------------------------------------

def sequence_length(seq: str) -> int:
    """Return the number of bases in *seq*."""
    return len(seq)


# ---------------------------------------------------------------------------
# 6. K-mer hashing
# ---------------------------------------------------------------------------

def kmer_hashes(seq: str, k: int, hash_bits: int = 64) -> List[bytes]:
    """
    Compute a hash for each k-mer in *seq*.

    Parameters
    ----------
    seq:
        Nucleotide sequence string.
    k:
        k-mer length (e.g. 3, 5, 7).
    hash_bits:
        Bit-length of each hash: 64, 128, or 256.

    Returns
    -------
    List[bytes]
        One hash value per k-mer in the order they appear in *seq*.
    """
    upper = seq.upper()
    return [_hash_bytes(upper[i : i + k].encode(), hash_bits) for i in range(len(upper) - k + 1)]


# ---------------------------------------------------------------------------
# 7. Quality statistics
# ---------------------------------------------------------------------------

def quality_stats(quality_scores: List[int]) -> Dict[str, float]:
    """
    Compute descriptive statistics for a list of Phred quality scores.

    Returns
    -------
    dict with keys: ``mean``, ``median``, ``q25``, ``q75``.
    """
    if not quality_scores:
        return {"mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}

    sorted_scores = sorted(quality_scores)
    n = len(sorted_scores)

    def _quantile(data: List[int], p: float) -> float:
        """Linear-interpolation quantile (same as numpy default)."""
        idx = p * (len(data) - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= len(data):
            return float(data[-1])
        frac = idx - lo
        return data[lo] * (1 - frac) + data[hi] * frac

    return {
        "mean": statistics.mean(quality_scores),
        "median": statistics.median(sorted_scores),
        "q25": _quantile(sorted_scores, 0.25),
        "q75": _quantile(sorted_scores, 0.75),
    }


# ---------------------------------------------------------------------------
# 8. Quality values hashing
# ---------------------------------------------------------------------------

def quality_hash(quality_scores: List[int], hash_bits: int = 64) -> bytes:
    """
    Hash a list of Phred quality scores.

    The scores are packed as a byte string (each score is clipped to [0, 255])
    and hashed with the selected algorithm.

    Parameters
    ----------
    quality_scores:
        List of integer Phred quality values.
    hash_bits:
        Bit-length of the hash: 64, 128, or 256.

    Returns
    -------
    bytes
        Raw hash bytes of length ``hash_bits // 8``.
    """
    data = bytes(min(max(q, 0), 255) for q in quality_scores)
    return _hash_bytes(data, hash_bits)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_bytes(data: bytes, hash_bits: int) -> bytes:
    """Return a *hash_bits*-wide hash of *data*.

    Supported sizes: 64, 128, 256 bits.
    64  bits → first 8 bytes of SHA-256
    128 bits → MD5 digest (128 bits)
    256 bits → SHA-256 digest (256 bits)
    """
    if hash_bits == 64:
        return hashlib.sha256(data).digest()[:8]
    elif hash_bits == 128:
        return hashlib.md5(data).digest()  # noqa: S324 – non-crypto use
    elif hash_bits == 256:
        return hashlib.sha256(data).digest()
    else:
        raise ValueError(f"Unsupported hash_bits value: {hash_bits}. Choose 64, 128, or 256.")
