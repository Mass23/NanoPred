import os
import subprocess
import tempfile
from Bio import SeqIO


def _run_cutadapt(input_path, output_path, primer5, primer3):
    """Trim primers from a Nanopore FASTQ/FASTA file using cutadapt.

    For Nanopore single-end reads:
    - ``-g`` trims the forward primer from the 5' end.
    - ``-a`` trims the reverse primer from the 3' end.

    Reads where a primer is not found are kept unchanged (no
    ``--discard-untrimmed``). Error tolerance is set to 20 % to handle
    Nanopore's high base-error rate.

    Args:
        input_path:  Path to the input FASTA/FASTQ file.
        output_path: Path for the trimmed output file.
        primer5:     Forward primer sequence (or None/empty to skip 5' trimming).
        primer3:     Reverse primer sequence (or None/empty to skip 3' trimming).

    Raises:
        RuntimeError: If cutadapt is not installed or exits with an error.
    """
    cmd = ["cutadapt"]
    if primer5:
        cmd.extend(["-g", primer5])
    if primer3:
        cmd.extend(["-a", primer3])
    cmd.extend(["-e", "0.2", "-o", output_path, input_path])

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


def trim_primers_fasta(input_fasta, output_fasta, primer5, primer3=None):
    """Trim primers from Nanopore FASTA reads (forward direction only).

    Nanopore reads are always in the forward orientation, so no
    reverse-complement pass is performed. Both primer5 (forward primer) and
    primer3 (reverse primer) are used to cut the read from each end.

    Args:
        input_fasta:  Path to input FASTA file.
        output_fasta: Path to write trimmed FASTA records (optional).
        primer5:      Forward primer sequence to trim from the 5' end.
        primer3:      Reverse primer sequence to trim from the 3' end (optional).

    Returns:
        List of trimmed SeqRecord objects.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trimmed_path = os.path.join(tmpdir, "trimmed.fasta")
        _run_cutadapt(input_fasta, trimmed_path, primer5, primer3)
        trimmed_records = list(SeqIO.parse(trimmed_path, "fasta"))

        if output_fasta:
            SeqIO.write(trimmed_records, output_fasta, "fasta")
        return trimmed_records


def trim_primers_fastq(input_fastq, output_fastq, primer5, primer3=None):
    """Trim primers from Nanopore FASTQ reads (forward direction only).

    Nanopore reads are always in the forward orientation, so no
    reverse-complement pass is performed. Both primer5 (forward primer) and
    primer3 (reverse primer) are used to cut the read from each end.

    Args:
        input_fastq:  Path to input FASTQ file.
        output_fastq: Path to write trimmed FASTQ records (optional).
        primer5:      Forward primer sequence to trim from the 5' end.
        primer3:      Reverse primer sequence to trim from the 3' end (optional).

    Returns:
        List of trimmed SeqRecord objects.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trimmed_path = os.path.join(tmpdir, "trimmed.fastq")
        _run_cutadapt(input_fastq, trimmed_path, primer5, primer3)
        trimmed_records = list(SeqIO.parse(trimmed_path, "fastq"))

        if output_fastq:
            SeqIO.write(trimmed_records, output_fastq, "fastq")
        return trimmed_records


def process_fasta_for_benchmark(input_fasta, primer5, primer3):
    """Trim primers from a FASTA file and return trimmed records (for benchmarking).

    Args:
        input_fasta: Path to input FASTA file.
        primer5:     Forward primer sequence.
        primer3:     Reverse primer sequence.

    Returns:
        List of trimmed SeqRecord objects.
    """
    return trim_primers_fasta(input_fasta, None, primer5, primer3)
