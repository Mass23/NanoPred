import os
import subprocess
import tempfile
from Bio import SeqIO


def _trim_record_forward(record, primer5, primer3, fmt):
    """Trim a single SeqRecord using simple substring matching (forward direction only).

    For Nanopore reads, which are always in the forward orientation:
    - primer5 (forward primer) is trimmed from the 5' end of the read.
    - primer3 (reverse primer) is trimmed from the 3' end of the read.
    """
    seq = str(record.seq).upper()
    p5 = primer5.upper() if primer5 else ''
    p3 = primer3.upper() if primer3 else ''

    start = 0
    end = len(seq)

    if p5:
        idx = seq.find(p5)
        if idx != -1:
            start = idx + len(p5)

    if p3:
        idx = seq.find(p3, start)
        if idx != -1:
            end = idx

    if start >= end:
        return None

    return record[start:end]


def _run_cutprimers(input_path, output_path, primer5, primer3, fmt):
    """Run cutPrimers (forward pass only) and return the path to trimmed sequences.

    Falls back to simple substring-based trimming if cutPrimers is not available.
    Returns the path to the trimmed output file.
    """
    try:
        cmd = [
            'cutPrimers',
            '--reads', input_path,
            '--primer5', primer5 or '',
            '--primer3', primer3 or '',
            '--output', output_path,
            '--outputDiscarded', '/dev/null',
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: simple substring trimming (forward direction only)
    records = list(SeqIO.parse(input_path, fmt))
    trimmed = []
    for rec in records:
        new_rec = _trim_record_forward(rec, primer5, primer3, fmt)
        if new_rec is not None:
            trimmed.append(new_rec)
    SeqIO.write(trimmed, output_path, fmt)
    return output_path


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
        _run_cutprimers(input_fasta, trimmed_path, primer5, primer3, fmt="fasta")
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
        _run_cutprimers(input_fastq, trimmed_path, primer5, primer3, fmt="fastq")
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
