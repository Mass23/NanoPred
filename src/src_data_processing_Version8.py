def trim_primers_bidirectional_fasta(input_fasta, output_fasta, primer5, primer3=None):
    """
    Trim with cutPrimers. If primers not found, reverse-complement sequence and swap primer sides.
    Returns: list of all successfully trimmed SeqRecords (FASTA mode).
    """
    from Bio import SeqIO
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1st pass: as-is
        trimmed_path = os.path.join(tmpdir, "trimmed.fasta")
        utr_path = os.path.join(tmpdir, "untrimmed.fasta")
        t_path, u_path = run_cutprimers(input_fasta, trimmed_path, primer5, primer3, fmt="fasta")
        trimmed_records = list(SeqIO.parse(t_path, "fasta"))
        untrimmed_records = list(SeqIO.parse(u_path, "fasta")) if os.path.exists(u_path) else []

        if not untrimmed_records:
            return trimmed_records

        # 2nd pass: reverse-complement untrimmed, swap primer orientation
        rc_untrimmed_path = os.path.join(tmpdir, "untrimmed_rc.fasta")
        rc_records = reverse_complement_records(untrimmed_records)
        write_seqs(rc_records, rc_untrimmed_path, "fasta")
        rc_trimmed_path = os.path.join(tmpdir, "trimmed_rc.fasta")

        # Now: primer at 5' is original primer3, and at 3' is original primer5
        t2_path, _ = run_cutprimers(
            rc_untrimmed_path,
            rc_trimmed_path,
            primer3 if primer3 else "",
            primer5 if primer5 else "",
            fmt="fasta"
        )
        trimmed_rc_records = list(SeqIO.parse(t2_path, "fasta")) if os.path.exists(t2_path) else []

        # Combine both sets of trimmed records
        out_records = trimmed_records + trimmed_rc_records
        # Optionally: write to output_fasta
        if output_fasta:
            write_seqs(out_records, output_fasta, "fasta")
        return out_records

def trim_primers_bidirectional_fastq(input_fastq, output_fastq, primer5, primer3=None):
    """
    Same logic as above, but works with FASTQ.
    """
    from Bio import SeqIO
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1st pass: as-is
        trimmed_path = os.path.join(tmpdir, "trimmed.fastq")
        utr_path = os.path.join(tmpdir, "untrimmed.fastq")
        t_path, u_path = run_cutprimers(input_fastq, trimmed_path, primer5, primer3, fmt="fastq")
        trimmed_records = list(SeqIO.parse(t_path, "fastq"))
        untrimmed_records = list(SeqIO.parse(u_path, "fastq")) if os.path.exists(u_path) else []

        if not untrimmed_records:
            return trimmed_records

        # 2nd pass: reverse-complement untrimmed, swap primer orientation
        rc_untrimmed_path = os.path.join(tmpdir, "untrimmed_rc.fastq")
        rc_records = reverse_complement_records(untrimmed_records)
        write_seqs(rc_records, rc_untrimmed_path, "fastq")
        rc_trimmed_path = os.path.join(tmpdir, "trimmed_rc.fastq")

        # Now: primer at 5' is original primer3, and at 3' is original primer5
        t2_path, _ = run_cutprimers(
            rc_untrimmed_path,
            rc_trimmed_path,
            primer3 if primer3 else "",
            primer5 if primer5 else "",
            fmt="fastq"
        )
        trimmed_rc_records = list(SeqIO.parse(t2_path, "fastq")) if os.path.exists(t2_path) else []

        out_records = trimmed_records + trimmed_rc_records
        if output_fastq:
            write_seqs(out_records, output_fastq, "fastq")
        return out_records