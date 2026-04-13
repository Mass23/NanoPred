"""
data_creation.py - Generate training dataset of sequence pairs with computed metrics.

Creates a large all_pairs_data.csv file where each row is a pair of sequences
with real_percent_identity (from alignment) as the target variable and various
sequence metrics as features.
"""

import os
import random
import hashlib
import subprocess
import tempfile
from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from tqdm import tqdm


# ---------------------------------------------------------------------------
# FASTA loading
# ---------------------------------------------------------------------------

def rna_to_dna(sequence: str) -> str:
    """Convert an RNA sequence to DNA by replacing 'U' with 'T'.

    Also normalises the sequence to uppercase so that downstream processing
    works consistently regardless of the input case.
    """
    seq = sequence.upper()
    return seq.replace('U', 'T')


def load_fasta(fasta_path: str) -> list:
    """Load sequences from a FASTA file. Returns list of (id, sequence) tuples.

    RNA sequences (containing 'U') are automatically converted to DNA.
    All sequences are normalised to uppercase.
    """
    records = []
    converted = 0
    with open(fasta_path, 'r') as fh:
        for record in SeqIO.parse(fh, 'fasta'):
            raw = str(record.seq)
            if 'U' in raw.upper():
                converted += 1
            seq = rna_to_dna(raw)
            if len(seq) > 0:
                records.append((record.id, seq))
    if converted:
        print(f"  Converted {converted} RNA sequence(s) to DNA in {fasta_path}.")
    return records


# ---------------------------------------------------------------------------
# Pair creation
# ---------------------------------------------------------------------------

def create_sequence_pairs(sequences: list, num_pairs: int, seed: int = 42) -> list:
    """
    Randomly sample num_pairs pairs from the list of (id, seq) tuples.
    Returns list of ((id1, seq1), (id2, seq2)) pairs.
    """
    rng = random.Random(seed)
    n = len(sequences)
    pairs = []
    for _ in range(num_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        pairs.append((sequences[i], sequences[j]))
    return pairs


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_sequences(seq1: str, seq2: str) -> float:
    """
    Pairwise global alignment using Biopython PairwiseAligner.
    Returns percent identity in [0, 100].
    """
    if not seq1 or not seq2:
        return 0.0

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    alignments = aligner.align(seq1, seq2)
    best = next(iter(alignments), None)
    if best is None:
        return 0.0

    aligned_len = max(len(seq1), len(seq2))
    if aligned_len == 0:
        return 0.0

    return float(best.score) / aligned_len * 100.0


# ---------------------------------------------------------------------------
# CutPrimers integration
# ---------------------------------------------------------------------------

def _run_cutprimers(input_fasta: str, output_fasta: str, primer5: str, primer3: str) -> str:
    """
    Run cutPrimers on input_fasta and write trimmed sequences to output_fasta.
    Falls back to simple string-based trimming if cutPrimers is not available.
    Returns path to trimmed FASTA (may be input_fasta if trimming failed).
    """
    try:
        cmd = [
            'cutPrimers',
            '--reads', input_fasta,
            '--primer5', primer5,
            '--primer3', primer3,
            '--output', output_fasta,
            '--outputDiscarded', '/dev/null',
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            # Allow up to 60 s per invocation; individual files are small
            # (single sequences) so this is more than sufficient.
            timeout=60,
        )
        if result.returncode == 0 and os.path.exists(output_fasta):
            return output_fasta
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: simple substring trimming
    records = list(SeqIO.parse(input_fasta, 'fasta'))
    trimmed = []
    for rec in records:
        seq = str(rec.seq).upper()
        p5 = primer5.upper()
        p3 = primer3.upper()
        start = seq.find(p5)
        if start != -1:
            seq = seq[start + len(p5):]
        end = seq.find(p3)
        if end != -1:
            seq = seq[:end]
        rec.seq = type(rec.seq)(seq)
        trimmed.append(rec)
    SeqIO.write(trimmed, output_fasta, 'fasta')
    return output_fasta


def trim_sequence_with_primers(seq: str, primer5: str, primer3: str) -> str:
    """
    Trim a single sequence using primer locations (simple substring match).
    Returns the trimmed sequence or original if primers not found.
    """
    if not primer5 and not primer3:
        return seq

    seq_upper = seq.upper()
    p5 = primer5.upper() if primer5 else ''
    p3 = primer3.upper() if primer3 else ''

    start = 0
    end = len(seq_upper)

    if p5:
        idx = seq_upper.find(p5)
        if idx != -1:
            start = idx + len(p5)

    if p3:
        idx = seq_upper.find(p3, start)
        if idx != -1:
            end = idx

    trimmed = seq[start:end]
    return trimmed if trimmed else seq


# ---------------------------------------------------------------------------
# Error simulation
# ---------------------------------------------------------------------------

BASES = ['A', 'T', 'G', 'C']
SUBSTITUTIONS = {
    'A': ['T', 'G', 'C'],
    'T': ['A', 'G', 'C'],
    'G': ['A', 'T', 'C'],
    'C': ['A', 'T', 'G'],
    'N': ['A', 'T', 'G', 'C'],
}


def process_sequence(seq: str, primer5: str = '', primer3: str = '',
                     rng: np.random.Generator = None) -> tuple:
    """
    Trim primers from seq and simulate sequencing errors.

    Steps:
    1. Trim primers using simple substring matching.
    2. Draw average Phred score uniformly from Q10 to Q50.
    3. Compute per-base error probability: P = 10^(-Q/10).
    4. Randomly introduce substitution errors.
    5. Generate per-base quality scores (Phred) as Gaussian around average.

    Returns:
        (processed_seq, quality_scores, avg_phred)
        - processed_seq: string with errors applied
        - quality_scores: list of int Phred scores per base
        - avg_phred: float average Phred score drawn
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: primer trimming
    trimmed = trim_sequence_with_primers(seq, primer5, primer3)
    if len(trimmed) == 0:
        trimmed = seq

    # Step 2: draw average Phred score
    avg_phred = rng.uniform(10.0, 50.0)

    # Step 3: per-base error probability
    p_error = 10 ** (-avg_phred / 10.0)

    # Step 4: introduce substitution errors
    seq_list = list(trimmed)
    for i, base in enumerate(seq_list):
        if rng.random() < p_error:
            b = base if base in SUBSTITUTIONS else 'N'
            seq_list[i] = rng.choice(SUBSTITUTIONS[b])

    processed_seq = ''.join(seq_list)

    # Step 5: generate per-base Phred quality scores
    std = max(1.0, avg_phred * 0.1)
    raw_scores = rng.normal(loc=avg_phred, scale=std, size=len(processed_seq))
    quality_scores = np.clip(np.round(raw_scores), 0, 93).astype(int).tolist()

    return processed_seq, quality_scores, avg_phred


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def _mmh3_like_hash(data: bytes, seed: int = 0) -> int:
    """Deterministic hash using SHA-256 with seed for reproducibility."""
    h = hashlib.sha256(seed.to_bytes(4, 'big') + data)
    return int.from_bytes(h.digest(), 'big')


def _hash_to_bits(data: bytes, num_bits: int) -> np.ndarray:
    """Convert data to a bit array of length num_bits using multiple hash passes."""
    bits = []
    seed = 0
    while len(bits) < num_bits:
        h = hashlib.sha256(seed.to_bytes(4, 'big') + data).digest()
        for byte in h:
            for shift in range(8):
                bits.append((byte >> shift) & 1)
        seed += 1
    return np.array(bits[:num_bits], dtype=np.uint8)


def dna_binary_hash(seq: str, num_bits: int) -> np.ndarray:
    """
    Convert DNA sequence to binary then hash to num_bits.
    Mapping: A=00, T=01, G=10, C=11.  Ambiguous/unknown bases (e.g. N) are
    encoded as 00 (same as A) because they represent masked positions and carry
    no distinct information in the binary representation.
    """
    mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
    binary_str = ''.join(mapping.get(b, '00') for b in seq.upper())
    return _hash_to_bits(binary_str.encode(), num_bits)


def quality_hash(quality_scores: list, num_bits: int) -> np.ndarray:
    """Hash quality score array to num_bits. Phred scores are capped at 93
    (the maximum valid Phred+33 FASTQ value) before encoding."""
    data = bytes(min(q, 93) for q in quality_scores)
    return _hash_to_bits(data, num_bits)


# ---------------------------------------------------------------------------
# K-mer spectrum -> hashed sketch (fixed vocabulary)
# ---------------------------------------------------------------------------

_BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Pre-generated (deterministic) coefficients for fast MinHash
# Using numpy RNG with a fixed seed for reproducibility across runs.
# We generate up to 256 so existing code can still request 64/128/256.
_MINHASH_MAX_PERM = 256
_rng_mh = np.random.default_rng(12345)
# Ensure 'a' is odd to improve mixing in mod 2^64 arithmetic
_MINHASH_A = (_rng_mh.integers(1, np.iinfo(np.uint64).max, size=_MINHASH_MAX_PERM, dtype=np.uint64) | np.uint64(1))
_MINHASH_B = _rng_mh.integers(0, np.iinfo(np.uint64).max, size=_MINHASH_MAX_PERM, dtype=np.uint64)

def _kmer_set_indices(seq: str, k: int) -> np.ndarray:
    """
    Return unique indices (base-4 encoding) of all A/C/G/T k-mers present in seq.
    Comparable across sequences without storing a 4^k dense vector.
    """
    seq = seq.upper()
    if len(seq) < k:
        return np.array([], dtype=np.uint32)

    present = set()
    for i in range(len(seq) - k + 1):
        idx = 0
        ok = True
        for ch in seq[i:i+k]:
            v = _BASE_TO_INT.get(ch)
            if v is None:
                ok = False
                break
            idx = (idx * 4) + v
        if ok:
            present.add(idx)

    if not present:
        return np.array([], dtype=np.uint32)

    return np.fromiter(present, dtype=np.uint32)

def _minhash_signature_from_indices(indices: np.ndarray, num_perm: int) -> np.ndarray:
    """
    Fast MinHash signature for a set of integer indices using arithmetic hashing:
      h_i(x) = (a_i * x + b_i) mod 2^64
    Returns uint64 array length num_perm.
    """
    if num_perm > _MINHASH_MAX_PERM:
        raise ValueError(f"num_perm={num_perm} exceeds max {_MINHASH_MAX_PERM}")

    if indices.size == 0:
        return np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)

    x = indices.astype(np.uint64)  # shape (m,)
    a = _MINHASH_A[:num_perm]      # shape (num_perm,)
    b = _MINHASH_B[:num_perm]      # shape (num_perm,)

    # Compute hashes for all perms and all elements:
    # H has shape (num_perm, m). Then signature is min across m for each perm.
    # This is vectorized; much faster than per-seed SHA.
    H = (a[:, None] * x[None, :] + b[:, None]).astype(np.uint64)
    sig = H.min(axis=1)
    return sig

def minhash_jaccard(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Jaccard estimate from MinHash signatures."""
    if sig1.shape != sig2.shape or sig1.size == 0:
        return 0.0
    return float(np.mean(sig1 == sig2))

def kmer_hash(seq: str, k: int, num_bits: int) -> np.ndarray:
    """
    Return MinHash signature length=num_bits for the explicit k-mer set.
    'num_bits' kept for compatibility with existing naming (64/128/256).
    """
    indices = _kmer_set_indices(seq, k)
    return _minhash_signature_from_indices(indices, num_perm=num_bits)

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(seq: str, quality_scores: list) -> dict:
    """
    Compute all metrics for a single processed sequence.

    Returns dict with:
    - length
    - quality_mean, quality_median, quality_q25, quality_q75
    - gc_content
    - dna_binary_hash_64/128/256
    - quality_hash_64/128/256
    - kmer_3_hash_64/128/256
    - kmer_5_hash_64/128/256
    - kmer_7_hash_64/128/256
    """
    metrics = {}

    # Length
    metrics['length'] = len(seq)

    # Quality stats
    if quality_scores:
        q_arr = np.array(quality_scores, dtype=float)
        metrics['quality_mean'] = float(np.mean(q_arr))
        metrics['quality_median'] = float(np.median(q_arr))
        metrics['quality_q25'] = float(np.percentile(q_arr, 25))
        metrics['quality_q75'] = float(np.percentile(q_arr, 75))
    else:
        metrics['quality_mean'] = 0.0
        metrics['quality_median'] = 0.0
        metrics['quality_q25'] = 0.0
        metrics['quality_q75'] = 0.0

    # GC content
    gc = seq.count('G') + seq.count('C')
    metrics['gc_content'] = (gc / len(seq) * 100.0) if seq else 0.0

    # Hash bit arrays (stored as np.ndarray, used only internally for Jaccard)
    for bits in (64, 128, 256):
        metrics[f'dna_binary_hash_{bits}'] = dna_binary_hash(seq, bits)
        metrics[f'quality_hash_{bits}'] = quality_hash(quality_scores, bits)
        for k in (3, 5, 7):
            metrics[f'kmer_{k}_hash_{bits}'] = kmer_hash(seq, k, bits)

    return metrics


# ---------------------------------------------------------------------------
# Jaccard similarity between bit arrays
# ---------------------------------------------------------------------------

def jaccard_similarity(bits1: np.ndarray, bits2: np.ndarray) -> float:
    """Compute Jaccard similarity between two binary bit arrays."""
    union = np.sum(np.logical_or(bits1, bits2))
    if union == 0:
        return 0.0
    intersection = np.sum(np.logical_and(bits1, bits2))
    return float(intersection) / float(union)


# ---------------------------------------------------------------------------
# Pair feature computation
# ---------------------------------------------------------------------------

def _scalar_features(name: str, v1: float, v2: float) -> dict:
    """Compute min, max, diff, mean for a pair of scalar values."""
    lo, hi = min(v1, v2), max(v1, v2)
    return {
        f'{name}_min': lo,
        f'{name}_max': hi,
        f'{name}_diff': hi - lo,
        f'{name}_mean': (v1 + v2) / 2.0,
    }


def compute_pair_features(m1: dict, m2: dict) -> dict:
    """
    Compute pair-wise features from two metric dicts.
    Returns flat feature dict (no hash arrays, only scalars).
    """
    features = {}

    # Scalar comparisons: length, quality stats, gc_content
    scalar_keys = [
        'length',
        'quality_mean', 'quality_median', 'quality_q25', 'quality_q75',
        'gc_content',
    ]
    for key in scalar_keys:
        features.update(_scalar_features(key, float(m1[key]), float(m2[key])))

    # Jaccard comparisons for hash metrics
    hash_groups = [
        ('dna_binary', 'dna_binary_hash', 'bit'),
        ('quality', 'quality_hash', 'bit'),
        ('kmer_3', 'kmer_3_hash', 'minhash'),
        ('kmer_5', 'kmer_5_hash', 'minhash'),
        ('kmer_7', 'kmer_7_hash', 'minhash'),
    ]
    for label, key_prefix, kind in hash_groups:
        for bits in (64, 128, 256):
            col = f'{label}_jaccard_{bits}'
            a = m1[f'{key_prefix}_{bits}']
            b = m2[f'{key_prefix}_{bits}']
            if kind == 'minhash':
                features[col] = minhash_jaccard(a, b)
            else:
                features[col] = jaccard_similarity(a, b)
    return features


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    fasta_paths: Union[str, List[str]],
    num_pairs: int,
    output_csv: str,
    primer5: str = '',
    primer3: str = '',
    seed: int = 42,
    chunk_size: int = 1000,
) -> None:
    """
    Generate a dataset of num_pairs sequence pairs with computed features.

    Args:
        fasta_paths: Path (str) or list of paths to input FASTA file(s).
                     When multiple files are provided, all sequences are pooled
                     together before pair generation.
        num_pairs:   Number of pairs to generate (e.g. 500_000).
        output_csv:  Output CSV path.
        primer5:     Forward primer sequence (optional).
        primer3:     Reverse primer sequence (optional).
        seed:        Random seed for reproducibility.
        chunk_size:  Number of pairs to process per batch before writing.
    """
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # Accept a single path string for backward compatibility.
    if isinstance(fasta_paths, str):
        fasta_paths = [fasta_paths]

    sequences = []
    for fasta_path in fasta_paths:
        print(f"Loading sequences from {fasta_path} ...")
        loaded = load_fasta(fasta_path)
        print(f"  Loaded {len(loaded)} sequences.")
        sequences.extend(loaded)

    if not sequences:
        raise ValueError(f"No sequences found in {fasta_paths}")
    print(f"Total sequences loaded: {len(sequences)}")

    print(f"Creating {num_pairs} pairs ...")
    pairs = create_sequence_pairs(sequences, num_pairs, seed=seed)

    first_chunk = True
    rows_written = 0

    with tqdm(total=num_pairs, desc="Generating pairs", unit="pair") as pbar:
        for chunk_start in range(0, num_pairs, chunk_size):
            chunk = pairs[chunk_start: chunk_start + chunk_size]
            chunk_rows = []

            for (_, seq1_raw), (_, seq2_raw) in chunk:
                try:
                    # Process both sequences (trim + error simulation)
                    seq1, q1, _ = process_sequence(seq1_raw, primer5, primer3, rng)
                    seq2, q2, _ = process_sequence(seq2_raw, primer5, primer3, rng)

                    if len(seq1) == 0 or len(seq2) == 0:
                        continue

                    # Alignment-based percent identity (target variable)
                    pct_id = align_sequences(seq1, seq2)

                    # Per-sequence metrics
                    m1 = compute_metrics(seq1, q1)
                    m2 = compute_metrics(seq2, q2)

                    # Pair features
                    pair_feats = compute_pair_features(m1, m2)

                    row = {'real_percent_identity': pct_id}
                    row.update(pair_feats)
                    chunk_rows.append(row)

                except (ValueError, TypeError, ZeroDivisionError, StopIteration) as exc:
                    # Skip pairs that fail due to empty/invalid sequences or
                    # alignment edge cases; log the reason for debugging.
                    import warnings
                    warnings.warn(f"Skipping pair due to error: {exc}")
                    continue

            if chunk_rows:
                df = pd.DataFrame(chunk_rows)
                df.to_csv(
                    output_csv,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False,
                )
                first_chunk = False
                rows_written += len(chunk_rows)

            pbar.update(len(chunk))

    print(f"Done. Wrote {rows_written} rows to {output_csv}.")
