"""
data_creation.py - Generate training dataset of sequence pairs with computed metrics.

Creates a large all_pairs_data.csv file where each row is a pair of sequences
with real_percent_identity (from alignment) as the target variable and various
sequence metrics as features.
"""

import itertools
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


def quality_hash(quality_scores: list, num_bits: int) -> np.ndarray:
    """Hash quality score array to num_bits. Phred scores are capped at 93
    (the maximum valid Phred+33 FASTQ value) before encoding."""
    data = bytes(min(q, 93) for q in quality_scores)
    return _hash_to_bits(data, num_bits)


# ---------------------------------------------------------------------------
# K-mer presence sketching
# ---------------------------------------------------------------------------

def _build_kmer_vocabulary(k: int) -> dict:
    """Return a mapping from every ACGT k-mer to its index (alphabetical order)."""
    return {''.join(p): i for i, p in enumerate(itertools.product('ACGT', repeat=k))}


# Pre-built vocabularies for k = 3 (64 entries) and k = 5 (1024 entries)
_KMER_VOCABULARIES: dict[int, dict[str, int]] = {k: _build_kmer_vocabulary(k) for k in (3, 5)}


def _splitmix64(x: int) -> int:
    """SplitMix64 hash mixer — fast deterministic 64-bit hash with good avalanche."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x &= 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return x & 0xFFFFFFFFFFFFFFFF


def kmer_presence_sketch(seq: str, k: int, num_bits: int, hashes_per_kmer: int = 2) -> np.ndarray:
    """Compute a hashed presence sketch (bitset) for the k-mer spectrum of *seq*.

    For each distinct k-mer present in *seq*, ``hashes_per_kmer`` bit positions
    are derived from the k-mer's vocabulary index using the SplitMix64 mixer and
    set to 1.  The resulting ``num_bits``-element uint8 array is a compact,
    fixed-size representation of the k-mer presence set.

    Jaccard similarity between two such sketches approximates set Jaccard on the
    underlying k-mer presence sets and can be computed in O(num_bits) time,
    independent of sequence length or vocabulary size.

    Args:
        seq:            DNA sequence string.
        k:              k-mer length.  Must be 3 or 5 (values in _KMER_VOCABULARIES).
        num_bits:       Sketch size in bits (e.g. 64, 128, 256, 512).
        hashes_per_kmer: Number of bit positions set per present k-mer (default 2).

    Returns:
        uint8 array of length *num_bits* (each element 0 or 1).

    Raises:
        KeyError: If *k* is not a supported vocabulary size (currently 3 or 5).
    """
    vocab = _KMER_VOCABULARIES[k]
    sketch = np.zeros(num_bits, dtype=np.uint8)
    seq_upper = seq.upper()
    present: set[int] = set()
    for i in range(len(seq_upper) - k + 1):
        idx = vocab.get(seq_upper[i:i + k])
        if idx is not None:
            present.add(idx)
    for idx in present:
        for s in range(hashes_per_kmer):
            h = _splitmix64(idx + s * 0x9E3779B97F4A7C15)
            sketch[h % num_bits] = 1
    return sketch

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
    - quality_hash_64/128/256  (bitset sketches)
    - kmer_{k}_sketch_{b}  for k in {3,5}, b in {64,128,256,512}
      (hashed presence bitset sketches; use jaccard_similarity() for pair comparison)
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

    # Quality hash (bitset sketch)
    for bits in (64, 128, 256):
        metrics[f'quality_hash_{bits}'] = quality_hash(quality_scores, bits)

    # k-mer presence sketches (hashed bitsets; k=3 and k=5, four sketch sizes)
    for k in (3, 5):
        for bits in (64, 128, 256, 512):
            metrics[f'kmer_{k}_sketch_{bits}'] = kmer_presence_sketch(seq, k, bits)

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
    Returns flat feature dict (no sketch arrays, only scalars).

    K-mer features:
      kmer_{k}_hashjaccard_{b} — Jaccard on hashed presence sketches for
                                  k in {3,5}, b in {64,128,256,512}
    """
    features = {}

    scalar_keys = [
        'length',
        'quality_mean', 'quality_median', 'quality_q25', 'quality_q75',
        'gc_content',
    ]
    for key in scalar_keys:
        features.update(_scalar_features(key, float(m1[key]), float(m2[key])))

    # Quality hash Jaccard
    for bits in (64, 128, 256):
        a = m1[f'quality_hash_{bits}']
        b = m2[f'quality_hash_{bits}']
        features[f'quality_jaccard_{bits}'] = jaccard_similarity(a, b)

    # K-mer hashed presence sketch Jaccard
    for k in (3, 5):
        for bits in (64, 128, 256, 512):
            a = m1[f'kmer_{k}_sketch_{bits}']
            b = m2[f'kmer_{k}_sketch_{bits}']
            features[f'kmer_{k}_hashjaccard_{bits}'] = jaccard_similarity(a, b)

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
    shard_id: int = 0,
    num_shards: int = 1,
) -> None:
    """
    Generate a dataset of num_pairs sequence pairs with computed features.

    Sharding support:
        When num_shards > 1, this function generates only the pairs assigned to
        *shard_id* (0-indexed) and writes them to a shard-specific file:
            <base>.part{shard_id}<ext>   (e.g. all_pairs_data.part3.csv)
        Each shard uses an independent RNG seeded with ``seed + shard_id`` so
        results are fully reproducible regardless of execution order.
        Pairs are sampled on the fly — no large list is pre-allocated.

    Args:
        fasta_paths: Path (str) or list of paths to input FASTA file(s).
                     When multiple files are provided, all sequences are pooled
                     together before pair generation.
        num_pairs:   Total number of pairs across all shards (e.g. 500_000).
        output_csv:  Output CSV path.  When num_shards > 1, the shard index is
                     inserted before the file extension.
        primer5:     Forward primer sequence (optional).
        primer3:     Reverse primer sequence (optional).
        seed:        Global random seed.  Per-shard seed = seed + shard_id.
        chunk_size:  Number of pairs to process per batch before writing.
        shard_id:    Index of this shard (0 .. num_shards-1).  Default 0.
        num_shards:  Total number of shards.  Default 1 (no sharding).
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, num_shards-1], got {shard_id}/{num_shards}")

    # Per-shard RNG seeding — use a hash to ensure well-separated seeds even
    # for consecutive shard IDs rather than simple addition.
    shard_seed = hash((seed, shard_id)) & 0xFFFFFFFF
    rng = np.random.default_rng(shard_seed)

    # Number of pairs assigned to this shard
    base_pairs = num_pairs // num_shards
    extra = num_pairs % num_shards
    shard_pairs = base_pairs + (1 if shard_id < extra else 0)

    # Shard-specific output path
    if num_shards > 1:
        base, ext = os.path.splitext(output_csv)
        shard_output = f'{base}.part{shard_id}{ext}'
    else:
        shard_output = output_csv

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

    n_seq = len(sequences)

    print(
        f"Generating {shard_pairs} pairs "
        f"(shard {shard_id}/{num_shards}, seed {shard_seed}) → {shard_output}"
    )

    first_chunk = True
    rows_written = 0
    pairs_generated = 0

    with tqdm(total=shard_pairs, desc="Generating pairs", unit="pair") as pbar:
        while pairs_generated < shard_pairs:
            this_chunk = min(chunk_size, shard_pairs - pairs_generated)
            chunk_rows = []

            for _ in range(this_chunk):
                i = rng.integers(0, n_seq)
                j = rng.integers(0, n_seq)
                seq1_raw = sequences[i][1]
                seq2_raw = sequences[j][1]

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
                    import warnings
                    warnings.warn(f"Skipping pair due to error: {exc}")
                    continue

            if chunk_rows:
                df = pd.DataFrame(chunk_rows)
                df.to_csv(
                    shard_output,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False,
                )
                first_chunk = False
                rows_written += len(chunk_rows)

            pairs_generated += this_chunk
            pbar.update(this_chunk)

    print(f"Done. Wrote {rows_written} rows to {shard_output}.")


# ---------------------------------------------------------------------------
# Shard merging
# ---------------------------------------------------------------------------

def merge_shards(
    output_csv: str,
    num_shards: int,
    seed: int = 42,
    keep_shards: bool = False,
) -> None:
    """Merge all shard CSV files into a single shuffled output file.

    Reads ``<base>.part{0..num_shards-1}<ext>``, concatenates them,
    shuffles the rows (using *seed* for reproducibility), writes the
    result to *output_csv*, and removes the shard files unless
    *keep_shards* is True.

    Args:
        output_csv:  Final output CSV path (e.g. ``all_pairs_data.csv``).
        num_shards:  Total number of shard files expected.
        seed:        Random seed used for shuffling (default 42).
        keep_shards: When True, shard part files are not deleted after
                     merging (default False).
    """
    base, ext = os.path.splitext(output_csv)
    shard_paths = [f'{base}.part{i}{ext}' for i in range(num_shards)]

    missing = [p for p in shard_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Cannot merge: {len(missing)} shard file(s) not found: {missing[:5]}"
        )

    print(f"Merging {num_shards} shard(s) into {output_csv} ...")
    frames = [pd.read_csv(p) for p in shard_paths]
    df = pd.concat(frames, ignore_index=True)

    print(f"  Shuffling {len(df)} rows (seed={seed}) ...")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"  Wrote {len(df)} rows to {output_csv}.")

    if not keep_shards:
        for p in shard_paths:
            os.remove(p)
        print(f"  Deleted {len(shard_paths)} shard file(s).")
