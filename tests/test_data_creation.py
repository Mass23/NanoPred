"""
Tests for the balanced, GC-bin-guided dataset generation in src/data_creation.py.

Covers:
    - build_gc_bins: correct binning, empty-bin omission, boundary sequences
    - generate_dataset: balanced output (<=85 / >85 rows), odd num_pairs guard,
      correct CSV columns, shard-output filename, GC-bin-guided sampling integration
"""

import os
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

# Ensure the repo root is on the path so we can import src.data_creation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_creation import (
    align_sequences,
    build_gc_bins,
    generate_dataset,
    GC_BINS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fasta(path: str, records: list) -> None:
    """Write a minimal FASTA file from a list of (id, seq) tuples."""
    with open(path, "w") as fh:
        for rec_id, seq in records:
            fh.write(f">{rec_id}\n{seq}\n")


# ---------------------------------------------------------------------------
# build_gc_bins tests
# ---------------------------------------------------------------------------

class TestBuildGcBins(unittest.TestCase):

    def test_all_sequences_assigned(self):
        seqs = [("s1", "AAAA"), ("s2", "GCGC"), ("s3", "TTTT"), ("s4", "GCAA")]
        bins = build_gc_bins(seqs, n_bins=10)
        # Every index should appear in exactly one bin
        all_indices = sorted(idx for b in bins for idx in b)
        self.assertEqual(all_indices, list(range(len(seqs))))

    def test_no_empty_bins_returned(self):
        seqs = [("s1", "AAAA"), ("s2", "TTTT")]
        bins = build_gc_bins(seqs, n_bins=10)
        for b in bins:
            self.assertGreater(len(b), 0)

    def test_gc_zero_sequence_in_first_bin(self):
        seqs = [("pure_at", "AAATTT")]
        bins = build_gc_bins(seqs, n_bins=10)
        self.assertEqual(len(bins), 1)
        self.assertIn(0, bins[0])

    def test_gc_100_sequence_in_last_bin(self):
        seqs = [("pure_gc", "GCGCGC")]
        bins = build_gc_bins(seqs, n_bins=10)
        self.assertEqual(len(bins), 1)
        self.assertIn(0, bins[0])

    def test_correct_bin_assignment(self):
        # 25 % GC → bin index 2 out of 10
        seqs = [("s", "ACTTTTTT")]  # 2 GC out of 8 bases = 25 %
        bins = build_gc_bins(seqs, n_bins=10)
        self.assertEqual(len(bins), 1)
        self.assertIn(0, bins[0])

    def test_empty_sequence_skipped(self):
        seqs = [("empty", ""), ("real", "ACGT")]
        bins = build_gc_bins(seqs, n_bins=10)
        all_indices = [idx for b in bins for idx in b]
        self.assertNotIn(0, all_indices)   # empty seq skipped
        self.assertIn(1, all_indices)

    def test_default_n_bins(self):
        seqs = [("s", "ACGT")]
        bins_default = build_gc_bins(seqs)
        bins_explicit = build_gc_bins(seqs, n_bins=GC_BINS)
        self.assertEqual(bins_default, bins_explicit)


# ---------------------------------------------------------------------------
# generate_dataset integration tests
# ---------------------------------------------------------------------------

class TestGenerateDataset(unittest.TestCase):
    """Integration tests that write real FASTA and CSV files in a temp dir."""

    # A small set of sequences that gives a healthy mix of identities.
    # Using longer identical-ish sequences to ensure we can hit >85 %.
    _SEQ_A = "ACGTACGTACGTACGT" * 10  # 160 bp, 50 % GC
    _SEQ_B = "ACGTACGTACGTACGT" * 10  # identical to A → 100 % identity
    _SEQ_C = "TTTTTTTTTTTTTTTT" * 10  # 160 bp, 0 % GC → low identity with A
    _SEQ_D = "CCCCCCCCCCCCCCCC" * 10  # 160 bp, 100 % GC → low identity with C

    def _fasta_path(self, tmpdir: str) -> str:
        path = os.path.join(tmpdir, "seqs.fasta")
        records = [
            ("seqA", self._SEQ_A),
            ("seqB", self._SEQ_B),
            ("seqC", self._SEQ_C),
            ("seqD", self._SEQ_D),
        ]
        _write_fasta(path, records)
        return path

    def test_odd_num_pairs_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = self._fasta_path(tmpdir)
            out = os.path.join(tmpdir, "out.csv")
            with self.assertRaises(ValueError, msg="odd num_pairs should raise"):
                generate_dataset(fasta, num_pairs=5, output_csv=out)

    def test_balanced_output_equal_rows(self):
        """Output CSV must have exactly num_pairs/2 rows in each bucket."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = self._fasta_path(tmpdir)
            out = os.path.join(tmpdir, "out.csv")
            num_pairs = 10
            generate_dataset(fasta, num_pairs=num_pairs, output_csv=out, seed=0)

            import pandas as pd
            df = pd.read_csv(out)
            self.assertEqual(len(df), num_pairs)
            low = (df["real_percent_identity"] <= 85).sum()
            high = (df["real_percent_identity"] > 85).sum()
            self.assertEqual(low, num_pairs // 2)
            self.assertEqual(high, num_pairs // 2)

    def test_output_columns_unchanged(self):
        """Output CSV must still have real_percent_identity as first column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = self._fasta_path(tmpdir)
            out = os.path.join(tmpdir, "out.csv")
            generate_dataset(fasta, num_pairs=4, output_csv=out, seed=1)

            import pandas as pd
            df = pd.read_csv(out)
            self.assertIn("real_percent_identity", df.columns)
            self.assertEqual(df.columns[0], "real_percent_identity")
            # Spot-check a few known pair features
            for col in ("gc_content_min", "gc_content_max", "length_mean"):
                self.assertIn(col, df.columns)

    def test_shard_output_filename(self):
        """With num_shards=2, shard 0 must write to the .part0. path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = self._fasta_path(tmpdir)
            out = os.path.join(tmpdir, "data.csv")
            generate_dataset(
                fasta, num_pairs=4, output_csv=out,
                shard_id=0, num_shards=2, seed=2,
            )
            expected = os.path.join(tmpdir, "data.part0.csv")
            self.assertTrue(os.path.exists(expected))

    def test_reproducibility(self):
        """Same seed must produce identical output CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = self._fasta_path(tmpdir)
            out1 = os.path.join(tmpdir, "run1.csv")
            out2 = os.path.join(tmpdir, "run2.csv")
            generate_dataset(fasta, num_pairs=6, output_csv=out1, seed=99)
            generate_dataset(fasta, num_pairs=6, output_csv=out2, seed=99)

            import pandas as pd
            df1 = pd.read_csv(out1)
            df2 = pd.read_csv(out2)
            self.assertTrue(df1.equals(df2))

    @patch("src.data_creation.compute_pair_features", return_value={"pair_feature": 1.0})
    @patch("src.data_creation.compute_metrics", return_value={"m": 1})
    @patch("src.data_creation.process_sequence")
    @patch("src.data_creation.build_gc_bins", return_value=[[0, 1], [2, 3]])
    @patch("src.data_creation.load_fasta")
    @patch("src.data_creation.np.random.default_rng")
    def test_sampling_strategy_respects_bucket_need(
        self,
        mock_default_rng,
        mock_load_fasta,
        _mock_gc_bins,
        mock_process_sequence,
        _mock_compute_metrics,
        _mock_compute_pair_features,
    ):
        class FakeRng:
            def __init__(self):
                self._ints = iter([0, 0, 1, 0, 2])

            def random(self):
                return 0.0

            def integers(self, _low, _high=None):
                return next(self._ints)

        fake_rng = FakeRng()
        mock_default_rng.return_value = fake_rng
        mock_load_fasta.return_value = [
            ("s0", "AAAA"),
            ("s1", "AAAT"),
            ("s2", "TTTT"),
            ("s3", "CCCC"),
        ]
        mock_process_sequence.side_effect = lambda seq, *_args: (seq, [30] * len(seq), 30.0)

        attempted_pairs = []

        def fake_align(s1, s2, aligner=None):
            attempted_pairs.append((s1, s2))
            return 95.0 if {s1, s2} == {"AAAA", "AAAT"} else 20.0

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.csv")
            with patch("src.data_creation.align_sequences", side_effect=fake_align):
                generate_dataset("fake.fasta", num_pairs=2, output_csv=out, seed=0, chunk_size=2)

            import pandas as pd
            df = pd.read_csv(out)

        self.assertEqual(df["real_percent_identity"].gt(85).sum(), 1)
        self.assertEqual(df["real_percent_identity"].le(85).sum(), 1)
        self.assertEqual(attempted_pairs[0], ("AAAA", "AAAT"))
        self.assertEqual(attempted_pairs[1], ("AAAA", "TTTT"))


if __name__ == "__main__":
    unittest.main()
