import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from data_creation import _validate_shard_row_counts
from src.data_creation import BALANCE_THRESHOLD, generate_dataset


class TestDataCreationBalancing(unittest.TestCase):
    def test_generate_dataset_rejects_odd_num_pairs(self):
        with self.assertRaisesRegex(ValueError, "num_pairs must be even"):
            generate_dataset(
                fasta_paths=["/does/not/matter.fasta"],
                num_pairs=5,
                output_csv="/tmp/out.csv",
            )

    @patch("src.data_creation.compute_pair_features")
    @patch("src.data_creation.compute_metrics")
    @patch("src.data_creation.align_sequences")
    @patch("src.data_creation.process_sequence")
    @patch("src.data_creation.load_fasta")
    def test_metrics_only_for_accepted_balanced_pairs(
        self,
        mock_load_fasta,
        mock_process_sequence,
        mock_align_sequences,
        mock_compute_metrics,
        mock_compute_pair_features,
    ):
        mock_load_fasta.return_value = [("a", "AAAA"), ("b", "TTTT")]
        mock_process_sequence.side_effect = lambda seq, *_args: (seq, [30] * len(seq), 0)
        mock_align_sequences.side_effect = [10.0, 20.0, 30.0, 95.0, 96.0]
        mock_compute_metrics.return_value = {"m": 1}
        mock_compute_pair_features.return_value = {"pair_feature": 1.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = str(Path(tmpdir) / "balanced.csv")
            generate_dataset(
                fasta_paths=["fake.fasta"],
                num_pairs=4,
                output_csv=output_csv,
                chunk_size=4,
                seed=1,
            )

            with open(output_csv, newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 4)
        low = sum(float(r["real_percent_identity"]) <= BALANCE_THRESHOLD for r in rows)
        high = sum(float(r["real_percent_identity"]) > BALANCE_THRESHOLD for r in rows)
        self.assertEqual(low, 2)
        self.assertEqual(high, 2)
        self.assertEqual(mock_compute_metrics.call_count, 8)
        self.assertEqual(mock_compute_pair_features.call_count, 4)

    def test_validate_shards_checks_balance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "all_pairs_data.csv"
            part0 = Path(tmpdir) / "all_pairs_data.part0.csv"
            part1 = Path(tmpdir) / "all_pairs_data.part1.csv"

            with open(part0, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["real_percent_identity", "x"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"real_percent_identity": 80.0, "x": 1},
                        {"real_percent_identity": 85.0, "x": 1},
                        {"real_percent_identity": 90.0, "x": 1},
                    ]
                )
            with open(part1, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["real_percent_identity", "x"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"real_percent_identity": 70.0, "x": 1},
                        {"real_percent_identity": 91.0, "x": 1},
                        {"real_percent_identity": 99.0, "x": 1},
                    ]
                )

            _validate_shard_row_counts(str(out), num_pairs=6, num_shards=2)

            with open(part1, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["real_percent_identity", "x"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"real_percent_identity": 70.0, "x": 1},
                        {"real_percent_identity": 71.0, "x": 1},
                        {"real_percent_identity": 72.0, "x": 1},
                    ]
                )

            with self.assertRaisesRegex(ValueError, "expected <=85/>85"):
                _validate_shard_row_counts(str(out), num_pairs=6, num_shards=2)


if __name__ == "__main__":
    unittest.main()
