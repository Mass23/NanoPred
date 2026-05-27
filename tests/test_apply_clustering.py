import gzip
import os
import subprocess
import tempfile
import unittest
import unittest.mock

import numpy as np
import pandas as pd

from apply_clustering import (
    _run_cutadapt,
    build_otu_table,
    dereplicate_fastq_files,
    discover_fastq_files,
    greedy_cluster,
    resolve_selected_features,
    write_output,
)


class MockFastModel:
    def predict_proba(self, X):
        match = X["length_diff"].to_numpy() <= 1.0
        probs = np.where(match, 0.9, 0.1)
        return np.vstack([1.0 - probs, probs]).T


class MockFullModel:
    def predict(self, X):
        vals = X["length_diff"].to_numpy()
        return np.where(vals <= 1.0, 98.0, 80.0)


class MockModelWithFeatures:
    feature_names_in_ = np.array(["length_diff"])


class TestApplyClustering(unittest.TestCase):
    def _write_fastq(self, path, records, gz=False):
        opener = gzip.open if gz else open
        with opener(path, "wt", encoding="utf-8") as handle:
            for idx, seq in enumerate(records):
                handle.write(f"@r{idx}\n{seq}\n+\n{'I' * len(seq)}\n")

    def test_discovery_and_dereplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = os.path.join(tmpdir, "sample1.fastq")
            s2 = os.path.join(tmpdir, "sample2.fq.gz")
            ignore = os.path.join(tmpdir, "ignore.txt")
            # sample1: two AAAA + one CCCC
            self._write_fastq(s1, ["AAAA", "AAAA", "CCCC"], gz=False)
            # sample2: one AAAA + one GGGG
            self._write_fastq(s2, ["AAAA", "GGGG"], gz=True)
            with open(ignore, "w", encoding="utf-8") as handle:
                handle.write("noop")

            files = discover_fastq_files(tmpdir)
            self.assertEqual(len(files), 2)
            found_samples = {f[1] for f in files}
            self.assertIn("sample1", found_samples)
            self.assertIn("sample2", found_samples)

            sample_names, table = dereplicate_fastq_files(files)
            self.assertEqual(set(sample_names), {"sample1", "sample2"})
            # AAAA: 2 from sample1 + 1 from sample2 = 3
            self.assertEqual(table["AAAA"]["total_abundance"], 3)
            self.assertEqual(table["AAAA"]["sample_counts"]["sample1"], 2)
            self.assertEqual(table["AAAA"]["sample_counts"]["sample2"], 1)
            # CCCC: 1 from sample1 only
            self.assertEqual(table["CCCC"]["total_abundance"], 1)
            self.assertEqual(table["CCCC"]["sample_counts"]["sample1"], 1)
            # GGGG: 1 from sample2 only
            self.assertEqual(table["GGGG"]["total_abundance"], 1)
            self.assertEqual(table["GGGG"]["sample_counts"]["sample2"], 1)

    def test_greedy_cluster_assigns_and_creates_new_centroid(self):
        global_table = {
            "AAAA": {
                "sequence": "AAAA",
                "sequence_id": "id_a",
                "total_abundance": 5,
                "sample_counts": {"s1": 5},
                "representative_quality": [40, 40, 40, 40],
            },
            "AAAT": {
                "sequence": "AAAT",
                "sequence_id": "id_b",
                "total_abundance": 3,
                "sample_counts": {"s1": 3},
                "representative_quality": [40, 40, 40, 40],
            },
            "AAAAAAAAAAAA": {
                "sequence": "AAAAAAAAAAAA",
                "sequence_id": "id_c",
                "total_abundance": 2,
                "sample_counts": {"s2": 2},
                "representative_quality": [40] * 12,
            },
        }

        rows = greedy_cluster(
            global_table=global_table,
            fast_model=MockFastModel(),
            full_model=MockFullModel(),
            fast_features=["length_diff"],
            full_features=["length_diff"],
            fast_threshold=0.5,
            percent_identity=97.0,
        )

        by_id = {row["sequence_id"]: row for row in rows}
        self.assertTrue(by_id["id_a"]["is_centroid"])
        self.assertEqual(by_id["id_a"]["otu_id"], "OTU_1")
        self.assertEqual(by_id["id_b"]["assignment_type"], "fast+full")
        self.assertEqual(by_id["id_b"]["otu_id"], by_id["id_a"]["otu_id"])
        self.assertTrue(by_id["id_c"]["is_centroid"])
        self.assertEqual(by_id["id_c"]["otu_id"], "OTU_2")
        self.assertNotEqual(by_id["id_c"]["otu_id"], by_id["id_a"]["otu_id"])

    def test_selected_features_fallbacks(self):
        model = MockModelWithFeatures()
        self.assertEqual(
            resolve_selected_features(model, {"selected_features": ["length_diff"]}, "fast"),
            ["length_diff"],
        )
        self.assertEqual(resolve_selected_features(model, None, "fast"), ["length_diff"])

    def test_build_otu_table_aggregates_member_counts(self):
        rows = [
            {
                "sequence_id": "id_a",
                "sequence": "AAAA",
                "otu_id": "OTU_1",
                "centroid_sequence_id": "id_a",
                "predicted_percent_identity": None,
                "assignment_type": "new_centroid",
                "is_centroid": True,
                "total_abundance": 3,
                "sample_counts": {"s1": 2, "s2": 1},
            },
            {
                "sequence_id": "id_b",
                "sequence": "AAAT",
                "otu_id": "OTU_1",
                "centroid_sequence_id": "id_a",
                "predicted_percent_identity": 98.0,
                "assignment_type": "fast+full",
                "is_centroid": False,
                "total_abundance": 4,
                "sample_counts": {"s1": 1, "s2": 3},
            },
            {
                "sequence_id": "id_c",
                "sequence": "CCCC",
                "otu_id": "OTU_2",
                "centroid_sequence_id": "id_c",
                "predicted_percent_identity": None,
                "assignment_type": "new_centroid",
                "is_centroid": True,
                "total_abundance": 2,
                "sample_counts": {"s2": 2},
            },
        ]

        otu_table = build_otu_table(rows, ["s1", "s2"])

        self.assertEqual(list(otu_table.index), ["OTU_1", "OTU_2"])
        self.assertEqual(list(otu_table.columns), ["s1", "s2"])
        self.assertEqual(int(otu_table.loc["OTU_1", "s1"]), 3)
        self.assertEqual(int(otu_table.loc["OTU_1", "s2"]), 4)
        self.assertEqual(int(otu_table.loc["OTU_2", "s1"]), 0)
        self.assertEqual(int(otu_table.loc["OTU_2", "s2"]), 2)

    def test_write_output_creates_tsv_and_otu_table(self):
        rows = [
            {
                "sequence_id": "id_a",
                "sequence": "AAAA",
                "otu_id": "OTU_1",
                "centroid_sequence_id": "id_a",
                "predicted_percent_identity": None,
                "assignment_type": "new_centroid",
                "is_centroid": True,
                "total_abundance": 2,
                "sample_counts": {"sample1": 2},
            },
            {
                "sequence_id": "id_b",
                "sequence": "AAAT",
                "otu_id": "OTU_1",
                "centroid_sequence_id": "id_a",
                "predicted_percent_identity": 98.0,
                "assignment_type": "fast+full",
                "is_centroid": False,
                "total_abundance": 1,
                "sample_counts": {"sample2": 1},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "assignments.tsv")
            assignments_path, otu_table_path = write_output(rows, ["sample1", "sample2"], output_path)

            self.assertEqual(assignments_path, output_path)
            self.assertEqual(otu_table_path, os.path.join(tmpdir, "OTU_table.csv"))

            assignments_df = pd.read_csv(assignments_path, sep="\t")
            otu_df = pd.read_csv(otu_table_path)

            self.assertIn("sample_counts", assignments_df.columns)
            self.assertIn("assignment_status", assignments_df.columns)
            self.assertEqual(assignments_df.loc[0, "assignment_status"], "new_centroid")
            self.assertEqual(assignments_df.loc[1, "assignment_status"], "assigned_to_existing_otu")
            self.assertEqual(assignments_df.loc[0, "count_sample1"], 2)
            self.assertEqual(assignments_df.loc[1, "count_sample2"], 1)

            self.assertEqual(list(otu_df.columns), ["otu_id", "sample1", "sample2"])
            self.assertEqual(otu_df.loc[0, "otu_id"], "OTU_1")
            self.assertEqual(int(otu_df.loc[0, "sample1"]), 2)
            self.assertEqual(int(otu_df.loc[0, "sample2"]), 1)


class TestCutadaptTrimming(unittest.TestCase):
    def test_run_cutadapt_builds_linked_adapter_command(self):
        """Both primers → linked -g PRIMER5...PRIMER3 with --discard-untrimmed."""
        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stderr=b"")
        with unittest.mock.patch("subprocess.run", return_value=fake_result) as mock_run:
            _run_cutadapt("in.fastq", "out.fastq", "ACGT", "TGCA")
            cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "cutadapt")
        self.assertIn("-g", cmd)
        self.assertEqual(cmd[cmd.index("-g") + 1], "ACGT...TGCA")
        self.assertNotIn("-G", cmd)
        self.assertNotIn("-p", cmd)
        self.assertIn("--discard-untrimmed", cmd)
        self.assertIn("-o", cmd)
        self.assertEqual(cmd[cmd.index("-o") + 1], "out.fastq")
        self.assertIn("in.fastq", cmd)

    def test_run_cutadapt_only_primer5_uses_g_flag(self):
        """Only primer5 → -g without linked adapter, discards untrimmed."""
        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stderr=b"")
        with unittest.mock.patch("subprocess.run", return_value=fake_result) as mock_run:
            _run_cutadapt("in.fastq", "out.fastq", "ACGT", None)
            cmd = mock_run.call_args[0][0]
        self.assertIn("-g", cmd)
        self.assertEqual(cmd[cmd.index("-g") + 1], "ACGT")
        self.assertNotIn("-G", cmd)
        self.assertNotIn("-a", cmd)
        self.assertIn("--discard-untrimmed", cmd)

    def test_run_cutadapt_only_primer3_uses_a_flag(self):
        """Only primer3 → -a (3' single-end), discards untrimmed."""
        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stderr=b"")
        with unittest.mock.patch("subprocess.run", return_value=fake_result) as mock_run:
            _run_cutadapt("in.fastq", "out.fastq", None, "TGCA")
            cmd = mock_run.call_args[0][0]
        self.assertNotIn("-g", cmd)
        self.assertNotIn("-G", cmd)
        self.assertIn("-a", cmd)
        self.assertEqual(cmd[cmd.index("-a") + 1], "TGCA")
        self.assertIn("--discard-untrimmed", cmd)

    def test_run_cutadapt_raises_when_not_installed(self):
        """FileNotFoundError from subprocess should be re-raised as RuntimeError."""
        with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeError) as ctx:
                _run_cutadapt("in.fastq", "out.fastq", "ACGT", "TGCA")
        self.assertIn("cutadapt", str(ctx.exception))

    def test_run_cutadapt_raises_on_nonzero_exit(self):
        """A non-zero exit code from cutadapt should be raised as RuntimeError."""
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stderr=b"adapter not found"
        )
        with unittest.mock.patch("subprocess.run", return_value=failed):
            with self.assertRaises(RuntimeError) as ctx:
                _run_cutadapt("in.fastq", "out.fastq", "ACGT", "TGCA")
        self.assertIn("cutadapt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
