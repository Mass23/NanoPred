import gzip
import os
import tempfile
import unittest

import numpy as np

from apply_clustering import (
    discover_fastq_files,
    dereplicate_fastq_files,
    greedy_cluster,
    resolve_selected_features,
)


class DummyFastModel:
    def predict_proba(self, X):
        vals = (X["length_diff"].to_numpy() <= 1.0).astype(float)
        probs = np.where(vals > 0, 0.9, 0.1)
        return np.vstack([1.0 - probs, probs]).T


class DummyFullModel:
    def predict(self, X):
        vals = X["length_diff"].to_numpy()
        return np.where(vals <= 1.0, 98.0, 80.0)


class ModelWithFeatureNames:
    feature_names_in_ = np.array(["length_diff"])


class TestApplyClustering(unittest.TestCase):
    def _write_fastq(self, path, records, gz=False):
        opener = gzip.open if gz else open
        with opener(path, "wt", encoding="utf-8") as handle:
            for idx, seq in enumerate(records):
                handle.write(f"@r{idx}\n{seq}\n+\n{'I' * len(seq)}\n")

    def test_discovery_and_dereplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "sample1.fastq")
            p2 = os.path.join(tmpdir, "sample2.fq.gz")
            p3 = os.path.join(tmpdir, "ignore.txt")
            self._write_fastq(p1, ["AAAA", "AAAA", "CCCC"], gz=False)
            self._write_fastq(p2, ["AAAA", "GGGG"], gz=True)
            with open(p3, "w", encoding="utf-8") as handle:
                handle.write("noop")

            files = discover_fastq_files(tmpdir)
            self.assertEqual(len(files), 2)

            sample_names, table = dereplicate_fastq_files(files)
            self.assertEqual(sample_names, ["sample1.fastq", "sample2.fq.gz"])
            self.assertEqual(table["AAAA"]["total_abundance"], 3)
            self.assertEqual(table["AAAA"]["sample_counts"]["sample1.fastq"], 2)
            self.assertEqual(table["AAAA"]["sample_counts"]["sample2.fq.gz"], 1)
            self.assertEqual(table["CCCC"]["total_abundance"], 1)
            self.assertEqual(table["GGGG"]["total_abundance"], 1)

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
            fast_model=DummyFastModel(),
            full_model=DummyFullModel(),
            fast_features=["length_diff"],
            full_features=["length_diff"],
            fast_threshold=0.5,
            percent_identity=97.0,
        )

        by_id = {row["sequence_id"]: row for row in rows}
        self.assertTrue(by_id["id_a"]["is_centroid"])
        self.assertEqual(by_id["id_b"]["assignment_type"], "fast+full")
        self.assertEqual(by_id["id_b"]["otu_id"], by_id["id_a"]["otu_id"])
        self.assertTrue(by_id["id_c"]["is_centroid"])
        self.assertNotEqual(by_id["id_c"]["otu_id"], by_id["id_a"]["otu_id"])

    def test_selected_features_fallbacks(self):
        model = ModelWithFeatureNames()
        self.assertEqual(
            resolve_selected_features(model, {"selected_features": ["length_diff"]}, "fast"),
            ["length_diff"],
        )
        self.assertEqual(resolve_selected_features(model, None, "fast"), ["length_diff"])


if __name__ == "__main__":
    unittest.main()
