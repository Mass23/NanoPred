import gzip
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from apply_clustering import (
    _reverse_complement,
    trim_read_with_primers,
    build_otu_table,
    discover_fastq_files,
    dereplicate_fastq_files,
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


class TestPrimerTrimming(unittest.TestCase):
    def test_reverse_complement(self):
        self.assertEqual(_reverse_complement("ATGC"), "GCAT")
        self.assertEqual(_reverse_complement("AAAA"), "TTTT")
        self.assertEqual(_reverse_complement("ACGT"), "ACGT")

    def test_trim_no_primers(self):
        seq = "ACGTACGT"
        quality = [30, 30, 30, 30, 30, 30, 30, 30]
        trimmed_s, trimmed_q = trim_read_with_primers(seq, quality, None, None)
        self.assertEqual(trimmed_s, seq)
        self.assertEqual(trimmed_q, quality)

    def test_trim_both_primers_forward(self):
        # Sequence: AAA [insert] TTT where AAA=primer5, TTT=primer3
        seq = "AAACCCGTTTTT"
        quality = list(range(len(seq)))
        p5 = "AAA"
        p3 = "TTT"
        trimmed_s, trimmed_q = trim_read_with_primers(seq, quality, p5, p3)
        # After trimming: "CCCG" (between primer5 end and primer3 start)
        self.assertEqual(trimmed_s, "CCCG")
        # Quality should align with the trimmed region
        self.assertEqual(trimmed_q, quality[3:7])
        self.assertEqual(len(trimmed_s), len(trimmed_q))

    def test_trim_only_primer5(self):
        seq = "AAACCCGGG"
        quality = list(range(len(seq)))
        trimmed_s, trimmed_q = trim_read_with_primers(seq, quality, "AAA", None)
        self.assertEqual(trimmed_s, "CCCGGG")
        self.assertEqual(trimmed_q, quality[3:])
        self.assertEqual(len(trimmed_s), len(trimmed_q))

    def test_trim_only_primer3(self):
        seq = "CCCGGGTTT"
        quality = list(range(len(seq)))
        trimmed_s, trimmed_q = trim_read_with_primers(seq, quality, None, "TTT")
        self.assertEqual(trimmed_s, "CCCGGG")
        self.assertEqual(trimmed_q, quality[:6])
        self.assertEqual(len(trimmed_s), len(trimmed_q))

    def test_trim_not_found_returns_original(self):
        # Primers not present — relaxed: return unchanged
        seq = "GGGGGGGG"
        quality = [20] * len(seq)
        trimmed_s, trimmed_q = trim_read_with_primers(seq, quality, "AAA", "TTT")
        self.assertEqual(trimmed_s, seq)
        self.assertEqual(trimmed_q, quality)

    def test_trim_reverse_complement_orientation(self):
        # Build a read that comes from the reverse strand:
        #   read = RC(primer5) + RC(insert) + RC(primer3)
        # Taking the RC of this read gives: primer3 + insert + primer5.
        # Use primers that don't appear in their own RC to ensure the forward
        # pass fails and the RC pass is exercised.
        p5 = "AACC"   # RC(p5) = "GGTT"
        p3 = "TTGG"   # RC(p3) = "CCAA"
        insert = "GCGC"
        reverse_strand_read = _reverse_complement(p5) + _reverse_complement(insert) + _reverse_complement(p3)
        # = "GGTT" + "GCGC" + "CCAA" = "GGTTGCGCCCAA"
        quality = list(range(len(reverse_strand_read)))

        trimmed_s, trimmed_q = trim_read_with_primers(
            reverse_strand_read, quality, p5, p3
        )
        self.assertEqual(trimmed_s, insert)
        self.assertEqual(len(trimmed_s), len(trimmed_q))

    def test_dereplication_with_primer_trimming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Reads: primer5 + insert + primer3
            # After trimming, inserts should be dereplicated together
            p1 = os.path.join(tmpdir, "sample1.fastq")
            p5 = "AAA"
            p3 = "TTT"
            with open(p1, "w", encoding="utf-8") as fh:
                # Two reads with same insert "CCCC" but different flanking primers
                for rec_seq in ["AAACCCCTTTT", "AAACCCCTTTT", "AAAGGGGTTTT"]:
                    qual_str = "I" * len(rec_seq)
                    fh.write(f"@r\n{rec_seq}\n+\n{qual_str}\n")

            files = [p1]
            sample_names, table = dereplicate_fastq_files(files, primer5=p5, primer3=p3)
            # "AAACCCCTTTT" → trimmed to "CCCC" (2 copies)
            # "AAAGGGGTTTT" → trimmed to "GGGG" (1 copy)
            self.assertIn("CCCC", table)
            self.assertIn("GGGG", table)
            self.assertEqual(table["CCCC"]["total_abundance"], 2)
            self.assertEqual(table["GGGG"]["total_abundance"], 1)
            # Quality should be aligned: 4 scores for 4-base insert
            self.assertEqual(len(table["CCCC"]["representative_quality"]), 4)


if __name__ == "__main__":
    unittest.main()
