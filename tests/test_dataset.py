#!/usr/bin/env python3
"""
Unit tests for VoxFlow dataset manifest integrity and speaker exclusivity.
"""

import unittest
import pathlib
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

class TestVoxFlowDataset(unittest.TestCase):
    def setUp(self):
        self.manifest_dir = PROJECT_ROOT / "dataset/manifests"
        self.train_path = self.manifest_dir / "v2_train_manifest.csv"
        self.val_path = self.manifest_dir / "v2_val_manifest.csv"
        self.test_path = self.manifest_dir / "v2_test_manifest.csv"
        self.full_path = self.manifest_dir / "v2_full_clean_manifest.csv"

    def test_manifest_files_exist(self):
        self.assertTrue(self.train_path.exists(), "Train manifest missing")
        self.assertTrue(self.val_path.exists(), "Val manifest missing")
        self.assertTrue(self.test_path.exists(), "Test manifest missing")
        self.assertTrue(self.full_path.exists(), "Full clean manifest missing")

    def test_manifest_row_counts(self):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)
        full_df = pd.read_csv(self.full_path)

        self.assertEqual(len(train_df), 17555, "Train split count mismatch")
        self.assertEqual(len(val_df), 6632, "Validation split count mismatch")
        self.assertEqual(len(test_df), 6812, "Test split count mismatch")
        self.assertEqual(len(full_df), 30999, "Full clean count mismatch")
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(full_df))

    def test_speaker_exclusivity(self):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        train_spk = set(train_df['speaker'].dropna().unique())
        val_spk = set(val_df['speaker'].dropna().unique())
        test_spk = set(test_df[test_df['speaker'] != 'UNASSIGNED_MULTI_SPEAKER']['speaker'].dropna().unique())

        self.assertEqual(len(train_spk.intersection(val_spk)), 0, "Speaker leakage between Train and Val")
        self.assertEqual(len(train_spk.intersection(test_spk)), 0, "Speaker leakage between Train and Test")
        self.assertEqual(len(val_spk.intersection(test_spk)), 0, "Speaker leakage between Val and Test")

    def test_label_binary_validity(self):
        train_df = pd.read_csv(self.train_path)
        for col in ['repetition_label', 'prolongation_label', 'block_label']:
            unique_vals = set(train_df[col].unique())
            self.assertTrue(unique_vals.issubset({0, 1}), f"Non-binary values in {col}")

if __name__ == '__main__':
    unittest.main()
