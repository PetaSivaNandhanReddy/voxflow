#!/usr/bin/env python3
"""
Unit tests for VoxFlow SQLite database persistence layer.
"""

import os
import unittest
import tempfile
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.database import init_db, save_session, get_all_sessions, get_session_details

class TestVoxFlowDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = pathlib.Path(self.temp_dir.name) / "test_voxflow.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_and_save_session(self):
        init_db(self.db_path)
        self.assertTrue(self.db_path.exists())

        session_data = {
            "session_id": "test_sess_001",
            "user_id": "clinician_1",
            "duration_sec": 42.0,
            "model_name": "Model 1: Multi-label SVM Baseline",
            "total_windows": 40,
            "fluent_windows": 35,
            "fluency_ratio": 0.875,
            "repetition_count": 2,
            "prolongation_count": 0,
            "block_count": 1
        }

        windows = [
            {
                "start_time": 0.0,
                "end_time": 3.0,
                "probabilities": {"Repetition": 0.05, "Prolongation": 0.02, "Block": 0.01},
                "is_fluent": True,
                "active_labels": []
            },
            {
                "start_time": 1.0,
                "end_time": 4.0,
                "probabilities": {"Repetition": 0.65, "Prolongation": 0.05, "Block": 0.02},
                "is_fluent": False,
                "active_labels": ["Repetition"]
            }
        ]

        events = [
            {
                "event_type": "Repetition",
                "start_time": 1.0,
                "end_time": 4.0,
                "confidence": 0.65,
                "supporting_windows": 1
            }
        ]

        save_session(session_data, windows, events, db_path=self.db_path)

        sessions = get_all_sessions(limit=10, db_path=self.db_path)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], "test_sess_001")
        self.assertAlmostEqual(sessions[0]["fluency_ratio"], 0.875)

        details = get_session_details("test_sess_001", db_path=self.db_path)
        self.assertIsNotNone(details)
        self.assertEqual(len(details["windows"]), 2)
        self.assertEqual(len(details["events"]), 1)
        self.assertEqual(details["events"][0]["event_type"], "Repetition")

if __name__ == '__main__':
    unittest.main()
