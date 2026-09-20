#!/usr/bin/env python3
"""
Unit tests for VoxFlow unified inference engine and event aggregation logic.
"""

import unittest
import pathlib
import numpy as np
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.engine import VoxFlowInferenceEngine
from app.services.event_aggregator import aggregate_window_events
from app.services.session_summary import calculate_session_summary

class TestVoxFlowInference(unittest.TestCase):
    def setUp(self):
        self.engine = VoxFlowInferenceEngine()

    def test_silent_window_prediction(self):
        # 3 seconds of zero signal
        silent_audio = np.zeros(48000, dtype=np.float32)
        res = self.engine.predict_window(silent_audio)

        self.assertTrue(res["is_fluent"])
        self.assertEqual(len(res["active_labels"]), 0)
        self.assertEqual(res["probabilities"]["Repetition"], 0.0)
        self.assertEqual(res["probabilities"]["Block"], 0.0)

    def test_window_padding(self):
        # Short audio (1.0 second) gets automatically padded to 3.0 seconds
        short_audio = np.zeros(16000, dtype=np.float32)
        res = self.engine.predict_window(short_audio)
        self.assertTrue(res["is_fluent"])

    def test_multi_label_event_aggregation(self):
        # Create 3 contiguous windows predicting Repetition
        windows = [
            {
                "start_time": 0.0,
                "end_time": 3.0,
                "probabilities": {"Repetition": 0.60, "Prolongation": 0.05, "Block": 0.05},
                "predictions": {"Repetition": True, "Prolongation": False, "Block": False},
                "is_fluent": False
            },
            {
                "start_time": 1.0,
                "end_time": 4.0,
                "probabilities": {"Repetition": 0.70, "Prolongation": 0.05, "Block": 0.05},
                "predictions": {"Repetition": True, "Prolongation": False, "Block": False},
                "is_fluent": False
            },
            {
                "start_time": 2.0,
                "end_time": 5.0,
                "probabilities": {"Repetition": 0.50, "Prolongation": 0.05, "Block": 0.05},
                "predictions": {"Repetition": True, "Prolongation": False, "Block": False},
                "is_fluent": False
            }
        ]

        events = aggregate_window_events("sess_001", windows)
        # Should merge into a single event spanning 0.0s to 5.0s
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "Repetition")
        self.assertEqual(events[0]["start_time"], 0.0)
        self.assertEqual(events[0]["end_time"], 5.0)
        self.assertEqual(events[0]["supporting_windows"], 3)
        self.assertAlmostEqual(events[0]["confidence"], 0.60, places=2)

    def test_session_summary_fluency_ratio_bounds(self):
        windows = [
            {"is_fluent": True},
            {"is_fluent": False},
            {"is_fluent": True},
            {"is_fluent": True}
        ]
        events = [{"event_type": "Repetition"}]
        summary = calculate_session_summary("sess_002", 6.0, windows, events)

        self.assertEqual(summary["total_windows"], 4)
        self.assertEqual(summary["fluent_windows"], 3)
        self.assertAlmostEqual(summary["fluency_ratio"], 0.75)
        self.assertTrue(0.0 <= summary["fluency_ratio"] <= 1.0)

if __name__ == '__main__':
    unittest.main()
