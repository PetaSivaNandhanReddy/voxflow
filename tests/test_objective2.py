#!/usr/bin/env python3
"""
VoxFlow V2 — Objective 2 Test Suite
Comprehensive lightweight tests verifying:
  1. Real HuBERT_D model loading and checkpoint verification
  2. Single-window multi-label inference & threshold decisions
  3. Continuous sliding-window generation (3.0s window, 1.0s step)
  4. Temporal event aggregation without double-counting
  5. Deterministic session summary calculation & fluency ratio bounds
  6. SQLite persistence and foreign-key integrity
  7. REST API lifecycle (start -> audio -> stop -> analyze -> get)
"""

import os
import io
import wave
import tempfile
import pathlib
import unittest
import numpy as np

import sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MODEL_CONFIG, SESSION_CONFIG
from inference.engine import get_inference_engine, VoxFlowInferenceEngine
from app.services.session_engine import (
    validate_and_load_session_audio,
    process_session_windows,
    process_continuous_session,
    AudioValidationError
)
from app.services.event_aggregator import aggregate_window_events
from app.services.session_summary import calculate_session_summary
from app.db.database import (
    init_db,
    save_session,
    save_session_record,
    save_session_analysis,
    get_session,
    get_latest_session,
    get_all_sessions
)
from server import app


class TestObjective2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_inference_engine()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db = pathlib.Path(self.temp_dir.name) / "test_voxflow.db"
        init_db(self.test_db)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Model Loading & Verification
    # -------------------------------------------------------------------------
    def test_01_hubert_d_model_loading(self):
        """Verifies HuBERT_D model is loaded with strict checkpoint matching."""
        self.assertTrue(self.engine.model_loaded)
        self.assertEqual(self.engine.model_id, "HuBERT_D")
        self.assertEqual(self.engine.model_name, "HuBERT Stage B Extended v2")
        self.assertFalse(self.engine.model.training)  # Must be in eval() mode
        self.assertEqual(
            self.engine.thresholds,
            {"Repetition": 0.77, "Prolongation": 0.79, "Block": 0.57}
        )

    # -------------------------------------------------------------------------
    # 2. One-Window Inference
    # -------------------------------------------------------------------------
    def test_02_single_window_inference(self):
        """Verifies 3.0s window produces multi-label probabilities and binary predictions."""
        # 3.0s synthetic audio @ 16 kHz
        synthetic_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 48000)).astype(np.float32) * 0.1
        res = self.engine.predict_window(synthetic_audio)

        self.assertIn("probabilities", res)
        self.assertIn("predictions", res)
        self.assertIn("is_fluent", res)
        self.assertIn("active_labels", res)

        for c in ["Repetition", "Prolongation", "Block"]:
            self.assertIn(c, res["probabilities"])
            self.assertIn(c, res["predictions"])
            self.assertTrue(0.0 <= res["probabilities"][c] <= 1.0)
            self.assertIsInstance(res["predictions"][c], bool)

    def test_03_silent_window_is_fluent(self):
        """Verifies inaudible/silent windows default to fluent with zero disfluency probabilities."""
        silent_audio = np.zeros(48000, dtype=np.float32)
        res = self.engine.predict_window(silent_audio)
        self.assertTrue(res["is_fluent"])
        self.assertEqual(len(res["active_labels"]), 0)
        self.assertEqual(res["probabilities"]["Repetition"], 0.0)

    # -------------------------------------------------------------------------
    # 3. Continuous Sliding-Window Generation
    # -------------------------------------------------------------------------
    def test_04_sliding_window_timings(self):
        """Verifies continuous 5.0-second audio yields 3 overlapping windows [0-3, 1-4, 2-5]."""
        # 5.0s audio @ 16 kHz = 80,000 samples
        audio_5s = np.random.randn(80000).astype(np.float32) * 0.05
        windows = process_session_windows("sess_test_5s", audio_5s, 16000, 5.0, engine=self.engine)

        # Expected: floor((5.0 - 3.0) / 1.0) + 1 = 3 windows
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]["start_time"], 0.0)
        self.assertEqual(windows[0]["end_time"], 3.0)
        self.assertEqual(windows[1]["start_time"], 1.0)
        self.assertEqual(windows[1]["end_time"], 4.0)
        self.assertEqual(windows[2]["start_time"], 2.0)
        self.assertEqual(windows[2]["end_time"], 5.0)

    # -------------------------------------------------------------------------
    # 4. Temporal Event Aggregation
    # -------------------------------------------------------------------------
    def test_05_multi_label_event_aggregation(self):
        """Verifies overlapping windows are merged into a single event without double-counting."""
        mock_windows = [
            {
                "window_id": 0, "start_time": 0.0, "end_time": 3.0,
                "probabilities": {"Repetition": 0.85, "Prolongation": 0.10, "Block": 0.05},
                "predictions": {"Repetition": True, "Prolongation": False, "Block": False},
                "is_fluent": False
            },
            {
                "window_id": 1, "start_time": 1.0, "end_time": 4.0,
                "probabilities": {"Repetition": 0.80, "Prolongation": 0.10, "Block": 0.05},
                "predictions": {"Repetition": True, "Prolongation": False, "Block": False},
                "is_fluent": False
            },
            {
                "window_id": 2, "start_time": 2.0, "end_time": 5.0,
                "probabilities": {"Repetition": 0.20, "Prolongation": 0.10, "Block": 0.05},
                "predictions": {"Repetition": False, "Prolongation": False, "Block": False},
                "is_fluent": True
            }
        ]
        events = aggregate_window_events("sess_agg", mock_windows)

        # Windows 0 and 1 merge into one Repetition event from 0.0s to 4.0s
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "Repetition")
        self.assertEqual(events[0]["start_time"], 0.0)
        self.assertEqual(events[0]["end_time"], 4.0)
        self.assertEqual(events[0]["supporting_windows"], 2)

    # -------------------------------------------------------------------------
    # 5. Session Summary Calculation
    # -------------------------------------------------------------------------
    def test_06_session_summary_metrics(self):
        """Verifies Session Fluency Ratio is bounded in [0.0, 1.0]."""
        mock_windows = [
            {"is_fluent": True},
            {"is_fluent": False},
            {"is_fluent": True},
            {"is_fluent": True}
        ]
        mock_events = [{"event_type": "Prolongation"}]
        summary = calculate_session_summary("sess_sum", 6.0, mock_windows, mock_events)

        self.assertEqual(summary["total_windows"], 4)
        self.assertEqual(summary["fluent_windows"], 3)
        self.assertAlmostEqual(summary["fluency_ratio"], 0.75)
        self.assertEqual(summary["prolongation_count"], 1)
        self.assertEqual(summary["repetition_count"], 0)

    # -------------------------------------------------------------------------
    # 6. SQLite Persistence & Retrieval
    # -------------------------------------------------------------------------
    def test_07_database_persistence_and_query(self):
        """Verifies database records are saved and queried with full integrity."""
        session_id = "test_db_001"
        save_session_record(
            session_id=session_id,
            user_id="clinician_a",
            duration=4.0,
            status="ANALYZED",
            db_path=self.test_db
        )

        mock_windows = [
            {
                "window_id": 0, "start_time": 0.0, "end_time": 3.0,
                "prob_repetition": 0.1, "prob_prolongation": 0.8, "prob_block": 0.05,
                "is_fluent": False, "active_labels": ["Prolongation"], "confidence": 0.8
            }
        ]
        mock_events = [
            {
                "event_type": "Prolongation", "start_time": 0.0, "end_time": 3.0,
                "confidence": 0.8, "supporting_windows": 1
            }
        ]
        summary = {
            "duration_sec": 4.0, "total_windows": 1, "fluent_windows": 0,
            "fluency_ratio": 0.0, "prolongation_count": 1
        }

        save_session_analysis(session_id, mock_windows, mock_events, summary, db_path=self.test_db)

        retrieved = get_session(session_id, db_path=self.test_db)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["id"], session_id)
        self.assertEqual(len(retrieved["windows"]), 1)
        self.assertEqual(len(retrieved["events"]), 1)
        self.assertEqual(retrieved["events"][0]["event_type"], "Prolongation")

    # -------------------------------------------------------------------------
    # 7. Complete REST API Lifecycle Test
    # -------------------------------------------------------------------------
    def test_08_rest_api_lifecycle(self):
        """
        Verifies the complete TARP REST API session lifecycle:
          start -> audio upload -> stop -> analyze -> inspect
        """
        # Point app DB to temporary test database
        os.environ["VOXFLOW_DB_PATH"] = str(self.test_db)
        client = app.test_client()

        # Step 1: Start Session
        res_start = client.post('/api/v1/session/start', json={"user_id": "test_user"})
        self.assertEqual(res_start.status_code, 201)
        session_id = res_start.get_json()["session_id"]
        self.assertTrue(session_id.startswith("sess_"))

        # Step 2: Upload Audio (Create 4.0 seconds of 16 kHz WAV audio in-memory)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # 4 seconds = 64,000 samples of low-amplitude tone
            t = np.linspace(0, 4, 64000)
            samples = (np.sin(2 * np.pi * 300 * t) * 10000).astype(np.int16)
            wf.writeframes(samples.tobytes())
        audio_bytes = buf.getvalue()

        res_audio = client.post(
            f'/api/v1/session/{session_id}/audio',
            data=audio_bytes,
            content_type='application/octet-stream'
        )
        self.assertEqual(res_audio.status_code, 200)

        # Step 3: Stop Session
        res_stop = client.post(f'/api/v1/session/{session_id}/stop')
        self.assertEqual(res_stop.status_code, 200)

        # Step 4: Analyze Session (Executes real HuBERT_D inference!)
        res_analyze = client.post(f'/api/v1/session/{session_id}/analyze')
        self.assertEqual(res_analyze.status_code, 200)
        analyze_data = res_analyze.get_json()
        self.assertEqual(analyze_data["status"], "success")
        self.assertIn("summary", analyze_data)
        self.assertEqual(analyze_data["window_count"], 2)  # 4s -> 2 windows (0-3, 1-4)

        # Step 5: Get Session Details
        res_get = client.get(f'/api/v1/session/{session_id}')
        self.assertEqual(res_get.status_code, 200)
        get_data = res_get.get_json()["session"]
        self.assertEqual(get_data["id"], session_id)
        self.assertEqual(len(get_data["windows"]), 2)


if __name__ == '__main__':
    unittest.main()
