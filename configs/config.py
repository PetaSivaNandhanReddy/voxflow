#!/usr/bin/env python3
"""
VoxFlow V2 — System Configuration
Centralized configuration for Objective 2 session processing,
HuBERT_D final application model, locked thresholds, SQLite database,
and REST API server.
"""

import os
import pathlib

# Base Paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT

# Model Configuration — Final Selected Application Model: HuBERT_D
MODEL_CONFIG = {
    "model_id": "HuBERT_D",
    "model_name": "HuBERT Stage B Extended v2",
    "pretrained_model": "facebook/hubert-base-ls960",
    "stage": "B",
    "unfrozen_encoder_layers": 2,  # Layers 10 and 11
    "checkpoint_path": str(PROJECT_ROOT / "models" / "Hubert_D" / "hubert_stage_B_best.pt"),
    "val_summary_path": str(PROJECT_ROOT / "models" / "Hubert_D" / "val_summary.json"),
    "classes": ["Repetition", "Prolongation", "Block"],
    # Locked decision thresholds optimized on validation split:
    "locked_thresholds": {
        "Repetition": 0.77,
        "Prolongation": 0.79,
        "Block": 0.57
    },
    # Locked test-set benchmark metrics:
    "test_metrics": {
        "macro_f1": 0.4599,
        "mean_pr_auc": 0.4532,
        "mean_roc_auc": 0.8099,
        "per_class_f1": {
            "Repetition": 0.5565,
            "Prolongation": 0.4841,
            "Block": 0.3393
        }
    }
}

# ESP32 Hardware Configuration (Known-good ESP32 Dev Module + INMP441)
ESP32_CONFIG = {
    "device": "ESP32 Dev Module",
    "microphone": "INMP441",
    "port": os.getenv("VOXFLOW_SERIAL_PORT", "COM8"),            # Default serial port: COM8
    "baud_rate": int(os.getenv("VOXFLOW_BAUD_RATE", "460800")),   # Serial baud rate: 460800
    "inmp441_pin_bck": 26,     # BCK/SCK = GPIO26
    "inmp441_pin_ws": 25,      # WS = GPIO25
    "inmp441_pin_data": 32,    # DATA = GPIO32
    "sample_rate": 16000,      # 16000 Hz
    "channels": 1,             # Mono
    "audio_format": "PCM"      # PCM audio
}

# Session Audio & Sliding-Window Configuration (ESP32 Dev Module + INMP441)
SESSION_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "window_duration_sec": 3.0,
    "window_step_sec": 1.0,
    "window_samples": 48000,  # 16000 * 3.0
    "step_samples": 16000,    # 16000 * 1.0
    "min_duration_sec": 3.0,  # Minimum window for single analysis / test
    "min_standard_session_duration_sec": 30.0,  # Standard clinical session
    "max_standard_session_duration_sec": 60.0,
    "audio_upload_dir": str(PROJECT_ROOT / "audio_buffer"),
    "storage_dir": str(PROJECT_ROOT / "data")
}

# Event Aggregation Configuration (Multi-Label Temporal Clustering)
AGGREGATION_CONFIG = {
    "min_support_windows": 1,
    "merge_gap_sec": 1.5,     # Merge same-class detections separated by <= 1.5s
    "min_event_duration_sec": 0.5,
    "version": "v2_multilabel_aggregator"
}

# Database Configuration (Zero-ORM SQLite)
DB_CONFIG = {
    "db_path": str(PROJECT_ROOT / "app" / "voxflow.db")
}

# Flask REST API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "service_name": "VoxFlow Session API",
    "api_version": "v2.0"
}

# Streamlit Dashboard Configuration
DASHBOARD_CONFIG = {
    "title": "VoxFlow — Speech Fluency & Disfluency Monitoring Platform",
    "theme_color": "#1E3A8A"
}
