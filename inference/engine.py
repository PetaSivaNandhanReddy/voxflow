#!/usr/bin/env python3
"""
VoxFlow V2 — Unified Inference Engine (HuBERT_D)
Provides real multi-label disfluency inference for sliding audio windows.
Uses the locked HuBERT Stage B Extended v2 checkpoint (HuBERT_D).
Targets: Repetition (0.77), Prolongation (0.79), Block (0.57).
"""

import os
import sys
import json
import pathlib
import numpy as np
import torch
import torch.nn as nn

# Project root resolution
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MODEL_CONFIG, SESSION_CONFIG


class HubertMultiLabelClassifier(nn.Module):
    """
    Sequence classifier combining pretrained HuBERT encoder representations
    with a lightweight multi-label MLP classification head.
    Architecture strictly matches HuBERT Stage B Extended v2 training.
    """
    def __init__(self, hubert_model, num_labels=3):
        super().__init__()
        self.hubert = hubert_model
        self.hidden_size = self.hubert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values):
        outputs = self.hubert(input_values=input_values)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(pooled)
        return logits


class VoxFlowInferenceEngine:
    """
    Inference Engine for runtime speech fluency analysis.
    Directly evaluates 3.0s audio windows using the trained HuBERT_D model.
    """
    _instance = None

    def __init__(self, model_path=None, config_path=None, device=None):
        self.class_names = ["Repetition", "Prolongation", "Block"]
        self.thresholds = dict(MODEL_CONFIG["locked_thresholds"])
        self.model_name = MODEL_CONFIG["model_name"]
        self.model_id = MODEL_CONFIG["model_id"]
        self.sample_rate = SESSION_CONFIG["sample_rate"]
        self.window_duration = SESSION_CONFIG["window_duration_sec"]
        self.target_samples = SESSION_CONFIG["window_samples"]

        # Resolve checkpoint path
        self.model_path = model_path or MODEL_CONFIG["checkpoint_path"]
        if not os.path.exists(self.model_path):
            # Check secondary location
            alt_path = str(PROJECT_ROOT / "models" / "HuBERT" / "hubert_D" / "hubert_stage_B_best.pt")
            if os.path.exists(alt_path):
                self.model_path = alt_path

        self.config_path = config_path or str(PROJECT_ROOT / "configs" / "hubert_base_config.json")

        # Device selection: CUDA -> MPS -> CPU
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.model_loaded = False
        self._load_hubert_model()

    @classmethod
    def get_instance(cls, model_path=None):
        if cls._instance is None or model_path is not None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def _load_hubert_model(self):
        """
        Loads the trained HuBERT_D checkpoint strictly.
        Fails clearly if the checkpoint is missing.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"HuBERT_D checkpoint not found at: {self.model_path}\n"
                "Please verify that hubert_stage_B_best.pt is in v2/models/Hubert_D/."
            )

        from transformers import HubertConfig, HubertModel

        # Load HuBERT configuration offline
        if os.path.exists(self.config_path):
            config = HubertConfig.from_json_file(self.config_path)
        else:
            config = HubertConfig.from_pretrained(MODEL_CONFIG["pretrained_model"], local_files_only=True)

        raw_hubert = HubertModel(config)
        self.model = HubertMultiLabelClassifier(raw_hubert, num_labels=3)

        # Load weights
        checkpoint = torch.load(self.model_path, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {self.model_path} missing required 'model_state_dict' key.")

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True

        # Ensure optimal thresholds from checkpoint if present
        if "optimal_thresholds" in checkpoint:
            self.thresholds = checkpoint["optimal_thresholds"]

    def predict_window(self, waveform, sample_rate=16000):
        """
        Executes multi-label disfluency inference on a 3.0-second audio window.
        Input:
            waveform: 1D numpy array, list, or torch tensor of float32 audio samples in [-1.0, 1.0]
            sample_rate: Expected 16,000 Hz
        Returns:
            dict containing probabilities, binary predictions, fluency flag, and metadata.
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("HuBERT_D model is not loaded. Cannot perform inference.")

        # Ensure numpy float32
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.detach().cpu().numpy().astype(np.float32).flatten()
        else:
            audio_np = np.asarray(waveform, dtype=np.float32).flatten()

        # Handle shape / padding to target_samples (48,000)
        if len(audio_np) < self.target_samples:
            pad = np.zeros(self.target_samples - len(audio_np), dtype=np.float32)
            audio_np = np.concatenate([audio_np, pad])
        elif len(audio_np) > self.target_samples:
            audio_np = audio_np[:self.target_samples]

        # Calculate signal RMS energy
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        if rms < 1e-4:
            # Silent / inaudible window is defined as fluent
            return {
                "probabilities": {"Repetition": 0.0, "Prolongation": 0.0, "Block": 0.0},
                "predictions": {"Repetition": False, "Prolongation": False, "Block": False},
                "is_fluent": True,
                "active_labels": [],
                "model_name": self.model_name,
                "model_id": self.model_id,
                "thresholds": self.thresholds,
                "rms_energy": rms,
                "model_loaded": True
            }

        # PyTorch forward pass with HuBERT_D
        input_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs_tensor = torch.sigmoid(logits)[0]
            probs_np = probs_tensor.detach().cpu().numpy()

        probs = {
            c: round(float(probs_np[i]), 4)
            for i, c in enumerate(self.class_names)
        }

        preds = {
            c: bool(probs[c] >= self.thresholds[c])
            for c in self.class_names
        }

        is_fluent = not any(preds.values())
        active_labels = [c for c in self.class_names if preds[c]]

        return {
            "probabilities": probs,
            "predictions": preds,
            "is_fluent": is_fluent,
            "active_labels": active_labels,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "thresholds": self.thresholds,
            "rms_energy": round(rms, 6),
            "model_loaded": True
        }

    def get_model_info(self):
        """Returns runtime model metadata for logging and API/dashboard display."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "pretrained_model": MODEL_CONFIG["pretrained_model"],
            "targets": self.class_names,
            "thresholds": self.thresholds,
            "sample_rate": self.sample_rate,
            "window_duration_sec": self.window_duration,
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "checkpoint_path": self.model_path,
            "test_metrics": MODEL_CONFIG.get("test_metrics")
        }


def get_inference_engine(model_path=None):
    """Factory helper returning singleton VoxFlowInferenceEngine instance."""
    return VoxFlowInferenceEngine.get_instance(model_path=model_path)
