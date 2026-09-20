#!/usr/bin/env python3
"""
VoxFlow V2 — Session Processing Engine
Handles end-to-end continuous recording analysis:
  Audio validation & normalization -> Sliding-window slicing (3.0s window, 1.0s step) ->
  HuBERT_D multi-label inference -> Temporal event aggregation -> Session summary.
"""

import io
import os
import wave
import uuid
import pathlib
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

from configs.config import SESSION_CONFIG, MODEL_CONFIG
from inference.engine import get_inference_engine
from app.services.event_aggregator import aggregate_window_events
from app.services.session_summary import calculate_session_summary


class AudioValidationError(Exception):
    pass


def validate_and_load_session_audio(audio_path_or_bytes, sample_rate=16000):
    """
    Validates and standardizes session audio:
    - Supports file path or raw bytes (WAV or raw 16-bit PCM from ESP32).
    - Ensures 16 kHz mono float32 in [-1.0, 1.0].
    - Verifies audible speech signal (RMS energy).
    - Checks duration: must be >= 3.0s (single window) up to standard session max.
    Returns:
        tuple: (audio_array, sample_rate, duration_seconds)
    """
    audio = None
    sr = None

    if isinstance(audio_path_or_bytes, (str, os.PathLike)):
        if not os.path.exists(audio_path_or_bytes):
            raise AudioValidationError(f"Audio file not found: {audio_path_or_bytes}")
        try:
            sr, audio = wavfile.read(str(audio_path_or_bytes))
        except Exception:
            with open(audio_path_or_bytes, "rb") as f:
                raw_bytes = f.read()
            return validate_and_load_session_audio(raw_bytes, sample_rate=sample_rate)

    elif isinstance(audio_path_or_bytes, bytes):
        raw_bytes = audio_path_or_bytes
        # 1. Try reading as standard WAV
        try:
            sr, audio = wavfile.read(io.BytesIO(raw_bytes))
        except Exception:
            # 2. Try standard wave library
            try:
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)
                    if n_channels > 1:
                        audio = audio.reshape(-1, n_channels).mean(axis=1)
            except Exception:
                # 3. Handle raw 16-bit PCM stream from ESP32 Dev Module
                audio = np.frombuffer(raw_bytes, dtype=np.int16)
                sr = sample_rate

    else:
        raise AudioValidationError("Audio input must be a valid file path or raw bytes.")

    if audio is None or len(audio) == 0:
        raise AudioValidationError("Audio input is empty or invalid.")

    # Convert to float32 normalized [-1.0, 1.0]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    # Convert stereo / multi-channel to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz if necessary
    if sr != sample_rate:
        num_target = int(len(audio) * sample_rate / sr)
        audio = signal.resample(audio, num_target).astype(np.float32)
        sr = sample_rate

    duration_sec = len(audio) / sr

    # Minimum duration check (at least one full 3-second analysis window)
    if duration_sec < SESSION_CONFIG["min_duration_sec"]:
        raise AudioValidationError(
            f"Session duration ({duration_sec:.2f}s) is shorter than the minimum "
            f"{SESSION_CONFIG['min_duration_sec']:.1f}s window."
        )

    # Check for complete silence
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-4:
        raise AudioValidationError("Session audio contains only silence or undetectable speech signal.")

    return audio, sr, duration_sec


# Backward-compatible alias
load_and_validate_audio = validate_and_load_session_audio


def process_session_windows(session_id, audio, sr, duration_sec, engine=None):
    """
    Slices session audio into 3.0s overlapping windows with a 1.0s step
    and runs real HuBERT_D multi-label inference on each window.
    """
    engine = engine or get_inference_engine()
    window_sec = SESSION_CONFIG["window_duration_sec"]  # 3.0s
    step_sec = SESSION_CONFIG["window_step_sec"]        # 1.0s
    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)

    windows = []
    window_id = 0
    start_sample = 0

    while start_sample + window_samples <= len(audio):
        end_sample = start_sample + window_samples
        start_time = round(start_sample / sr, 2)
        end_time = round(end_sample / sr, 2)

        window_waveform = audio[start_sample:end_sample]
        pred_res = engine.predict_window(window_waveform, sample_rate=sr)

        # Confidence: highest probability among active disfluencies, or max probability
        prob_rep = pred_res["probabilities"]["Repetition"]
        prob_pro = pred_res["probabilities"]["Prolongation"]
        prob_blk = pred_res["probabilities"]["Block"]
        confidence = max(prob_rep, prob_pro, prob_blk) if not pred_res["is_fluent"] else 0.99

        windows.append({
            "session_id": session_id,
            "window_id": window_id,
            "start_time": start_time,
            "end_time": end_time,
            "probabilities": pred_res["probabilities"],
            "predictions": pred_res["predictions"],
            "prob_repetition": prob_rep,
            "prob_prolongation": prob_pro,
            "prob_block": prob_blk,
            "is_fluent": pred_res["is_fluent"],
            "active_labels": pred_res["active_labels"],
            "confidence": round(float(confidence), 4),
            "model_name": pred_res["model_name"]
        })

        window_id += 1
        start_sample += step_samples

    return windows


def process_continuous_session(audio_source, session_id=None, filename="session.wav", engine=None):
    """
    High-level end-to-end continuous session processing pipeline.
    Validates audio -> Windowing -> HuBERT_D inference -> Event aggregation -> Session summary.
    Returns:
        tuple: (summary_dict, windows_list, events_list)
    """
    audio, sr, duration_sec = validate_and_load_session_audio(audio_source)
    session_id = session_id or f"sess_{uuid.uuid4().hex[:10]}"
    engine = engine or get_inference_engine()

    # 1. Slicing & Window Inference
    windows = process_session_windows(session_id, audio, sr, duration_sec, engine=engine)

    # 2. Temporal Multi-Label Event Aggregation
    events = aggregate_window_events(
        session_id=session_id,
        windows=windows,
        min_support_windows=1,
        merge_gap_sec=1.5
    )

    # 3. Session Summary Calculation
    summary = calculate_session_summary(
        session_id=session_id,
        duration_sec=duration_sec,
        windows=windows,
        events=events,
        model_name=engine.model_name
    )
    summary["audio_filename"] = filename

    return summary, windows, events
