#!/usr/bin/env python3
"""
VoxFlow — Session Summary Service
Calculates session-level statistics, disfluency counts, and Session Fluency Ratio.
"""

def calculate_session_summary(session_id, duration_sec, windows, events, model_name="VoxFlow"):
    """
    Computes summary metrics for a continuous speech session.
    A window is fluent if and only if no disfluency label is active.
    Fluency ratio is bounded between 0.0 and 1.0.
    """
    total_windows = len(windows)
    fluent_windows = sum(1 for w in windows if w["is_fluent"])

    rep_events = sum(1 for e in events if e["event_type"] == "Repetition")
    pro_events = sum(1 for e in events if e["event_type"] == "Prolongation")
    blk_events = sum(1 for e in events if e["event_type"] == "Block")

    fluency_ratio = (fluent_windows / total_windows) if total_windows > 0 else 1.0

    return {
        "session_id": session_id,
        "duration_sec": round(float(duration_sec), 2),
        "total_windows": total_windows,
        "fluent_windows": fluent_windows,
        "fluency_ratio": round(float(fluency_ratio), 4),
        "repetition_count": rep_events,
        "prolongation_count": pro_events,
        "block_count": blk_events,
        "model_name": model_name,
        "events": events
    }
