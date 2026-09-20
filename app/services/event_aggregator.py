#!/usr/bin/env python3
"""
VoxFlow — Temporal Event Aggregator
Clusters overlapping and adjacent sliding-window detections into discrete clinical events.
Operates independently on each disfluency category (Repetition, Prolongation, Block)
to support true multi-label stuttering detection without winner-take-all suppression.
"""

def aggregate_window_events(session_id, windows, min_support_windows=1, merge_gap_sec=1.5):
    """
    Groups contiguous windows detecting the same disfluency into unified temporal events.
    Input:
        session_id: Unique string session identifier
        windows: List of window dicts with keys: start_time, end_time, probabilities, predictions
    Returns:
        List of aggregated event dicts:
        [
            {
                "session_id": str,
                "event_type": "Repetition" | "Prolongation" | "Block",
                "start_time": float,
                "end_time": float,
                "confidence": float,
                "supporting_windows": int
            }, ...
        ]
    """
    target_classes = ["Repetition", "Prolongation", "Block"]
    aggregated_events = []

    for event_type in target_classes:
        # Extract windows active for this disfluency
        active_windows = [
            w for w in windows if w["predictions"].get(event_type, False)
        ]

        if not active_windows:
            continue

        # Sort by start_time
        active_windows = sorted(active_windows, key=lambda w: w["start_time"])

        clusters = []
        current_cluster = [active_windows[0]]

        for i in range(1, len(active_windows)):
            w_prev = current_cluster[-1]
            w_curr = active_windows[i]

            # Merge if overlapping or within merge_gap_sec
            if w_curr["start_time"] <= w_prev["end_time"] + merge_gap_sec:
                current_cluster.append(w_curr)
            else:
                clusters.append(current_cluster)
                current_cluster = [w_curr]
        clusters.append(current_cluster)

        # Convert clusters to events
        for cluster in clusters:
            if len(cluster) >= min_support_windows:
                start_t = cluster[0]["start_time"]
                end_t = max(w["end_time"] for w in cluster)
                conf = float(sum(w["probabilities"][event_type] for w in cluster) / len(cluster))

                aggregated_events.append({
                    "session_id": session_id,
                    "event_type": event_type,
                    "start_time": round(float(start_t), 2),
                    "end_time": round(float(end_t), 2),
                    "confidence": round(conf, 4),
                    "supporting_windows": len(cluster)
                })

    # Sort final events chronologically by start_time
    aggregated_events = sorted(aggregated_events, key=lambda e: (e["start_time"], e["event_type"]))
    return aggregated_events
