#!/usr/bin/env python3
"""
VoxFlow V2 — Clinical Session Monitoring Dashboard
Streamlit interface for continuous speech recording analysis, sliding-window disfluency
detection with HuBERT_D, interactive multi-label event timelines, and fluency trend tracking.
"""

import os
import sys
import pathlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup Python path to include project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MODEL_CONFIG, SESSION_CONFIG
from app.db.database import (
    init_db,
    save_session,
    get_all_sessions,
    get_session_details,
    get_latest_session
)
from app.services.session_engine import process_continuous_session
from inference.engine import get_inference_engine

# Initialize database schema
init_db()

# Page configuration
st.set_page_config(
    page_title="VoxFlow — Speech Fluency Analysis Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .vox-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }
    .vox-stat-label {
        font-size: 12px;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .vox-stat-val {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400E;
        font-size: 13px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🎙️ VoxFlow V2")
st.sidebar.caption("Continuous Speech Fluency Monitoring")

page = st.sidebar.radio(
    "Navigation",
    ["Analyze Session", "Current Session", "Session History", "Fluency Trends", "Model Information"]
)

st.sidebar.divider()
engine = get_inference_engine()
info = engine.get_model_info()

st.sidebar.markdown(f"**Active Model:** `{info['model_id']}`")
st.sidebar.caption(f"**Backbone:** `{info['model_name']}`")
st.sidebar.caption(f"**Device:** `{info['device'].upper()}`")
st.sidebar.caption(f"**Locked Thresholds:**")
for k, v in info['thresholds'].items():
    st.sidebar.caption(f"• {k}: **{v:.2f}**")

st.sidebar.divider()
st.sidebar.caption(f"Test Macro F1: **0.4599**")
st.sidebar.caption(f"Test ROC-AUC: **0.8099**")


# =============================================================================
# Helper Visualizations & Data Formatting
# =============================================================================
def create_probability_timeline(windows, thresholds):
    """Generates an interactive Plotly timeline of window-level disfluency probabilities."""
    if not windows:
        return None
    timeline_rows = []
    for w in windows:
        start_t = w.get("start_time", 0.0)
        prob_rep = w.get("prob_repetition") if "prob_repetition" in w else w.get("probabilities", {}).get("Repetition", 0.0)
        prob_pro = w.get("prob_prolongation") if "prob_prolongation" in w else w.get("probabilities", {}).get("Prolongation", 0.0)
        prob_blk = w.get("prob_block") if "prob_block" in w else w.get("probabilities", {}).get("Block", 0.0)
        timeline_rows.append({
            "Time (s)": start_t,
            "Repetition": float(prob_rep or 0.0),
            "Prolongation": float(prob_pro or 0.0),
            "Block": float(prob_blk or 0.0)
        })
    df_time = pd.DataFrame(timeline_rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_time["Time (s)"], y=df_time["Repetition"], name="Repetition", line=dict(color="#EF4444", width=2.5), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df_time["Time (s)"], y=df_time["Prolongation"], name="Prolongation", line=dict(color="#F59E0B", width=2.5), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df_time["Time (s)"], y=df_time["Block"], name="Block", line=dict(color="#3B82F6", width=2.5), mode="lines+markers"))

    th_rep = thresholds.get("Repetition", 0.77)
    th_pro = thresholds.get("Prolongation", 0.79)
    th_blk = thresholds.get("Block", 0.57)

    fig.add_hline(y=th_rep, line_dash="dash", line_color="#EF4444", annotation_text=f"Rep Th: {th_rep:.2f}", annotation_position="top right")
    fig.add_hline(y=th_pro, line_dash="dash", line_color="#F59E0B", annotation_text=f"Pro Th: {th_pro:.2f}", annotation_position="top right")
    fig.add_hline(y=th_blk, line_dash="dash", line_color="#3B82F6", annotation_text=f"Blk Th: {th_blk:.2f}", annotation_position="top right")

    fig.update_layout(
        xaxis_title="Session Time (seconds)",
        yaxis_title="Model Posterior Probability",
        yaxis_range=[0.0, 1.05],
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_event_timeline(events):
    """Generates a horizontal timeline chart for detected disfluency events."""
    if not events:
        return None
    color_map = {
        "Repetition": "#EF4444",
        "Prolongation": "#F59E0B",
        "Block": "#3B82F6"
    }
    fig = go.Figure()
    for ev in events:
        ev_type = ev.get("event_type", "Event")
        start = float(ev.get("start_time", 0.0))
        end = float(ev.get("end_time", start + 1.0))
        dur = round(end - start, 2)
        conf = round(float(ev.get("confidence", 0.0)), 2)
        color = color_map.get(ev_type, "#6366F1")
        
        fig.add_trace(go.Bar(
            name=ev_type,
            x=[dur],
            y=[ev_type],
            base=[start],
            orientation="h",
            marker=dict(color=color),
            hovertemplate=f"<b>{ev_type}</b><br>Start: {start}s<br>End: {end}s<br>Duration: {dur}s<br>Confidence: {conf}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title="Session Time (seconds)",
        yaxis_title="Disfluency Category",
        barmode="overlay",
        height=220,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def create_window_audit_df(windows):
    """Formats raw window-level model predictions for transparent audit inspection."""
    if not windows:
        return pd.DataFrame()
    rows = []
    for i, w in enumerate(windows):
        idx = w.get("window_index", w.get("window_id", i))
        start_t = w.get("start_time", 0.0)
        end_t = w.get("end_time", 0.0)
        prob_rep = w.get("prob_repetition") if "prob_repetition" in w else w.get("probabilities", {}).get("Repetition", 0.0)
        prob_pro = w.get("prob_prolongation") if "prob_prolongation" in w else w.get("probabilities", {}).get("Prolongation", 0.0)
        prob_blk = w.get("prob_block") if "prob_block" in w else w.get("probabilities", {}).get("Block", 0.0)
        
        is_fluent = w.get("is_fluent")
        if isinstance(is_fluent, (int, float)):
            fluent_str = "Fluent" if is_fluent == 1 else "Disfluent"
        elif isinstance(is_fluent, bool):
            fluent_str = "Fluent" if is_fluent else "Disfluent"
        else:
            fluent_str = str(is_fluent)

        labels = w.get("active_labels", [])
        if isinstance(labels, list):
            active_str = ", ".join(labels) if labels else "None (Fluent)"
        elif isinstance(labels, str):
            active_str = labels if labels else "None (Fluent)"
        else:
            active_str = str(labels)

        conf = float(w.get("confidence", 0.0))

        rows.append({
            "Window #": idx,
            "Start Time (s)": round(float(start_t), 2),
            "End Time (s)": round(float(end_t), 2),
            "Repetition Prob": round(float(prob_rep or 0.0), 4),
            "Prolongation Prob": round(float(prob_pro or 0.0), 4),
            "Block Prob": round(float(prob_blk or 0.0), 4),
            "Active Labels": active_str,
            "Status": fluent_str,
            "Confidence": round(conf, 4)
        })
    return pd.DataFrame(rows)


# =============================================================================
# 1. Analyze Session
# =============================================================================
if page == "Analyze Session":
    st.title("🎙️ Speech Session Analysis & Ingestion")
    st.markdown("VoxFlow supports continuous speech processing via two operational workflows: direct WAV file upload for clinical inspection and hardware audio capture via the ESP32 serial bridge.")

    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <strong>Clinical Notice:</strong> VoxFlow is an engineering research prototype designed for automated speech fluency monitoring
        and temporal event localization. It is <strong>not a medical diagnostic system</strong>.
    </div>
    """, unsafe_allow_html=True)

    tab_wav, tab_hw = st.tabs(["📁 A. WAV Upload & Analysis", "📡 B. ESP32 Phase-1 Hardware Capture"])

    with tab_wav:
        st.subheader("WAV Upload & Analysis")
        st.caption("Upload a speech recording (WAV format, 16 kHz mono) for sliding-window inference with the locked HuBERT_D model.")

        uploaded_file = st.file_uploader("Select Speech Audio File (WAV)", type=["wav"], key="wav_uploader")

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            col_btn, _ = st.columns([1, 4])
            with col_btn:
                run_analysis = st.button("Run Full Session Analysis", type="primary", use_container_width=True)

            if run_analysis:
                with st.spinner("Processing continuous audio through HuBERT_D sliding-window pipeline..."):
                    try:
                        audio_bytes = uploaded_file.read()
                        summary, windows, events = process_continuous_session(
                            audio_source=audio_bytes,
                            filename=uploaded_file.name,
                            engine=engine
                        )
                        # Persist session in SQLite
                        save_session(summary, windows, events)

                        st.success(f"Analysis complete! Session ID: `{summary['session_id']}`")

                        # Key Metrics Cards
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Session Duration", f"{summary['duration_sec']:.1f}s")
                        m2.metric("Total Windows (3s)", summary['total_windows'])
                        m3.metric("Fluent Windows", summary['fluent_windows'])
                        m4.metric("Session Fluency Ratio", f"{summary['fluency_ratio']*100:.1f}%")

                        # Disfluency Event Counts
                        e1, e2, e3 = st.columns(3)
                        e1.metric(f"Repetitions (Th: {info['thresholds'].get('Repetition', 0.77):.2f})", summary['repetition_count'])
                        e2.metric(f"Prolongations (Th: {info['thresholds'].get('Prolongation', 0.79):.2f})", summary['prolongation_count'])
                        e3.metric(f"Blocks (Th: {info['thresholds'].get('Block', 0.57):.2f})", summary['block_count'])

                        st.divider()

                        # Interactive Probability Timeline Chart
                        st.subheader("Temporal Multi-Label Probability Flow")
                        fig_prob = create_probability_timeline(windows, info["thresholds"])
                        if fig_prob:
                            st.plotly_chart(fig_prob, use_container_width=True)

                        # Window-level audit table
                        st.subheader("Window-Level Model Output")
                        df_audit = create_window_audit_df(windows)
                        if not df_audit.empty:
                            st.dataframe(df_audit, use_container_width=True)

                        st.divider()

                        # Aggregated Event Timeline
                        st.subheader("Aggregated Disfluency Events")
                        if events:
                            df_events = pd.DataFrame(events)
                            df_events["duration_sec"] = (df_events["end_time"] - df_events["start_time"]).round(2)
                            st.dataframe(
                                df_events[["event_type", "start_time", "end_time", "duration_sec", "confidence", "supporting_windows"]],
                                column_config={
                                    "event_type": "Disfluency Category",
                                    "start_time": "Start Time (s)",
                                    "end_time": "End Time (s)",
                                    "duration_sec": "Duration (s)",
                                    "confidence": "Avg Confidence",
                                    "supporting_windows": "Support Windows"
                                },
                                use_container_width=True
                            )
                            fig_ev = create_event_timeline(events)
                            if fig_ev:
                                st.plotly_chart(fig_ev, use_container_width=True)
                        else:
                            st.info("No stuttering events detected above locked decision thresholds. Speech is classified as fluent.")

                    except Exception as ex:
                        st.error(f"Error processing session: {ex}")

    with tab_hw:
        st.subheader("ESP32 Phase-1 Hardware Capture Workflow")
        st.markdown("""
        The Phase-1 hardware integration captures 16 kHz 16-bit PCM audio from the **ESP32 + INMP441** microphone module over USB serial.
        Hardware capture is managed by the external Python serial bridge utility.
        """)

        st.markdown("#### Hardware End-to-End Pipeline")
        st.code("""
ESP32 + INMP441
      ↓
USB / COM8
      ↓
Python Serial Bridge
      ↓
10-second WAV
      ↓
VoxFlow REST API
      ↓
HuBERT-D Analysis
      ↓
SQLite
      ↓
Streamlit Dashboard
""", language="text")

        st.markdown("#### Running Hardware Capture")
        st.markdown("Execute the serial bridge script from your terminal to stream audio from the ESP32 and automatically submit it to the VoxFlow API:")
        st.code("python tools/esp32_serial_bridge.py --port COM8", language="bash")

        st.markdown("To capture and save a local WAV file without uploading to the API:")
        st.code("python tools/esp32_serial_bridge.py --port COM8 --skip-api", language="bash")

        st.markdown("#### Architecture Notes")
        st.markdown("""
        - **Fixed Capture Duration:** Phase-1 recording duration is fixed at **10.0 seconds** (320,000 bytes at 16 kHz 16-bit mono PCM).
        - **Default Serial Port:** The standard hardware port is `COM8` (configurable via `--port`).
        - **Automatic Analysis & Ingestion:** The Python serial bridge sends the captured WAV to the existing VoxFlow session API (`POST /api/v1/session/<session_id>/audio`), then stops and analyzes the session; the results are saved to SQLite.
        - **Result Inspection:** Once the bridge completes, view the analysis under **Current Session** or **Session History**.
        - **Decoupled Architecture:** The Streamlit dashboard does not directly communicate with serial hardware ports.
        """)


# =============================================================================
# 2. Current Session
# =============================================================================
elif page == "Current Session":
    st.title("🎙️ Current Session Monitoring")

    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.subheader("Latest Analyzed Session")
    with col_refresh:
        if st.button("🔄 Refresh Latest Session", use_container_width=True):
            st.rerun()

    latest = get_latest_session()

    if not latest:
        st.info("No analyzed sessions found in the SQLite database yet. Please analyze a WAV file or run the ESP32 serial bridge to record a session.")
    else:
        # Session Metadata Card
        with st.container():
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"**Session ID:** `{latest['id']}`")
                created_ts = latest.get("created_at") or latest.get("started_at", "N/A")
                st.markdown(f"**Recorded / Ingested:** `{created_ts}`")
                st.markdown(f"**Status:** `{latest.get('status', 'ANALYZED')}`")
            with col_m2:
                st.markdown(f"**Model Name:** `{latest.get('model_name', 'HuBERT Stage B Extended v2')}`")
                audio_file = latest.get("audio_filename") or (os.path.basename(latest["audio_path"]) if latest.get("audio_path") else "N/A")
                st.markdown(f"**Audio File:** `{audio_file}`")

        # Audio playback if audio file exists locally
        if latest.get("audio_path") and os.path.exists(latest["audio_path"]):
            st.audio(latest["audio_path"], format="audio/wav")

        st.divider()

        # Key Summary Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{latest['duration_sec']:.1f}s")
        c2.metric("Total Windows (3s)", latest['total_windows'])
        c3.metric("Fluent Windows", latest['fluent_windows'])
        c4.metric("Session Fluency Ratio", f"{latest['fluency_ratio']*100:.1f}%")

        # Disfluency Event Counts
        e1, e2, e3 = st.columns(3)
        e1.metric(f"Repetitions (Th: {info['thresholds'].get('Repetition', 0.77):.2f})", latest['repetition_count'])
        e2.metric(f"Prolongations (Th: {info['thresholds'].get('Prolongation', 0.79):.2f})", latest['prolongation_count'])
        e3.metric(f"Blocks (Th: {info['thresholds'].get('Block', 0.57):.2f})", latest['block_count'])

        st.divider()

        # Temporal Probability Timeline
        st.subheader("Temporal Multi-Label Probability Timeline")
        windows = latest.get("windows", [])
        if windows:
            fig_prob = create_probability_timeline(windows, info["thresholds"])
            if fig_prob:
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.caption("No window probability points stored for this session.")

        # Window-Level Model Output Audit Table
        st.subheader("Window-Level Model Output")
        st.caption("Transparent audit log of 3.0s sliding-window multi-label predictions across the recording duration.")
        if windows:
            df_audit = create_window_audit_df(windows)
            if not df_audit.empty:
                st.dataframe(df_audit, use_container_width=True)
        else:
            st.info("No window records found for this session.")

        st.divider()

        # Aggregated Disfluency Events
        st.subheader("Aggregated Disfluency Events")
        events = latest.get("events", [])
        if events:
            df_ev = pd.DataFrame(events)
            df_ev["duration_sec"] = (df_ev["end_time"] - df_ev["start_time"]).round(2)
            st.dataframe(
                df_ev[["event_type", "start_time", "end_time", "duration_sec", "confidence", "supporting_windows"]],
                column_config={
                    "event_type": "Disfluency Category",
                    "start_time": "Start Time (s)",
                    "end_time": "End Time (s)",
                    "duration_sec": "Duration (s)",
                    "confidence": "Avg Confidence",
                    "supporting_windows": "Support Windows"
                },
                use_container_width=True
            )
            fig_ev = create_event_timeline(events)
            if fig_ev:
                st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info("No stuttering events detected in this session.")


# =============================================================================
# 3. Session History
# =============================================================================
elif page == "Session History":
    st.title("📊 Clinical Session History")
    st.markdown("All continuous speech recordings evaluated and saved in local SQLite storage.")

    sessions = get_all_sessions(limit=50)

    if not sessions:
        st.info("No sessions recorded in the database yet.")
    else:
        df_sessions = pd.DataFrame(sessions)
        st.dataframe(
            df_sessions[[
                "id", "created_at", "duration_sec", "fluency_ratio",
                "repetition_count", "prolongation_count", "block_count", "status"
            ]],
            column_config={
                "id": "Session ID",
                "created_at": "Timestamp",
                "duration_sec": "Duration (s)",
                "fluency_ratio": st.column_config.ProgressColumn("Fluency Ratio", min_value=0.0, max_value=1.0, format="%.2f"),
                "repetition_count": "Repetitions",
                "prolongation_count": "Prolongations",
                "block_count": "Blocks",
                "status": "State"
            },
            use_container_width=True
        )

        st.subheader("Inspect Specific Session")
        selected_id = st.selectbox("Select Session ID to inspect:", [s["id"] for s in sessions])
        if selected_id:
            details = get_session_details(selected_id)
            if details:
                st.markdown(f"**Session:** `{selected_id}` | Duration: `{details['duration_sec']:.1f}s` | Fluency: `{details['fluency_ratio']*100:.1f}%`")
                if details.get("audio_path") and os.path.exists(details["audio_path"]):
                    st.audio(details["audio_path"], format="audio/wav")

                # Show Probability Timeline
                if details.get("windows"):
                    st.markdown("##### Temporal Probability Flow")
                    fig_det = create_probability_timeline(details["windows"], info["thresholds"])
                    if fig_det:
                        st.plotly_chart(fig_det, use_container_width=True)

                if details.get("events"):
                    st.markdown("##### Aggregated Events")
                    st.dataframe(pd.DataFrame(details["events"]), use_container_width=True)
                    fig_ev_det = create_event_timeline(details["events"])
                    if fig_ev_det:
                        st.plotly_chart(fig_ev_det, use_container_width=True)


# =============================================================================
# 4. Fluency Trends
# =============================================================================
elif page == "Fluency Trends":
    st.title("📈 Speech Fluency & Longitudinal Trends")
    st.markdown("Tracks fluency progression across recorded sessions over time.")

    sessions = get_all_sessions(limit=50)

    if len(sessions) < 2:
        st.info("Record at least 2 sessions to visualize longitudinal fluency trends.")
    else:
        df_trend = pd.DataFrame(sessions).sort_values("created_at")
        df_trend["Session Number"] = range(1, len(df_trend) + 1)
        df_trend["Fluency %"] = (df_trend["fluency_ratio"] * 100).round(1)

        fig_trend = px.line(
            df_trend,
            x="Session Number",
            y="Fluency %",
            markers=True,
            title="Session Fluency Ratio Progression",
            hover_data=["id", "created_at"]
        )
        fig_trend.update_layout(yaxis_range=[0, 105], height=380)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("Disfluency Breakdown Across Sessions")
        fig_bar = px.bar(
            df_trend,
            x="Session Number",
            y=["repetition_count", "prolongation_count", "block_count"],
            title="Event Frequency per Session",
            barmode="group",
            labels={"value": "Detected Event Count", "variable": "Disfluency Category"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# 5. Model Information
# =============================================================================
elif page == "Model Information":
    st.title("🔬 Model Architecture & Locked Benchmark")
    st.markdown("Detailed specifications of the active production model and validation-calibrated decision thresholds.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Active Model: HuBERT_D")
        st.markdown(f"- **Architecture:** `HuBERT Multi-Label Classifier`")
        st.markdown(f"- **Pretrained Backbone:** `facebook/hubert-base-ls960` (12 transformer layers, 768 dim)")
        st.markdown(f"- **Tuned Layers:** Final 2 transformer encoder layers (layers 10 and 11)")
        st.markdown(f"- **Classification Head:** `Linear(768, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 3)`")
        st.markdown(f"- **Output Representation:** 3 independent sigmoid probabilities")
        st.markdown(f"- **Device In Use:** `{info['device'].upper()}`")

    with col2:
        st.subheader("Locked Decision Thresholds")
        st.markdown("""
        Thresholds were optimized exclusively on the 6,632-clip validation split via independent grid search to maximize per-class F1 score:
        """)
        st.markdown(f"- **Repetition:** `{info['thresholds'].get('Repetition', 0.77):.2f}` *(Prevents false alarms on fluent lexical restarts)*")
        st.markdown(f"- **Prolongation:** `{info['thresholds'].get('Prolongation', 0.79):.2f}` *(High precision on sustained phonemes)*")
        st.markdown(f"- **Block:** `{info['thresholds'].get('Block', 0.57):.2f}` *(Balanced detection on silent tense pauses)*")

    st.divider()
    st.subheader("Locked Test Set Benchmark Comparison (6,812 Clips)")
    bench_data = [
        {"Model": "Model 1: LinearSVC Baseline", "Features": "162 MFCC + Stats", "Macro F1": 0.3121, "Mean PR-AUC": 0.2545, "Mean ROC-AUC": 0.6633, "Status": "Baseline"},
        {"Model": "Model 3: Wav2Vec2 Mean+Max Stage C", "Features": "facebook/wav2vec2-base", "Macro F1": 0.4498, "Mean PR-AUC": 0.4288, "Mean ROC-AUC": 0.8020, "Status": "Competitor"},
        {"Model": "Model 2: HuBERT_D (Selected Final)", "Features": "facebook/hubert-base-ls960", "Macro F1": 0.4599, "Mean PR-AUC": 0.4532, "Mean ROC-AUC": 0.8099, "Status": "Winning Production Model"}
    ]
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True)

