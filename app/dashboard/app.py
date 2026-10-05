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

# Setup Python path to include project root and prevent app.py shadowing
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
dashboard_dir = str(pathlib.Path(__file__).resolve().parent)
while dashboard_dir in sys.path:
    sys.path.remove(dashboard_dir)

root_str = str(PROJECT_ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

# Clear any shadowed 'app' module without __path__
if 'app' in sys.modules and not hasattr(sys.modules['app'], '__path__'):
    del sys.modules['app']

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
# 1. Analyze Session
# =============================================================================
if page == "Analyze Session":
    st.title("🎙️ Analyze Speech Session")
    st.markdown("Process continuous speech audio (WAV format, 16 kHz mono) through sliding-window inference with the locked **HuBERT_D** model.")

    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <strong>Clinical Notice:</strong> VoxFlow is an engineering research prototype designed for automated speech fluency monitoring
        and temporal event localization. It is <strong>not a medical diagnostic system</strong>.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select Speech Audio File (WAV)", type=["wav"])

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
                    e1.metric("Repetitions (Th: 0.77)", summary['repetition_count'])
                    e2.metric("Prolongations (Th: 0.79)", summary['prolongation_count'])
                    e3.metric("Blocks (Th: 0.57)", summary['block_count'])

                    # Aggregated Event Timeline Table
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
                    else:
                        st.info("No stuttering events detected above locked decision thresholds. Speech is classified as fluent.")

                    # Interactive Probability Timeline Chart
                    st.subheader("Temporal Multi-Label Probability Flow")
                    timeline_rows = []
                    for w in windows:
                        timeline_rows.append({
                            "Time (s)": w["start_time"],
                            "Repetition": w["probabilities"]["Repetition"],
                            "Prolongation": w["probabilities"]["Prolongation"],
                            "Block": w["probabilities"]["Block"]
                        })
                    df_time = pd.DataFrame(timeline_rows)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_time["Time (s)"], y=df_time["Repetition"], name="Repetition", line=dict(color="#EF4444", width=2)))
                    fig.add_trace(go.Scatter(x=df_time["Time (s)"], y=df_time["Prolongation"], name="Prolongation", line=dict(color="#F59E0B", width=2)))
                    fig.add_trace(go.Scatter(x=df_time["Time (s)"], y=df_time["Block"], name="Block", line=dict(color="#3B82F6", width=2)))

                    # Add horizontal threshold reference lines
                    fig.add_hline(y=0.77, line_dash="dash", line_color="#EF4444", annotation_text="Rep Th: 0.77", annotation_position="top right")
                    fig.add_hline(y=0.79, line_dash="dash", line_color="#F59E0B", annotation_text="Pro Th: 0.79", annotation_position="top right")
                    fig.add_hline(y=0.57, line_dash="dash", line_color="#3B82F6", annotation_text="Blk Th: 0.57", annotation_position="top right")

                    fig.update_layout(
                        xaxis_title="Session Time (seconds)",
                        yaxis_title="Model Posterior Probability",
                        yaxis_range=[0.0, 1.05],
                        hovermode="x unified",
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=360
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as ex:
                    st.error(f"Error processing session: {ex}")


# =============================================================================
# 2. Current Session
# =============================================================================
elif page == "Current Session":
    st.title("🎙️ Current / Latest Analyzed Session")
    latest = get_latest_session()

    if not latest:
        st.info("No analyzed sessions found in the SQLite database yet. Please analyze a session first.")
    else:
        st.subheader(f"Session ID: `{latest['id']}`")
        st.caption(f"Recorded: {latest['created_at']} | Model: **{latest['model_name']}**")

        if latest.get("audio_path") and os.path.exists(latest["audio_path"]):
            st.audio(latest["audio_path"], format="audio/wav")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{latest['duration_sec']:.1f}s")
        c2.metric("Total Windows", latest['total_windows'])
        c3.metric("Fluent Windows", latest['fluent_windows'])
        c4.metric("Session Fluency Ratio", f"{latest['fluency_ratio']*100:.1f}%")

        e1, e2, e3 = st.columns(3)
        e1.metric("Repetitions", latest['repetition_count'])
        e2.metric("Prolongations", latest['prolongation_count'])
        e3.metric("Blocks", latest['block_count'])

        st.subheader("Aggregated Events Timeline")
        events = latest.get("events", [])
        if events:
            df_ev = pd.DataFrame(events)
            df_ev["duration_sec"] = (df_ev["end_time"] - df_ev["start_time"]).round(2)
            st.dataframe(
                df_ev[["event_type", "start_time", "end_time", "duration_sec", "confidence", "supporting_windows"]],
                use_container_width=True
            )
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
                st.write(f"**Session:** `{selected_id}` | Duration: `{details['duration_sec']:.1f}s` | Fluency: `{details['fluency_ratio']*100:.1f}%`")
                if details.get("events"):
                    st.dataframe(pd.DataFrame(details["events"]), use_container_width=True)


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
        st.markdown(f"- **Repetition:** `0.77` *(Prevents false alarms on fluent lexical restarts)*")
        st.markdown(f"- **Prolongation:** `0.79` *(High precision on sustained phonemes)*")
        st.markdown(f"- **Block:** `0.57` *(Balanced detection on silent tense pauses)*")

    st.divider()
    st.subheader("Locked Test Set Benchmark Comparison (6,812 Clips)")
    bench_data = [
        {"Model": "Model 1: LinearSVC Baseline", "Features": "162 MFCC + Stats", "Macro F1": 0.3121, "Mean PR-AUC": 0.2545, "Mean ROC-AUC": 0.6633, "Status": "Baseline"},
        {"Model": "Model 3: Wav2Vec2 Mean+Max Stage C", "Features": "facebook/wav2vec2-base", "Macro F1": 0.4498, "Mean PR-AUC": 0.4288, "Mean ROC-AUC": 0.8020, "Status": "Competitor"},
        {"Model": "Model 2: HuBERT_D (Selected Final)", "Features": "facebook/hubert-base-ls960", "Macro F1": 0.4599, "Mean PR-AUC": 0.4532, "Mean ROC-AUC": 0.8099, "Status": "Winning Production Model"}
    ]
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True)
