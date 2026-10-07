#!/usr/bin/env python3
"""
VoxFlow V2 — Speech Fluency & Disfluency Session Analyzer Dashboard
Streamlit interface for continuous speech recording analysis, sliding-window disfluency
detection with HuBERT-D, event localization, session history, and longitudinal trends.
"""

import os
import sys
import pathlib
import datetime
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

# Initialize database schema safely
init_db()

# Page configuration
st.set_page_config(
    page_title="VoxFlow — Speech Fluency Session Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom Design System (Clean, Modern, Calm, Accessible)
# =============================================================================
st.markdown("""
<style>
    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        color: #1E293B;
    }
    
    /* App Header Banner */
    .app-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
    }
    .app-title {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 15px;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Content Cards */
    .vf-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Metric Cards */
    .vf-metric-container {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .vf-metric-card {
        flex: 1;
        min-width: 160px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    .vf-metric-card.primary {
        background: #F8FAFC;
        border-color: #CBD5E1;
    }
    .vf-metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .vf-metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
    }
    .vf-metric-val.fluency {
        color: #059669;
    }
    .vf-metric-val.rep {
        color: #D97706;
    }
    .vf-metric-val.pro {
        color: #2563EB;
    }
    .vf-metric-val.blk {
        color: #7C3AED;
    }

    /* Empty States */
    .vf-empty-state {
        text-align: center;
        padding: 40px 20px;
        background: #F8FAFC;
        border: 1px dashed #CBD5E1;
        border-radius: 12px;
        margin: 20px 0;
    }
    .vf-empty-title {
        font-size: 17px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 6px;
    }
    .vf-empty-desc {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 16px;
    }

    /* Subtle Notice Banner */
    .vf-notice-box {
        background-color: #F8FAFC;
        border-left: 3px solid #64748B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #475569;
        margin-top: 24px;
    }

    /* Pill Badges */
    .badge-rep {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-pro {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-blk {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-fluent {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Color Scheme for Event Categories
EVENT_COLORS = {
    "Repetition": "#D97706",     # Warm Amber
    "Prolongation": "#2563EB",   # Clear Blue
    "Block": "#7C3AED",          # Purple
    "Fluent": "#059669"          # Emerald Green
}

# =============================================================================
# Helper Formatting & Plotly Visualizations
# =============================================================================

def format_timestamp(ts_str):
    """Parses ISO timestamp into clean human-readable date string."""
    if not ts_str or ts_str == "N/A":
        return "Recent Session"
    try:
        # Handle string formats like '2026-10-06T17:15:29' or '2026-10-06 17:15:29'
        clean_ts = ts_str.replace("T", " ")
        if "." in clean_ts:
            clean_ts = clean_ts.split(".")[0]
        dt = datetime.datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d, %Y · %I:%M %p")
    except Exception:
        return str(ts_str)


def create_event_timeline(events, session_duration=None):
    """Generates a clean horizontal timeline chart for detected disfluency events."""
    if not events:
        return None

    fig = go.Figure()
    
    # Track min/max to set reasonable axis range
    max_time = session_duration or 0.0

    for ev in events:
        ev_type = ev.get("event_type", "Event")
        start = float(ev.get("start_time", 0.0))
        end = float(ev.get("end_time", start + 1.0))
        dur = round(end - start, 2)
        conf = round(float(ev.get("confidence", 0.0)) * 100, 1)
        color = EVENT_COLORS.get(ev_type, "#475569")
        if end > max_time:
            max_time = end
        
        fig.add_trace(go.Bar(
            name=ev_type,
            x=[dur],
            y=[ev_type],
            base=[start],
            orientation="h",
            marker=dict(
                color=color,
                line=dict(color="#FFFFFF", width=1.5)
            ),
            hovertemplate=(
                f"<b>{ev_type}</b><br>"
                f"Start: {start:.1f}s<br>"
                f"End: {end:.1f}s<br>"
                f"Duration: {dur:.1f}s<br>"
                f"Confidence: {conf:.0f}%<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(
            title="Session Time (seconds)",
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            range=[0, max(max_time + 1.0, 5.0)]
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=["Block", "Prolongation", "Repetition"],
            showgrid=False
        ),
        barmode="overlay",
        height=200,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=20, t=10, b=35)
    )
    return fig


def create_event_table(events):
    """Formats detected events into a clean, human-readable dataframe."""
    if not events:
        return pd.DataFrame()
    rows = []
    for ev in events:
        start = float(ev.get("start_time", 0.0))
        end = float(ev.get("end_time", start + 1.0))
        dur = round(end - start, 2)
        conf = float(ev.get("confidence", 0.0)) * 100
        rows.append({
            "Type": ev.get("event_type", "Event"),
            "Start": f"{start:.1f}s",
            "End": f"{end:.1f}s",
            "Duration": f"{dur:.1f}s",
            "Confidence": f"{conf:.0f}%"
        })
    return pd.DataFrame(rows)


def create_probability_timeline(windows, thresholds):
    """Generates an interactive Plotly timeline for window-level probabilities (Advanced Analysis)."""
    if not windows:
        return None
    timeline_rows = []
    for w in windows:
        start_t = float(w.get("start_time", 0.0))
        prob_rep = float(w.get("prob_repetition") if "prob_repetition" in w else w.get("probabilities", {}).get("Repetition", 0.0))
        prob_pro = float(w.get("prob_prolongation") if "prob_prolongation" in w else w.get("probabilities", {}).get("Prolongation", 0.0))
        prob_blk = float(w.get("prob_block") if "prob_block" in w else w.get("probabilities", {}).get("Block", 0.0))
        timeline_rows.append({
            "Time (s)": start_t,
            "Repetition": prob_rep,
            "Prolongation": prob_pro,
            "Block": prob_blk
        })
    df_time = pd.DataFrame(timeline_rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_time["Time (s)"], y=df_time["Repetition"],
        name="Repetition", line=dict(color=EVENT_COLORS["Repetition"], width=2.5), mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=df_time["Time (s)"], y=df_time["Prolongation"],
        name="Prolongation", line=dict(color=EVENT_COLORS["Prolongation"], width=2.5), mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=df_time["Time (s)"], y=df_time["Block"],
        name="Block", line=dict(color=EVENT_COLORS["Block"], width=2.5), mode="lines+markers"
    ))

    # Threshold guidelines
    th_rep = thresholds.get("Repetition", 0.77)
    th_pro = thresholds.get("Prolongation", 0.79)
    th_blk = thresholds.get("Block", 0.57)

    fig.add_hline(y=th_rep, line_dash="dash", line_color=EVENT_COLORS["Repetition"], annotation_text=f"Rep Th: {th_rep:.2f}", annotation_position="top right")
    fig.add_hline(y=th_pro, line_dash="dash", line_color=EVENT_COLORS["Prolongation"], annotation_text=f"Pro Th: {th_pro:.2f}", annotation_position="top right")
    fig.add_hline(y=th_blk, line_dash="dash", line_color=EVENT_COLORS["Block"], annotation_text=f"Blk Th: {th_blk:.2f}", annotation_position="top right")

    fig.update_layout(
        xaxis_title="Session Time (seconds)",
        yaxis_title="Probability",
        yaxis_range=[0.0, 1.05],
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_window_audit_df(windows):
    """Formats raw window-level model predictions for transparent audit inspection."""
    if not windows:
        return pd.DataFrame()
    rows = []
    for i, w in enumerate(windows):
        idx = w.get("window_index", w.get("window_id", i))
        start_t = float(w.get("start_time", 0.0))
        end_t = float(w.get("end_time", start_t + 3.0))
        prob_rep = float(w.get("prob_repetition") if "prob_repetition" in w else w.get("probabilities", {}).get("Repetition", 0.0))
        prob_pro = float(w.get("prob_prolongation") if "prob_prolongation" in w else w.get("probabilities", {}).get("Prolongation", 0.0))
        prob_blk = float(w.get("prob_block") if "prob_block" in w else w.get("probabilities", {}).get("Block", 0.0))
        
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

        conf = float(w.get("confidence", 0.0)) * 100

        rows.append({
            "Window #": idx + 1 if isinstance(idx, int) else idx,
            "Start": f"{start_t:.1f}s",
            "End": f"{end_t:.1f}s",
            "Rep Prob": f"{prob_rep:.2f}",
            "Pro Prob": f"{prob_pro:.2f}",
            "Blk Prob": f"{prob_blk:.2f}",
            "Detections": active_str,
            "Status": fluent_str,
            "Confidence": f"{conf:.0f}%"
        })
    return pd.DataFrame(rows)


def render_session_report(session_data, engine):
    """Reusable visual report renderer for a session result or inspected session."""
    summary = session_data
    windows = session_data.get("windows", [])
    events = session_data.get("events", [])
    
    # 1. Summary Metric Cards
    fluency_pct = round(float(summary.get("fluency_ratio", 1.0)) * 100, 1)
    duration_s = float(summary.get("duration_sec", 0.0))
    rep_count = int(summary.get("repetition_count", 0))
    pro_count = int(summary.get("prolongation_count", 0))
    blk_count = int(summary.get("block_count", 0))
    
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="vf-metric-card primary">
            <div class="vf-metric-label">Session Fluency</div>
            <div class="vf-metric-val fluency">{fluency_pct:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="vf-metric-card">
            <div class="vf-metric-label">Repetition</div>
            <div class="vf-metric-val rep">{rep_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="vf-metric-card">
            <div class="vf-metric-label">Prolongation</div>
            <div class="vf-metric-val pro">{pro_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="vf-metric-card">
            <div class="vf-metric-label">Block</div>
            <div class="vf-metric-val blk">{blk_count}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Audio Playback
    audio_path = summary.get("audio_path")
    if audio_path and os.path.exists(audio_path):
        st.markdown("##### Audio Recording")
        st.audio(audio_path, format="audio/wav")

    st.markdown("---")

    # 3. Detected Events Section
    st.markdown("#### Detected Events")
    if events:
        fig_ev = create_event_timeline(events, session_duration=duration_s)
        if fig_ev:
            st.plotly_chart(fig_ev, use_container_width=True)

        df_ev = create_event_table(events)
        if not df_ev.empty:
            st.dataframe(df_ev, use_container_width=True, hide_index=True)
    else:
        st.info("No disfluency events were detected in this session.")

    # 4. Collapsible Advanced Analysis
    with st.expander("▸ Advanced Analysis (Model Probabilities & Window Audit)"):
        st.caption("Detailed window-level model posterior probabilities and sliding analysis window outputs.")
        
        info = engine.get_model_info() if engine else MODEL_CONFIG
        thresholds = info.get("thresholds", MODEL_CONFIG["locked_thresholds"])
        
        if windows:
            fig_prob = create_probability_timeline(windows, thresholds)
            if fig_prob:
                st.markdown("###### Multi-Label Probability Timeline")
                st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown("###### Window-Level Prediction Details")
            df_audit = create_window_audit_df(windows)
            if not df_audit.empty:
                st.dataframe(df_audit, use_container_width=True, hide_index=True)
        else:
            st.caption("No window-level audit data stored for this session.")


# =============================================================================
# Navigation Setup
# =============================================================================

# Initialize session state for navigation
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

if "last_analyzed_session" not in st.session_state:
    st.session_state["last_analyzed_session"] = None

# Sidebar
st.sidebar.markdown("""
<div style="margin-bottom: 12px;">
    <span style="font-size: 20px; font-weight: 800; color: #0F172A; letter-spacing: -0.01em;">VOXFLOW</span><br>
    <span style="font-size: 13px; color: #64748B;">Speech Fluency & Disfluency Analyzer</span>
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation",
    ["Home", "Analyze", "Sessions", "Trends"],
    index=["Home", "Analyze", "Sessions", "Trends"].index(st.session_state["active_page"]) if st.session_state["active_page"] in ["Home", "Analyze", "Sessions", "Trends"] else 0,
    label_visibility="collapsed"
)

# Update state if changed via sidebar radio
if nav_choice != st.session_state["active_page"]:
    st.session_state["active_page"] = nav_choice

st.sidebar.markdown("<br>", unsafe_allow_html=True)

with st.sidebar.expander("About VoxFlow"):
    st.markdown("""
    **VoxFlow** analyzes recorded speech sessions and identifies three disfluency categories: **Repetition**, **Prolongation**, and **Block**.
    
    Speech analysis is powered by the project's trained HuBERT acoustic model.
    
    *Notice:* VoxFlow is a research prototype and is not a medical diagnostic system.
    """)

# Load inference engine singleton safely
try:
    engine = get_inference_engine()
except Exception as ex:
    engine = None
    st.sidebar.warning(f"Inference engine notice: {ex}")


# =============================================================================
# PAGE 1 — HOME
# =============================================================================
if st.session_state["active_page"] == "Home":
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">VoxFlow</h1>
        <p class="app-subtitle">Speech Fluency & Disfluency Session Analyzer</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Analyze speech sessions, identify disfluency events, and review fluency over time.
    """)

    col_btn, _ = st.columns([1.5, 4])
    with col_btn:
        if st.button("🎙️ Analyze New Session", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Analyze"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Check for latest session
    latest = get_latest_session()

    if latest:
        st.markdown("### Latest Session")
        created_str = format_timestamp(latest.get("created_at") or latest.get("started_at"))
        dur = float(latest.get("duration_sec", 0.0))
        fluency = round(float(latest.get("fluency_ratio", 1.0)) * 100, 1)
        total_events = int(latest.get("repetition_count", 0)) + int(latest.get("prolongation_count", 0)) + int(latest.get("block_count", 0))

        st.markdown(f"""
        <div class="vf-card">
            <div style="font-size: 13px; color: #64748B; margin-bottom: 12px;">{created_str} · Duration: {dur:.1f}s</div>
            <div class="vf-metric-container" style="margin-bottom: 12px;">
                <div class="vf-metric-card primary">
                    <div class="vf-metric-label">Fluency</div>
                    <div class="vf-metric-val fluency">{fluency:.0f}%</div>
                </div>
                <div class="vf-metric-card">
                    <div class="vf-metric-label">Total Events</div>
                    <div class="vf-metric-val">{total_events}</div>
                </div>
                <div class="vf-metric-card">
                    <div class="vf-metric-label">Repetition</div>
                    <div class="vf-metric-val rep">{latest.get('repetition_count', 0)}</div>
                </div>
                <div class="vf-metric-card">
                    <div class="vf-metric-label">Prolongation</div>
                    <div class="vf-metric-val pro">{latest.get('prolongation_count', 0)}</div>
                </div>
                <div class="vf-metric-card">
                    <div class="vf-metric-label">Block</div>
                    <div class="vf-metric-val blk">{latest.get('block_count', 0)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_view, _ = st.columns([1.5, 4])
        with col_view:
            if st.button("View Full Session Details", use_container_width=True):
                st.session_state["active_page"] = "Sessions"
                st.session_state["selected_session_id"] = latest["id"]
                st.rerun()

    else:
        st.markdown("""
        <div class="vf-empty-state">
            <div class="vf-empty-title">No sessions analyzed yet</div>
            <div class="vf-empty-desc">Upload a speech recording to get started with your first analysis.</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — ANALYZE
# =============================================================================
elif st.session_state["active_page"] == "Analyze":
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Analyze a Speech Session</h1>
        <p class="app-subtitle">Upload a speech recording to analyze the session.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a speech recording (WAV format, 16 kHz recommended)",
        type=["wav"],
        help="Upload a continuous speech recording (3 to 60 seconds)."
    )

    if uploaded_file is not None:
        st.markdown(f"**Selected file:** `{uploaded_file.name}`")
        st.audio(uploaded_file, format="audio/wav")

        col_act, _ = st.columns([1.5, 4])
        with col_act:
            analyze_clicked = st.button("Analyze Session", type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner("Analyzing session..."):
                try:
                    audio_bytes = uploaded_file.read()
                    summary, windows, events = process_continuous_session(
                        audio_source=audio_bytes,
                        filename=uploaded_file.name,
                        engine=engine
                    )
                    # Persist session atomically to SQLite
                    save_session(summary, windows, events)

                    # Store full object in session state
                    full_session = dict(summary)
                    full_session["windows"] = windows
                    full_session["events"] = events
                    st.session_state["last_analyzed_session"] = full_session

                    st.success("Analysis complete.")
                except Exception as ex:
                    st.error("We couldn't analyze this recording. Please check the file and try again.")
                    with st.expander("Technical details"):
                        st.exception(ex)

    # Display Analysis Result if available in session_state
    if st.session_state.get("last_analyzed_session"):
        st.markdown("### Session Analysis Result")
        render_session_report(st.session_state["last_analyzed_session"], engine)

    # Secondary Expandable Section: ESP32 Hardware
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("▸ ESP32 Hardware Capture"):
        st.markdown("""
        VoxFlow can also receive speech recordings captured using the **ESP32 + INMP441** hardware through the Python serial bridge.
        
        **To run hardware audio capture and send it for analysis:**
        ```bash
        python tools/esp32_serial_bridge.py --port COM8
        ```
        *Run the bridge from a terminal. Once the capture is uploaded and analyzed, the session will automatically appear in VoxFlow.*
        
        **To save the captured WAV locally without sending it for analysis:**
        ```bash
        python tools/esp32_serial_bridge.py --port COM8 --skip-api
        ```
        """)


# =============================================================================
# PAGE 3 — SESSIONS
# =============================================================================
elif st.session_state["active_page"] == "Sessions":
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Sessions</h1>
        <p class="app-subtitle">Review previously analyzed speech sessions.</p>
    </div>
    """, unsafe_allow_html=True)

    sessions = get_all_sessions(limit=100)

    if not sessions:
        st.markdown("""
        <div class="vf-empty-state">
            <div class="vf-empty-title">No sessions have been analyzed yet</div>
            <div class="vf-empty-desc">Record or upload a speech session to see it listed here.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze a Session", type="primary"):
            st.session_state["active_page"] = "Analyze"
            st.rerun()
    else:
        # Format table rows cleanly
        table_rows = []
        session_options = {}
        
        for s in sessions:
            sid = s["id"]
            created = format_timestamp(s.get("created_at") or s.get("started_at"))
            dur = float(s.get("duration_sec", 0.0))
            fluency = round(float(s.get("fluency_ratio", 1.0)) * 100, 1)
            rep = int(s.get("repetition_count", 0))
            pro = int(s.get("prolongation_count", 0))
            blk = int(s.get("block_count", 0))
            
            label = f"{created} · {dur:.0f}s · {fluency:.0f}% Fluency"
            session_options[label] = sid

            table_rows.append({
                "Date": created,
                "Duration": f"{dur:.1f}s",
                "Fluency": f"{fluency:.0f}%",
                "Repetition": rep,
                "Prolongation": pro,
                "Block": blk
            })

        df_sessions = pd.DataFrame(table_rows)
        st.dataframe(df_sessions, use_container_width=True, hide_index=True)

        st.markdown("### Inspect Session")
        
        # Determine initial selection index
        default_index = 0
        if "selected_session_id" in st.session_state:
            target_id = st.session_state["selected_session_id"]
            for idx, (lbl, sid) in enumerate(session_options.items()):
                if sid == target_id:
                    default_index = idx
                    break

        selected_label = st.selectbox(
            "Select session to inspect:",
            options=list(session_options.keys()),
            index=default_index,
            label_visibility="collapsed"
        )

        if selected_label:
            selected_id = session_options[selected_label]
            session_details = get_session_details(selected_id)
            if session_details:
                render_session_report(session_details, engine)
            else:
                st.warning("Could not load details for the selected session.")


# =============================================================================
# PAGE 4 — TRENDS
# =============================================================================
elif st.session_state["active_page"] == "Trends":
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Fluency Trends</h1>
        <p class="app-subtitle">Track fluency across recorded sessions.</p>
    </div>
    """, unsafe_allow_html=True)

    sessions = get_all_sessions(limit=100)

    if len(sessions) < 2:
        st.markdown("""
        <div class="vf-empty-state">
            <div class="vf-empty-title">Not enough session data</div>
            <div class="vf-empty-desc">Record at least 2 sessions to see your fluency trends over time.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze a Session", type="primary"):
            st.session_state["active_page"] = "Analyze"
            st.rerun()
    else:
        # Sort chronologically for trends
        df_trends = pd.DataFrame(sessions).sort_values("created_at")
        df_trends["Session Index"] = [f"Session {i+1}" for i in range(len(df_trends))]
        df_trends["Fluency %"] = (df_trends["fluency_ratio"] * 100).round(1)
        df_trends["Date"] = df_trends["created_at"].apply(format_timestamp)

        # Overview Stats
        avg_fluency = df_trends["Fluency %"].mean()
        tot_rep = df_trends["repetition_count"].sum()
        tot_pro = df_trends["prolongation_count"].sum()
        tot_blk = df_trends["block_count"].sum()
        tot_events = tot_rep + tot_pro + tot_blk

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="vf-metric-card primary">
                <div class="vf-metric-label">Total Sessions</div>
                <div class="vf-metric-val">{len(df_trends)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="vf-metric-card">
                <div class="vf-metric-label">Average Fluency</div>
                <div class="vf-metric-val fluency">{avg_fluency:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="vf-metric-card">
                <div class="vf-metric-label">Total Detected Events</div>
                <div class="vf-metric-val">{tot_events}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="vf-metric-card">
                <div class="vf-metric-label">Latest Fluency</div>
                <div class="vf-metric-val fluency">{df_trends['Fluency %'].iloc[-1]:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart 1: Fluency over time
        st.markdown("#### Fluency over time")
        fig_fluency = px.line(
            df_trends,
            x="Session Index",
            y="Fluency %",
            markers=True,
            hover_data={"Date": True, "Fluency %": True, "Session Index": False}
        )
        fig_fluency.update_traces(
            line=dict(color="#059669", width=3),
            marker=dict(size=8, color="#059669", line=dict(color="#FFFFFF", width=2))
        )
        fig_fluency.update_layout(
            xaxis_title="",
            yaxis_title="Fluency (%)",
            yaxis_range=[0, 105],
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig_fluency, use_container_width=True)

        # Chart 2: Disfluency Events breakdown
        st.markdown("#### Disfluency Events by Category")
        
        fig_events = go.Figure()
        fig_events.add_trace(go.Bar(
            name="Repetition",
            x=df_trends["Session Index"],
            y=df_trends["repetition_count"],
            marker_color=EVENT_COLORS["Repetition"]
        ))
        fig_events.add_trace(go.Bar(
            name="Prolongation",
            x=df_trends["Session Index"],
            y=df_trends["prolongation_count"],
            marker_color=EVENT_COLORS["Prolongation"]
        ))
        fig_events.add_trace(go.Bar(
            name="Block",
            x=df_trends["Session Index"],
            y=df_trends["block_count"],
            marker_color=EVENT_COLORS["Block"]
        ))
        fig_events.update_layout(
            barmode="group",
            xaxis_title="",
            yaxis_title="Event Count",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_events, use_container_width=True)
