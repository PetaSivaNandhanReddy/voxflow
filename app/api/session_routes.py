#!/usr/bin/env python3
"""
VoxFlow V2 — Session REST API Routes
Implements the TARP-compatible session lifecycle for ESP32 Dev Module + INMP441
and web clients:
  1. POST /api/v1/session/start
  2. POST /api/v1/session/<session_id>/audio
  3. POST /api/v1/session/<session_id>/stop
  4. POST /api/v1/session/<session_id>/analyze
  5. GET  /api/v1/session/<session_id>
  6. GET  /api/v1/session/latest
  7. GET  /api/v1/sessions/<user_id>
  8. GET  /api/v1/model/info
  9. GET  /api/v1/health
"""

import os
import uuid
import pathlib
from datetime import datetime
from flask import Blueprint, request, jsonify

from configs.config import API_CONFIG, SESSION_CONFIG, MODEL_CONFIG
from inference.engine import get_inference_engine
from app.db.database import (
    save_session_record,
    save_session_analysis,
    get_session,
    get_latest_session,
    get_all_sessions
)
from app.services.session_engine import (
    validate_and_load_session_audio,
    process_session_windows,
    AudioValidationError
)
from app.services.event_aggregator import aggregate_window_events
from app.services.session_summary import calculate_session_summary

session_bp = Blueprint('session_bp', __name__, url_prefix='/api/v1')

# In-memory session buffer for active audio streaming
ACTIVE_SESSIONS = {}


@session_bp.route('/health', methods=['GET'])
def health_check():
    """Healthcheck endpoint for monitoring."""
    engine = get_inference_engine()
    return jsonify({
        "status": "healthy",
        "service": API_CONFIG.get("service_name", "VoxFlow Session API"),
        "api_version": API_CONFIG.get("api_version", "v2.0"),
        "model_id": MODEL_CONFIG["model_id"],
        "model_name": MODEL_CONFIG["model_name"],
        "model_loaded": engine.model_loaded,
        "device": str(engine.device),
        "storage": "SQLite + Audio Buffer"
    }), 200


@session_bp.route('/session/start', methods=['POST'])
def start_session():
    """
    Initializes a new recording session.
    JSON payload: {"user_id": "optional_username"}
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id', 'default_user')
    session_id = f"sess_{uuid.uuid4().hex[:10]}"
    started_at = datetime.utcnow().isoformat()

    ACTIVE_SESSIONS[session_id] = {
        "user_id": user_id,
        "started_at": started_at,
        "ended_at": None,
        "status": "STARTED",
        "audio_chunks": bytearray(),
        "audio_path": None,
        "duration": 0.0
    }

    save_session_record(
        session_id=session_id,
        user_id=user_id,
        started_at=started_at,
        ended_at=None,
        duration=0.0,
        audio_path="",
        model_version=MODEL_CONFIG["model_name"],
        status="STARTED"
    )

    return jsonify({
        "status": "success",
        "message": "Session started successfully",
        "session_id": session_id,
        "started_at": started_at
    }), 201


@session_bp.route('/session/<session_id>/audio', methods=['POST'])
def upload_session_audio(session_id):
    """
    Accepts streamed chunked audio from ESP32 Dev Module + INMP441
    or multipart WAV audio from web clients.
    """
    if session_id not in ACTIVE_SESSIONS:
        # Check if session exists in DB
        db_s = get_session(session_id)
        if not db_s:
            return jsonify({"status": "error", "message": f"Session '{session_id}' not found"}), 404
        ACTIVE_SESSIONS[session_id] = {
            "user_id": db_s.get('user_id', 'default_user'),
            "started_at": db_s.get('started_at', datetime.utcnow().isoformat()),
            "ended_at": db_s.get('ended_at'),
            "status": db_s.get('status', 'STARTED'),
            "audio_chunks": bytearray(),
            "audio_path": db_s.get('audio_path'),
            "duration": db_s.get('duration_sec', 0.0)
        }

    session = ACTIVE_SESSIONS[session_id]
    if session['status'] == 'ANALYZED':
        return jsonify({"status": "error", "message": f"Session '{session_id}' is already finalized"}), 400

    # Accept raw bytes or multipart form file
    if 'audio' in request.files:
        audio_bytes = request.files['audio'].read()
    else:
        audio_bytes = request.get_data()

    if not audio_bytes:
        return jsonify({"status": "error", "message": "No audio payload provided"}), 400

    session['audio_chunks'].extend(audio_bytes)

    return jsonify({
        "status": "success",
        "message": f"Received {len(audio_bytes)} bytes",
        "total_buffer_bytes": len(session['audio_chunks'])
    }), 200


@session_bp.route('/session/<session_id>/stop', methods=['POST'])
def stop_session(session_id):
    """
    Stops an active recording session and writes the buffered audio to disk.
    """
    if session_id not in ACTIVE_SESSIONS:
        db_s = get_session(session_id)
        if not db_s:
            return jsonify({"status": "error", "message": f"Session '{session_id}' not found"}), 404
        return jsonify({"status": "success", "message": "Session already stopped", "session_id": session_id}), 200

    session = ACTIVE_SESSIONS[session_id]
    session['ended_at'] = datetime.utcnow().isoformat()
    session['status'] = "STOPPED"

    # Save audio buffer to storage directory
    audio_dir = pathlib.Path(SESSION_CONFIG["audio_upload_dir"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file_path = audio_dir / f"{session_id}.wav"

    with open(audio_file_path, 'wb') as f:
        f.write(session['audio_chunks'])

    session['audio_path'] = str(audio_file_path)

    save_session_record(
        session_id=session_id,
        user_id=session['user_id'],
        started_at=session['started_at'],
        ended_at=session['ended_at'],
        duration=session['duration'],
        audio_path=str(audio_file_path),
        model_version=MODEL_CONFIG["model_name"],
        status="STOPPED"
    )

    return jsonify({
        "status": "success",
        "message": "Session stopped successfully",
        "session_id": session_id,
        "ended_at": session['ended_at'],
        "audio_file": str(audio_file_path),
        "total_bytes": len(session['audio_chunks'])
    }), 200


@session_bp.route('/session/<session_id>/analyze', methods=['POST'])
def analyze_session(session_id):
    """
    Runs the complete V2 Objective 2 pipeline on a stopped session:
      Audio validation -> Sliding 3s windows (1s step) -> HuBERT_D inference ->
      Temporal event aggregation -> Summary calculation -> SQLite persistence.
    """
    db_session = get_session(session_id)
    if not db_session:
        return jsonify({"status": "error", "message": f"Session '{session_id}' not found"}), 404

    audio_path = db_session.get('audio_path')
    if not audio_path or not os.path.exists(audio_path):
        # Fallback to active session buffer if unwritten
        if session_id in ACTIVE_SESSIONS and len(ACTIVE_SESSIONS[session_id]['audio_chunks']) > 0:
            audio_dir = pathlib.Path(SESSION_CONFIG["audio_upload_dir"])
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = str(audio_dir / f"{session_id}.wav")
            with open(audio_path, 'wb') as f:
                f.write(ACTIVE_SESSIONS[session_id]['audio_chunks'])
        else:
            return jsonify({"status": "error", "message": "Session audio file missing or not uploaded"}), 400

    try:
        audio, sr, duration_sec = validate_and_load_session_audio(audio_path)
    except AudioValidationError as e:
        return jsonify({"status": "error", "type": "AudioValidationError", "message": str(e)}), 422
    except Exception as e:
        return jsonify({"status": "error", "type": "AudioProcessingError", "message": str(e)}), 500

    try:
        engine = get_inference_engine()

        # 1. Sliding 3.0s window multi-label processing (1.0s step)
        windows = process_session_windows(session_id, audio, sr, duration_sec, engine=engine)

        # 2. Temporal Multi-label Event Aggregation
        events = aggregate_window_events(
            session_id=session_id,
            windows=windows,
            min_support_windows=1,
            merge_gap_sec=1.5
        )

        # 3. Deterministic session summary calculation
        summary = calculate_session_summary(
            session_id=session_id,
            duration_sec=duration_sec,
            windows=windows,
            events=events,
            model_name=engine.model_name
        )
        summary["audio_filename"] = os.path.basename(audio_path)

        # 4. Save to SQLite database
        save_session_record(
            session_id=session_id,
            user_id=db_session.get('user_id', 'default_user'),
            started_at=db_session.get('started_at', datetime.utcnow().isoformat()),
            ended_at=db_session.get('ended_at') or datetime.utcnow().isoformat(),
            duration=duration_sec,
            audio_path=audio_path,
            model_version=engine.model_name,
            status="ANALYZED"
        )
        save_session_analysis(session_id, windows, events, summary)

        if session_id in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[session_id]['status'] = "ANALYZED"

        return jsonify({
            "status": "success",
            "message": "Session analysis completed successfully",
            "session_id": session_id,
            "duration": duration_sec,
            "summary": summary,
            "event_count": len(events),
            "window_count": len(windows)
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "type": "AnalysisFailure", "message": str(e)}), 500


@session_bp.route('/session/<session_id>', methods=['GET'])
def get_session_details_route(session_id):
    """Retrieves full session analysis, windows, and events."""
    session = get_session(session_id)
    if not session:
        return jsonify({"status": "error", "message": f"Session '{session_id}' not found"}), 404
    return jsonify({"status": "success", "session": session}), 200


@session_bp.route('/sessions/<user_id>', methods=['GET'])
def get_user_sessions(user_id):
    """Returns all session records for a given user."""
    sessions = get_all_sessions(user_id=user_id)
    return jsonify({"status": "success", "sessions": sessions, "count": len(sessions)}), 200


@session_bp.route('/session/latest', methods=['GET'])
def get_latest():
    """Returns the most recent analyzed session."""
    session = get_latest_session()
    if not session:
        return jsonify({"status": "empty", "message": "No analyzed sessions found"}), 200
    return jsonify({"status": "success", "session": session}), 200


@session_bp.route('/model/info', methods=['GET'])
def get_model_info_route():
    """Returns runtime model metadata and locked validation thresholds."""
    engine = get_inference_engine()
    return jsonify({
        "status": "success",
        "model_info": engine.get_model_info()
    }), 200
