#!/usr/bin/env python3
"""
VoxFlow V2 — SQLite Database Layer
Stores users, speech sessions, sliding-window predictions, aggregated events,
and fluency trends. Portable, zero-ORM, auto-initializing.
Supports both single-call processing and the multi-step REST API session lifecycle
(start -> audio -> stop -> analyze).
"""

import os
import json
import sqlite3
import pathlib
from datetime import datetime

from configs.config import DB_CONFIG

DEFAULT_DB_PATH = pathlib.Path(DB_CONFIG["db_path"])


def get_db_path():
    env_path = os.getenv("VOXFLOW_DB_PATH")
    if env_path:
        return pathlib.Path(env_path)
    return DEFAULT_DB_PATH


def get_connection(db_path=None):
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path=None):
    """Initializes tables if they do not already exist."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            started_at TEXT,
            ended_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration_sec REAL DEFAULT 0.0,
            audio_path TEXT DEFAULT '',
            audio_filename TEXT DEFAULT '',
            model_name TEXT NOT NULL,
            status TEXT DEFAULT 'ANALYZED',
            total_windows INTEGER DEFAULT 0,
            fluent_windows INTEGER DEFAULT 0,
            fluency_ratio REAL DEFAULT 1.0,
            repetition_count INTEGER DEFAULT 0,
            prolongation_count INTEGER DEFAULT 0,
            block_count INTEGER DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS session_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            prob_repetition REAL NOT NULL,
            prob_prolongation REAL NOT NULL,
            prob_block REAL NOT NULL,
            is_fluent INTEGER NOT NULL,
            active_labels TEXT,
            confidence REAL DEFAULT 0.0,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS session_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            confidence REAL NOT NULL,
            supporting_windows INTEGER NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """)
    conn.close()


def save_session_record(session_id, user_id="default_user", started_at=None, ended_at=None,
                        duration=0.0, audio_path="", model_version="HuBERT Stage B Extended v2",
                        status="STARTED", db_path=None):
    """
    Saves or updates a session lifecycle state record in SQLite
    (supports states: STARTED, STOPPED, ANALYZED).
    """
    init_db(db_path)
    conn = get_connection(db_path)
    started_at = started_at or datetime.utcnow().isoformat()
    audio_filename = os.path.basename(audio_path) if audio_path else ""

    with conn:
        conn.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))
        conn.execute("""
        INSERT INTO sessions (
            id, user_id, started_at, ended_at, duration_sec, audio_path,
            audio_filename, model_name, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            ended_at = excluded.ended_at,
            duration_sec = excluded.duration_sec,
            audio_path = CASE WHEN excluded.audio_path != '' THEN excluded.audio_path ELSE sessions.audio_path END,
            audio_filename = CASE WHEN excluded.audio_filename != '' THEN excluded.audio_filename ELSE sessions.audio_filename END,
            status = excluded.status;
        """, (
            session_id, user_id, started_at, ended_at, float(duration),
            str(audio_path), audio_filename, model_version, status
        ))
    conn.close()


def save_session_analysis(session_id, windows, events, summary, db_path=None):
    """
    Saves full sliding-window predictions, aggregated disfluency events,
    and final summary metrics for a finalized session.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    with conn:
        conn.execute("""
        UPDATE sessions SET
            duration_sec = ?,
            total_windows = ?,
            fluent_windows = ?,
            fluency_ratio = ?,
            repetition_count = ?,
            prolongation_count = ?,
            block_count = ?,
            status = 'ANALYZED'
        WHERE id = ?
        """, (
            summary["duration_sec"],
            summary["total_windows"],
            summary["fluent_windows"],
            summary["fluency_ratio"],
            summary.get("repetition_count", 0),
            summary.get("prolongation_count", 0),
            summary.get("block_count", 0),
            session_id
        ))

        # Clear existing window/event entries if re-analyzing
        conn.execute("DELETE FROM session_windows WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_events WHERE session_id = ?", (session_id,))

        for idx, w in enumerate(windows):
            prob_rep = w.get("prob_repetition", w.get("probabilities", {}).get("Repetition", 0.0))
            prob_pro = w.get("prob_prolongation", w.get("probabilities", {}).get("Prolongation", 0.0))
            prob_blk = w.get("prob_block", w.get("probabilities", {}).get("Block", 0.0))
            active_labels = w.get("active_labels", [])
            labels_str = ",".join(active_labels) if isinstance(active_labels, list) else str(active_labels)

            conn.execute("""
            INSERT INTO session_windows (
                session_id, window_index, start_time, end_time,
                prob_repetition, prob_prolongation, prob_block, is_fluent, active_labels, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, idx, w["start_time"], w["end_time"],
                float(prob_rep), float(prob_pro), float(prob_blk),
                1 if w["is_fluent"] else 0,
                labels_str,
                float(w.get("confidence", 0.0))
            ))

        for ev in events:
            conn.execute("""
            INSERT INTO session_events (
                session_id, event_type, start_time, end_time, confidence, supporting_windows
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                ev["event_type"],
                float(ev["start_time"]),
                float(ev["end_time"]),
                float(ev["confidence"]),
                int(ev["supporting_windows"])
            ))
    conn.close()


def save_session(session_data, windows, events, db_path=None):
    """Atomic convenience method for single-call session recording."""
    session_id = session_data["session_id"]
    user_id = session_data.get("user_id", "default_user")
    started_at = session_data.get("started_at", datetime.utcnow().isoformat())
    duration = session_data.get("duration_sec", 0.0)
    audio_path = session_data.get("audio_path", "")
    model_name = session_data.get("model_name", "HuBERT Stage B Extended v2")

    save_session_record(
        session_id=session_id,
        user_id=user_id,
        started_at=started_at,
        ended_at=datetime.utcnow().isoformat(),
        duration=duration,
        audio_path=audio_path,
        model_version=model_name,
        status="ANALYZED",
        db_path=db_path
    )
    save_session_analysis(session_id, windows, events, session_data, db_path=db_path)


def get_session(session_id, db_path=None):
    """Fetches session metadata, window details, and event timeline."""
    return get_session_details(session_id, db_path=db_path)


def get_session_details(session_id, db_path=None):
    """Retrieves full details for a session including windows and events."""
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = cur.fetchone()
    if not session:
        conn.close()
        return None

    session_dict = dict(session)

    cur.execute("SELECT * FROM session_windows WHERE session_id = ? ORDER BY window_index", (session_id,))
    windows = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT * FROM session_events WHERE session_id = ? ORDER BY start_time", (session_id,))
    events = [dict(r) for r in cur.fetchall()]

    conn.close()
    session_dict["windows"] = windows
    session_dict["events"] = events
    return session_dict


def get_latest_session(db_path=None):
    """Retrieves the most recently created or analyzed session."""
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM sessions WHERE status = 'ANALYZED' ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return get_session_details(row["id"], db_path=db_path)


def get_all_sessions(user_id=None, limit=50, db_path=None):
    """Fetches session history ordered by creation date."""
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()

    if user_id:
        cur.execute("""
        SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))
    else:
        cur.execute("""
        SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?
        """, (limit,))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
