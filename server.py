#!/usr/bin/env python3
"""
VoxFlow V2 — Production Session Server
Flask application hosting the REST API for ESP32 Dev Module + INMP441 audio streaming,
continuous session analysis, and SQLite database retrieval.
"""

import os
import sys
import pathlib

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify
from configs.config import API_CONFIG, MODEL_CONFIG
from app.db.database import init_db
from app.api.session_routes import session_bp

app = Flask(__name__)
app.register_blueprint(session_bp)


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "VoxFlow V2 Speech Fluency API",
        "model": MODEL_CONFIG["model_name"],
        "endpoints": {
            "health": "/api/v1/health",
            "model_info": "/api/v1/model/info",
            "session_start": "POST /api/v1/session/start",
            "session_audio": "POST /api/v1/session/<session_id>/audio",
            "session_stop": "POST /api/v1/session/<session_id>/stop",
            "session_analyze": "POST /api/v1/session/<session_id>/analyze",
            "session_get": "GET /api/v1/session/<session_id>",
            "session_latest": "GET /api/v1/session/latest"
        }
    }), 200


# Auto-initialize SQLite database schema on startup
init_db()

if __name__ == '__main__':
    host = API_CONFIG.get("host", "0.0.0.0")
    port = API_CONFIG.get("port", 5000)
    print("=" * 60)
    print(f"VoxFlow V2 Production Server starting on {host}:{port}")
    print(f"Active Model: {MODEL_CONFIG['model_name']} ({MODEL_CONFIG['model_id']})")
    print(f"Thresholds:   {MODEL_CONFIG['locked_thresholds']}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False)
