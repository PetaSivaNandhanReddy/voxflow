#!/usr/bin/env python3
"""
VoxFlow — Main Application Launcher
Provides a clean, single-command entrypoint to run the VoxFlow platform:
- Starts the Flask REST API server (port 5000)
- Starts the Streamlit Clinical Dashboard (port 8501)
- Handles graceful shutdown on SIGINT / Ctrl+C
"""

import os
import sys
import time
import signal
import pathlib
import argparse
import subprocess
import threading

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MODEL_CONFIG, API_CONFIG
from app.db.database import init_db


def check_prerequisites():
    """Verifies that database schema and model checkpoints are available."""
    init_db()
    ckpt_path = pathlib.Path(MODEL_CONFIG["checkpoint_path"])
    if not ckpt_path.exists():
        print("=" * 60)
        print("WARNING: Model checkpoint not found at:")
        print(f"  {ckpt_path}")
        print("To download or link the model checkpoint, please see:")
        print("  models/README.txt or README.md (Model Distribution section)")
        print("=" * 60)
    else:
        print(f"[OK] Model checkpoint verified: {ckpt_path.name} ({ckpt_path.stat().st_size // (1024*1024)} MB)")


def run_api_server(host, port):
    """Runs the Flask REST API server."""
    from server import app
    print(f"[*] Starting VoxFlow REST API on http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False, use_reloader=False)


def run_dashboard():
    """Runs the Streamlit dashboard."""
    dashboard_file = PROJECT_ROOT / "app" / "dashboard" / "dashboard_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_file),
        "--server.headless=false",
        "--theme.base=light"
    ]
    print("[*] Launching Streamlit Clinical Dashboard ...")
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(description="VoxFlow Speech Fluency System Launcher")
    parser.add_argument("--api-only", action="store_true", help="Launch only the Flask REST API server")
    parser.add_argument("--dashboard-only", action="store_true", help="Launch only the Streamlit UI dashboard")
    parser.add_argument("--host", default=API_CONFIG.get("host", "0.0.0.0"), help="API server host")
    parser.add_argument("--port", type=int, default=API_CONFIG.get("port", 5000), help="API server port")
    args = parser.parse_args()

    print("=" * 65)
    print("  VoxFlow — Speech Fluency & Stuttering Event Monitoring Platform")
    print(f"  Model: HuBERT Multi-label Classifier | Sample Rate: 16 kHz Mono")
    print("=" * 65)

    check_prerequisites()

    if args.api_only:
        run_api_server(args.host, args.port)
        return

    if args.dashboard_only:
        proc = run_dashboard()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    # Default: Run API in background thread and Streamlit in main process
    api_thread = threading.Thread(
        target=run_api_server,
        args=(args.host, args.port),
        daemon=True
    )
    api_thread.start()
    time.sleep(1.0)  # Brief pause to let API initialize

    dashboard_proc = run_dashboard()

    def handle_exit(signum, frame):
        print("\n[*] Shutting down VoxFlow services...")
        dashboard_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        dashboard_proc.wait()
    except KeyboardInterrupt:
        handle_exit(None, None)


if __name__ == "__main__":
    main()
