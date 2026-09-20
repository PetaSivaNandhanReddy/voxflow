#!/usr/bin/env python3
"""
VoxFlow V2 — Streamlit Dashboard Launcher
Entrypoint: streamlit run v2/dashboard.py
"""

import os
import sys
import runpy
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent
target_dashboard = ROOT_DIR / "app" / "dashboard" / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(target_dashboard), run_name="__main__")
