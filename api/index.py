"""
api/index.py — Vercel Python serverless function entrypoint for SignalScope.

This is a THIN adapter. It does nothing except:
  1. Ensure the project root is on sys.path.
  2. Force CPU device for Vercel (Linux, no DirectML).
  3. Import the existing FastAPI `app` from app.main.

The entire SignalScope application — ML pipeline, routes, static files,
Supabase persistence, Grad-CAM, robustness probes, provenance, admin
endpoints — comes from the existing `app.main` module untouched.

Do NOT add business logic here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Project root → sys.path ───────────────────────────────────────────────────
# Vercel executes this file from the project root, but we resolve explicitly
# so it works regardless of the working directory.
_HERE = Path(__file__).resolve()          # api/index.py
_ROOT = _HERE.parent.parent               # project root (parent of api/)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Force CPU on Vercel (Linux — no DirectML, no CUDA) ───────────────────────
# Uses setdefault so an explicit SIGNALSCOPE_DEVICE env var in the Vercel
# dashboard still takes effect (though "cpu" is the only sensible value).
os.environ.setdefault("SIGNALSCOPE_DEVICE", "cpu")

# ── SQLite: redirect writes to /tmp on Linux/Vercel ───────────────────────────
# Vercel's filesystem is read-only except for /tmp.
# The SQLite DB is a local fallback only; Supabase is the production store.
# We redirect to /tmp so the app doesn't crash on startup when it tries to
# look up the experiment_id from a pre-existing SQLite file.
if sys.platform != "win32":
    os.environ.setdefault("SIGNALSCOPE_SQLITE_PATH", "/tmp/signalscope.db")

# ── Import the existing FastAPI app ───────────────────────────────────────────
from app.main import app  # noqa: E402  (import after sys.path manipulation)

# Vercel's Python runtime expects the ASGI callable to be named `app`.
# app is already that callable — nothing else needed.
