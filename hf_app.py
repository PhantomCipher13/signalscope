"""
SignalScope — Hugging Face Spaces entry point (hf_app.py)
=========================================================
This is the THINNEST possible HF Spaces adapter.

Strategy:
  - HF Gradio SDK is free (no PRO required).
  - Gradio's underlying server is FastAPI.
  - We mount our existing FastAPI app routes INTO Gradio's FastAPI instance,
    so ALL existing /analyze, /health, /feedback, /admin endpoints remain intact.
  - The Gradio UI itself is just a single-page HTML redirect to the real frontend
    already served by SignalScope at /.
  - Port 7860 is what HF Spaces expects.

No ML logic lives here. No API contract changes.
"""

from __future__ import annotations
import os, sys
from pathlib import Path

# ── Ensure project root is on path ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Load secrets from environment (HF Spaces sets these as secrets) ───────────
# .env is loaded inside app.main too, but won't exist on HF — env vars take over.
try:
    from dotenv import load_dotenv
    _env = ROOT / ".env"
    if _env.exists():
        load_dotenv(str(_env), override=False)
except ImportError:
    pass

# ── Force CPU device on cloud (no DirectML on Linux) ─────────────────────────
if not os.environ.get("SIGNALSCOPE_DEVICE"):
    os.environ["SIGNALSCOPE_DEVICE"] = "cpu"

# ── Import the existing FastAPI application ───────────────────────────────────
from app.main import app as _fastapi_app   # full SignalScope FastAPI app

# ── Create a minimal Gradio interface ─────────────────────────────────────────
# This is ONLY so HF detects a Gradio app; the real UI is served by FastAPI at /.
import gradio as gr

_description = """
## SignalScope
**Telling Real From Synthetic in the Age of Generative Media**

👉 **[Open the full SignalScope interface →](/)**

Or use the API directly:
- `GET /health` — service status
- `POST /analyze` — upload an image for AI detection
- `POST /feedback` — submit verdict feedback
- `GET /docs` — interactive API documentation

---
*Baseline model: EfficientNet-B0 trained on CIFAKE (Stable Diffusion v1.4).*
*Results are in-distribution estimates — not universal AI detection.*
"""

with gr.Blocks(title="SignalScope — AI Image Detector") as _gradio_demo:
    gr.Markdown(_description)

# ── Mount Gradio into the existing FastAPI app ────────────────────────────────
# This adds Gradio at /gradio-ui and preserves ALL existing FastAPI routes.
app = gr.mount_gradio_app(_fastapi_app, _gradio_demo, path="/gradio-ui")

# ── Launch (HF Spaces runs this file directly) ────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
