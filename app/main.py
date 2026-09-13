"""
SignalScope — app/main.py
=========================
FastAPI application — full production endpoint set.

Endpoints:
    GET  /              → serves the web UI (index.html)
    GET  /health        → model/calibration status
    POST /analyze       → full analysis pipeline (detect + calibrate + probe + explain)
    POST /feedback      → record tester feedback for an analysis
    GET  /results/{id}  → retrieve a previous analysis result

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Scientific constraints enforced:
    - Never calls CIFAKE in-distribution results "unseen-generator".
    - Never fabricates calibration or reliability values.
    - Missing metadata is never called evidence of AI generation.
    - Grad-CAM is labeled as model attention, not proof.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ── Project root setup ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Load .env BEFORE any env-var reads (Supabase, device, etc.) ──────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_file = _PROJECT_ROOT / ".env"
    if _env_file.exists():
        _load_dotenv(str(_env_file), override=False)  # override=False: real env vars win
except ImportError:
    pass  # dotenv optional — env vars set by shell/system are still read

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel

from src.utils import get_logger, load_config, setup_logging

# ── Logging setup ─────────────────────────────────────────────────────────────
_cfg     = load_config(config_dir=_PROJECT_ROOT / "configs")
_log_cfg = _cfg.get("logging", {})
setup_logging(
    level=_log_cfg.get("level", "INFO"),
    log_dir=_PROJECT_ROOT / _log_cfg.get("log_dir", "logs"),
    filename=_log_cfg.get("filename", "signalscope.log"),
)
logger = get_logger("api")

_app_cfg    = _cfg.get("app", {})
APP_NAME    = _app_cfg.get("name", "SignalScope")
APP_VERSION = _app_cfg.get("version", "1.0.0")

# ── Load model at startup ─────────────────────────────────────────────────────
from app.analyzer import analyzer as _analyzer
import os as _os

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("SignalScope API startup — loading model...")
    device_pref = _os.environ.get("SIGNALSCOPE_DEVICE", "auto")
    status = _analyzer.load(device_pref=device_pref)
    if status["model_loaded"]:
        logger.info(
            "Model loaded: arch=%s epoch=%s device=%s calibration=%s",
            status.get("architecture"),
            status.get("checkpoint_epoch"),
            status.get("device"),
            "yes" if status.get("calibration_loaded") else "no",
        )
    else:
        logger.error("Model FAILED to load: %s", status.get("errors"))

    # Initialise Supabase if configured
    from src.database.supabase_repo import supabase_repo
    sb_ok = supabase_repo.connect()
    if sb_ok:
        logger.info("Supabase persistence: ACTIVE")
    else:
        logger.info("Supabase not configured — local SQLite only. (%s)", supabase_repo.last_error or "no URL set")

    yield
    logger.info("SignalScope API shutting down.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "SignalScope — Telling Real From Synthetic in the Age of Generative Media. "
        "CIFAKE baseline EfficientNet-B0 in-distribution detector."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static / frontend ─────────────────────────────────────────────────────────
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
if _FRONTEND_DIR.exists() and (_FRONTEND_DIR / "index.html").exists():
    # Serve static assets (CSS, JS, images) from /static
    _static_dir = _FRONTEND_DIR / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": APP_NAME, "version": APP_VERSION, "docs": "/docs"}


# ── Pydantic models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibration_loaded: bool
    architecture: Optional[str]
    version: str
    schema_version: int


class FeedbackRequest(BaseModel):
    analysis_id: int
    verdict: str          # "correct" | "incorrect" | "unsure"
    comment: Optional[str] = None
    tester_id: Optional[str] = None  # anonymized tester identifier


class FeedbackResponse(BaseModel):
    feedback_id: Optional[int]
    status: str
    message: str


# ── Allowed image MIME types ──────────────────────────────────────────────────
_ALLOWED_MIME = {
    "image/jpeg", "image/png", "image/webp",
    "image/bmp", "image/tiff",
}
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB hard limit


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """
    Service health check.

    Returns model load status, calibration status, and schema version.
    """
    return HealthResponse(
        status="ok" if _analyzer.model_loaded else "degraded",
        model_loaded=_analyzer.model_loaded,
        calibration_loaded=_analyzer.calibration_loaded,
        architecture=_analyzer._model_cfg.get("architecture") if _analyzer.model_loaded else None,
        version=f"{APP_NAME} {APP_VERSION}",
        schema_version=2,
    )


@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="Image file to analyse (JPEG/PNG/WebP/BMP/TIFF)."),
    run_stress_test: bool = Form(default=True),
    run_gradcam:     bool = Form(default=True),
    run_provenance:  bool = Form(default=True),
) -> JSONResponse:
    """
    Full image analysis pipeline.

    Runs:
    1. Validate and decode image
    2. Compute SHA-256 hash (no binary stored)
    3. EfficientNet-B0 baseline detection
    4. Temperature-scaled calibration (T=2.8756)
    5. 5-probe adaptive stress test (JPEG Q90/70/50, resize 75%/50%)
    6. Reliability aggregation
    7. Grad-CAM attention map (optional)
    8. EXIF / C2PA provenance (optional)
    9. Persist to database
    10. Return structured JSON

    **Scientific disclaimer:**
    - This is an in-distribution CIFAKE-trained model.
    - Results apply to Stable Diffusion v1.4–style synthetic images.
    - Generalization to other generators is not validated.
    - Grad-CAM shows model attention, not proof of manipulation.
    - Metadata absence is NOT evidence of AI generation.
    """
    if not _analyzer.model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "message": "Model is not available. Server may be starting up.",
            },
        )

    # ── Read and validate upload ──────────────────────────────────────────
    if file.content_type and file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_media_type",
                "message": (
                    f"Content type {file.content_type!r} not supported. "
                    f"Supported: {sorted(_ALLOWED_MIME)}"
                ),
            },
        )

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "read_failed", "message": str(e)})

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail={"error": "empty_file", "message": "Empty file."})

    if len(image_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
            },
        )

    # ── Run analysis pipeline ─────────────────────────────────────────────
    logger.info(
        "Analyzing upload: filename=%s size=%d mime=%s",
        file.filename, len(image_bytes), file.content_type,
    )

    result = _analyzer.analyze(
        image_bytes=image_bytes,
        filename=file.filename or "upload",
        run_stress_test=run_stress_test,
        run_gradcam=run_gradcam,
        run_provenance=run_provenance,
        save_to_db=True,
    )

    if result.get("error"):
        raise HTTPException(
            status_code=422,
            detail={"error": "analysis_failed", "message": result["error"]},
        )

    # Strip Grad-CAM base64 from main response if very large (can be fetched separately)
    # but keep it for now — frontend needs it for inline display
    logger.info(
        "Analysis complete: prediction=%s p_raw=%.4f p_cal=%s t=%.0fms id=%s",
        result.get("prediction"),
        result.get("raw_probability") or 0,
        f"{result.get('calibrated_probability'):.4f}" if result.get("calibrated_probability") else "N/A",
        result.get("processing_time_ms") or 0,
        result.get("analysis_id"),
    )

    return JSONResponse(content=result)


@app.get("/results/{analysis_id}", tags=["Analysis"])
async def get_result(analysis_id: int) -> JSONResponse:
    """Retrieve a previous analysis result by ID."""
    try:
        from src.database.connection    import DatabaseManager
        from src.database.repositories  import get_analysis
        from src.database.schema        import init_db

        db = DatabaseManager(str(_PROJECT_ROOT / "data" / "signalscope.db"))
        db.connect()
        init_db(db)
        row = get_analysis(db, analysis_id)
        db.close()

        if row is None:
            raise HTTPException(status_code=404, detail={"error": "not_found"})
        return JSONResponse(content=row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post("/feedback", response_model=FeedbackResponse, tags=["Tester Workflow"])
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """
    Submit tester feedback for a completed analysis.

    Feedback is linked to analysis_id and stored for later review.
    Feedback is NOT used for automatic retraining.
    Verdict must be one of: correct | incorrect | unsure.

    **Note:** Tester feedback is not scientific benchmark ground truth.
    It is collected for UX improvement and error analysis only.
    """
    allowed_verdicts = {"correct", "incorrect", "unsure"}
    if req.verdict not in allowed_verdicts:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_verdict", "message": f"Verdict must be one of {sorted(allowed_verdicts)}"}
        )

    try:
        feedback_id = _save_feedback(
            analysis_id=req.analysis_id,
            verdict=req.verdict,
            comment=req.comment,
            tester_id=req.tester_id,
        )
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="stored",
            message="Feedback recorded. Thank you.",
        )
    except Exception as e:
        logger.warning("Feedback storage failed: %s", e)
        return FeedbackResponse(
            feedback_id=None,
            status="storage_failed",
            message=f"Feedback could not be stored: {e}",
        )


def _save_feedback(
    analysis_id: int,
    verdict: str,
    comment: Optional[str],
    tester_id: Optional[str],
) -> Optional[int]:
    """Persist tester feedback to SQLite and Supabase (when configured)."""
    from src.database.connection   import DatabaseManager
    from src.database.schema       import init_db

    db_path = _PROJECT_ROOT / "data" / "signalscope.db"
    db = DatabaseManager(str(db_path))
    db.connect()
    init_db(db)

    # Ensure feedback table exists (with supabase_analysis_id column)
    conn = db.connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tester_feedback (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id          INTEGER REFERENCES analysis_runs(id),
            supabase_analysis_id TEXT,
            verdict              TEXT NOT NULL,
            comment              TEXT,
            tester_id            TEXT,
            created_at           TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # Migration: add supabase_analysis_id if missing in existing tables
    try:
        conn.execute("ALTER TABLE tester_feedback ADD COLUMN supabase_analysis_id TEXT")
    except Exception:
        pass  # column already exists
    conn.commit()

    # Look up the Supabase UUID for this analysis from analysis_runs
    sb_analysis_id: Optional[str] = None
    try:
        row = db.fetchone(
            "SELECT supabase_analysis_id FROM analysis_runs WHERE id = ?", (analysis_id,)
        )
        if row:
            sb_analysis_id = row["supabase_analysis_id"] if "supabase_analysis_id" in row.keys() else None
    except Exception:
        pass  # column may not exist yet; fallback to None

    db.execute(
        "INSERT INTO tester_feedback (analysis_id, supabase_analysis_id, verdict, comment, tester_id) VALUES (?,?,?,?,?)",
        (analysis_id, sb_analysis_id, verdict, comment, tester_id),
    )
    db.commit()
    fid = db.last_insert_id()
    db.close()
    logger.info("Feedback stored locally: id=%s analysis_id=%s verdict=%s sb_id=%s", fid, analysis_id, verdict, sb_analysis_id)

    # Also persist to Supabase if configured
    try:
        from src.database.supabase_repo import supabase_repo
        if supabase_repo.ready and sb_analysis_id:
            # Use the real Supabase UUID
            sb_fid = supabase_repo.insert_feedback(
                analysis_id=sb_analysis_id,
                verdict=verdict,
                comment=comment,
                tester_id=tester_id,
            )
            if sb_fid:
                logger.info("Feedback stored in Supabase: %s", sb_fid)
        elif supabase_repo.ready and not sb_analysis_id:
            logger.warning("Supabase feedback skipped — no Supabase UUID for analysis_id=%s", analysis_id)
    except Exception as e:
        logger.warning("Supabase feedback storage failed: %s", e)

    return fid


# ── Admin token authentication ────────────────────────────────────────────────
def _require_admin_token(request: Request) -> None:
    """
    Verify ADMIN_TOKEN header/query-param before allowing admin endpoint access.
    Token is read from ADMIN_TOKEN environment variable.
    If ADMIN_TOKEN is not set, admin endpoints are disabled for safety.
    """
    expected = _os.environ.get("ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={"error": "admin_disabled", "message": "Admin endpoints are not configured. Set ADMIN_TOKEN env var."},
        )
    # Accept token from: header X-Admin-Token or query param ?token=
    provided = (
        request.headers.get("X-Admin-Token", "")
        or request.query_params.get("token", "")
    )
    if not provided or provided != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Invalid or missing admin token."},
        )


@app.get("/admin/feedback", tags=["Admin"])
async def list_feedback(request: Request, limit: int = 50, format: str = "json") -> JSONResponse:
    """
    Export tester feedback for review.
    Optional: ?format=csv returns CSV for spreadsheet analysis.
    Feedback is NOT used for automatic retraining.
    Requires X-Admin-Token header or ?token= query param.
    """
    _require_admin_token(request)

    from src.database.connection   import DatabaseManager
    from src.database.schema       import init_db

    db = DatabaseManager(str(_PROJECT_ROOT / "data" / "signalscope.db"))
    db.connect()
    init_db(db)

    try:
        rows = db.fetchall(
            """
            SELECT tf.id, tf.analysis_id, tf.supabase_analysis_id, tf.verdict,
                   tf.comment, tf.tester_id, tf.created_at,
                   ar.model_version, ar.calibrated_probability
            FROM tester_feedback tf
            LEFT JOIN analysis_runs ar ON tf.analysis_id = ar.id
            ORDER BY tf.created_at DESC LIMIT ?
            """,
            (limit,)
        )
    except Exception:
        rows = db.fetchall(
            "SELECT * FROM tester_feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    db.close()

    data = [dict(r) for r in rows]

    if format.lower() == "csv":
        import csv, io
        out = io.StringIO()
        if data:
            writer = csv.DictWriter(out, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=signalscope_feedback.csv"},
        )

    return JSONResponse(content=data)


@app.get("/admin/analyses", tags=["Admin"])
async def list_analyses(request: Request, limit: int = 50) -> JSONResponse:
    """List recent analyses for admin review. Requires X-Admin-Token header."""
    _require_admin_token(request)

    from src.database.connection import DatabaseManager
    from src.database.schema     import init_db

    db = DatabaseManager(str(_PROJECT_ROOT / "data" / "signalscope.db"))
    db.connect()
    init_db(db)
    try:
        rows = db.fetchall(
            "SELECT * FROM analysis_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        db.close()
        return JSONResponse(content=[dict(r) for r in rows])
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)



# ── Development entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    host = _os.environ.get("SIGNALSCOPE_HOST", _app_cfg.get("host", "0.0.0.0"))
    port = int(_os.environ.get("SIGNALSCOPE_PORT", _app_cfg.get("port", 8000)))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)

