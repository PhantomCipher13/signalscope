"""
SignalScope — src/database/supabase_repo.py
============================================
Supabase repository for live product data.

This module provides the same interface as the SQLite repositories
but writes to Supabase (PostgreSQL) instead.

Configuration (environment variables — never commit):
    SUPABASE_URL    : your Supabase project URL
    SUPABASE_KEY    : your Supabase anon/service key

Usage:
    Controlled by SIGNALSCOPE_DB_BACKEND env var:
        "sqlite"    → local SQLite (default, for dev/experiments)
        "supabase"  → Supabase (for live tester deployment)

Design:
    - No raw Supabase calls in the analysis pipeline.
    - The analysis service uses get_repo() which returns the right backend.
    - All failures are caught and logged; SQLite fallback is not automatic
      (fail explicitly so operators know Supabase is broken).

Scientific constraints enforced:
    - image binary data is NEVER stored.
    - evaluation_type is preserved from the local analysis.
    - feedback is NOT used for automatic retraining.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("signalscope.supabase")

# ── Schema table names (must match Supabase SQL migration) ────────────────────
T_ANALYSES  = "analyses"
T_PROBES    = "probe_results"
T_FEEDBACK  = "feedback"
T_PROVENANCE = "provenance"


class SupabaseRepo:
    """
    Repository that persists analysis data to Supabase.

    Instantiated once and reused for the life of the process.
    Thread-safe for reads; inserts use Supabase's HTTP API.
    """

    def __init__(self) -> None:
        self._client = None
        self._ready  = False
        self._error: Optional[str] = None

    def connect(self) -> bool:
        """
        Initialise Supabase client from environment variables.
        Returns True if connected, False if configuration is missing.
        """
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_KEY", "").strip()

        if not url or not key:
            self._error = (
                "SUPABASE_URL and SUPABASE_KEY environment variables are required. "
                "Set them before starting the server."
            )
            logger.warning("Supabase not configured: %s", self._error)
            return False

        try:
            from supabase import create_client, Client  # type: ignore
            self._client: Client = create_client(url, key)
            self._ready = True
            logger.info("Supabase client connected to %s", url[:30] + "...")
            return True
        except Exception as e:
            self._error = str(e)
            logger.error("Supabase connection failed: %s", e)
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    # ── Insert analysis ───────────────────────────────────────────────────────

    def insert_analysis(
        self,
        *,
        image_hash: str,
        model_version: Optional[str] = None,
        prediction: Optional[str] = None,
        raw_probability: Optional[float] = None,
        calibrated_probability: Optional[float] = None,
        calibration_status: str = "not_calibrated",
        reliability_status: Optional[str] = None,
        stability: Optional[float] = None,
        probes_run: int = 0,
        processing_time_ms: Optional[float] = None,
    ) -> Optional[str]:
        """
        Insert one analysis row. Returns Supabase row id (UUID) or None.
        Never stores image binary data.
        """
        if not self._ready:
            logger.warning("Supabase not ready — analysis not persisted remotely.")
            return None

        try:
            row = {
                "image_hash":            image_hash,
                "model_version":         model_version,
                "prediction":            prediction,
                "raw_probability":       raw_probability,
                "calibrated_probability": calibrated_probability,
                "calibration_status":    calibration_status,
                "reliability_status":    reliability_status,
                "stability":             stability,
                "probes_run":            probes_run,
                "processing_time_ms":    processing_time_ms,
            }
            resp = self._client.table(T_ANALYSES).insert(row).execute()
            inserted = resp.data
            if inserted:
                row_id = inserted[0].get("id")
                logger.info("Supabase analysis inserted id=%s", row_id)
                return str(row_id)
            return None
        except Exception as e:
            logger.error("Supabase insert_analysis failed: %s", e)
            return None

    # ── Insert probes ─────────────────────────────────────────────────────────

    def insert_probe_results(
        self,
        analysis_id: str,
        probe_results: list[dict],
    ) -> bool:
        """Insert per-probe rows. analysis_id is the Supabase UUID."""
        if not self._ready or not analysis_id:
            return False
        try:
            rows = []
            for p in probe_results:
                rows.append({
                    "analysis_id":         analysis_id,
                    "transformation":      p.get("probe_name"),
                    "parameters":          json.dumps(p.get("parameters", {})),
                    "probability":         p.get("raw_probability"),
                    "calibrated_probability": p.get("calibrated_probability"),
                    "status":              "ok" if p.get("success") else "failed",
                    "error_message":       p.get("error"),
                })
            if rows:
                self._client.table(T_PROBES).insert(rows).execute()
                logger.debug("Supabase inserted %d probe rows", len(rows))
            return True
        except Exception as e:
            logger.error("Supabase insert_probe_results failed: %s", e)
            return False

    # ── Insert provenance ─────────────────────────────────────────────────────

    def insert_provenance(
        self,
        analysis_id: str,
        *,
        exif_status: str = "unknown",
        c2pa_status: str = "unknown",
        summary: Optional[str] = None,
    ) -> bool:
        if not self._ready or not analysis_id:
            return False
        try:
            row = {
                "analysis_id": analysis_id,
                "exif_status": exif_status,
                "c2pa_status": c2pa_status,
                "summary":     summary,
            }
            self._client.table(T_PROVENANCE).insert(row).execute()
            return True
        except Exception as e:
            logger.error("Supabase insert_provenance failed: %s", e)
            return False

    # ── Insert feedback ───────────────────────────────────────────────────────

    def insert_feedback(
        self,
        analysis_id: str,
        verdict: str,
        comment: Optional[str] = None,
        tester_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store tester feedback.
        verdict must be: correct | incorrect | unsure
        Feedback is NOT used for automatic retraining.
        """
        if not self._ready:
            logger.warning("Supabase not ready — feedback not persisted remotely.")
            return None
        try:
            row = {
                "analysis_id": analysis_id,
                "verdict":     verdict,
                "comment":     comment,
                "tester_id":   tester_id,
            }
            resp = self._client.table(T_FEEDBACK).insert(row).execute()
            inserted = resp.data
            if inserted:
                fid = inserted[0].get("id")
                logger.info("Supabase feedback inserted id=%s verdict=%s", fid, verdict)
                return str(fid)
            return None
        except Exception as e:
            logger.error("Supabase insert_feedback failed: %s", e)
            return None

    # ── Read feedback (admin) ─────────────────────────────────────────────────

    def list_feedback(self, limit: int = 100) -> list[dict]:
        """List recent feedback for admin review."""
        if not self._ready:
            return []
        try:
            resp = (
                self._client.table(T_FEEDBACK)
                .select("*, analyses(prediction, calibrated_probability, model_version)")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("Supabase list_feedback failed: %s", e)
            return []

    # ── Read analyses (admin) ─────────────────────────────────────────────────

    def list_analyses(self, limit: int = 50) -> list[dict]:
        """List recent analyses for admin review."""
        if not self._ready:
            return []
        try:
            resp = (
                self._client.table(T_ANALYSES)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as e:
            logger.error("Supabase list_analyses failed: %s", e)
            return []


# ── Module-level singleton ────────────────────────────────────────────────────
supabase_repo = SupabaseRepo()
