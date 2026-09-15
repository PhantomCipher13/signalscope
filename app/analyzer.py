"""
SignalScope — app/analyzer.py
==============================
Core analysis service: singleton that loads the model once and provides
the full analysis pipeline for /analyze requests.

Pipeline:
    image bytes
    → validate
    → hash
    → preprocess
    → detector (baseline probability)
    → calibration (temperature scaling)
    → stress testing (5 probes)
    → reliability aggregation
    → grad-cam (optional)
    → provenance (EXIF / C2PA)
    → DB persist
    → return AnalysisResponse

Design:
    - Model loaded once at startup.
    - All components are optional with graceful fallback.
    - No component failure should crash the entire pipeline.
    - Scientific constraints always enforced (no fabricated values).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Optional

import logging
import torch
from PIL import Image

logger = logging.getLogger("analyzer")

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults — resolved relative to project root
_DEFAULT_CHECKPOINT  = _PROJECT_ROOT / "models" / "signalscope_b0_v3.pt"
_DEFAULT_CALIBRATION = _PROJECT_ROOT / "models" / "calibration_v3.json"


class SignalScopeAnalyzer:
    """
    Singleton analysis engine for SignalScope.

    Load once at app startup. Thread-safe for concurrent reads (inference only).
    """

    def __init__(self) -> None:
        self._model           = None
        self._device          = None
        self._transform       = None
        self._model_cfg       = {}
        self._ckpt_meta       = {}
        self._cal_temperature: Optional[float] = None
        self._cal_status      = "not_calibrated"
        self._cal_detail      = {}
        self._experiment_id: Optional[int] = None
        self._loaded          = False

    # ── Startup ───────────────────────────────────────────────────────────

    def load(
        self,
        checkpoint_path: Optional[Path] = None,
        calibration_path: Optional[Path] = None,
        device_pref: str = "auto",
    ) -> dict:
        """
        Load model and calibration. Returns a status dict.
        Safe to call multiple times — re-loads if called again.
        """
        from src.detector    import load_checkpoint, resolve_device
        from src.preprocessing import build_val_transforms

        ckpt  = checkpoint_path  or _DEFAULT_CHECKPOINT
        cal_f = calibration_path or _DEFAULT_CALIBRATION

        if not ckpt.exists():
            release_url = "https://github.com/PhantomCipher13/signalscope/releases/download/signalscope-sih-2026-final/signalscope_b0_v3.pt"
            try:
                import urllib.request
                logger.info(f"Downloading V3 model weights from release...")
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(release_url, str(ckpt))
                logger.info(f"Downloaded {ckpt.name} successfully.")
            except Exception as dl_err:
                logger.warning(f"Could not download V3 weights: {dl_err}")
                fallback = _PROJECT_ROOT / "models" / "fast_baseline_checkpoint.pt"
                if fallback.exists():
                    ckpt = fallback
                    logger.info(f"Using bundled fallback checkpoint: {fallback.name}")

        status = {
            "model_loaded":       False,
            "calibration_loaded": False,
            "device":             None,
            "architecture":       None,
            "checkpoint_epoch":   None,
            "experiment_id":      None,
            "errors":             [],
        }

        # Model
        try:
            self._device = resolve_device(device_pref)
            model, meta  = load_checkpoint(ckpt, device=self._device)
            model.eval()
            self._model    = model
            self._ckpt_meta = meta
            self._model_cfg = meta.get("model_config", {})
            input_size      = self._model_cfg.get("input_size", 224)
            self._transform = build_val_transforms(image_size=input_size)
            status["model_loaded"]     = True
            status["device"]           = str(self._device)
            status["architecture"]     = self._model_cfg.get("architecture", "unknown")
            status["checkpoint_epoch"] = meta.get("epoch", -1) + 1
        except Exception as e:
            status["errors"].append(f"Model load failed: {e}")

        # Calibration
        try:
            if cal_f.exists():
                data  = json.loads(cal_f.read_text(encoding="utf-8"))
                inner = data.get("calibration_result", data)
                st    = inner.get("status", data.get("calibration_status", ""))
                temp  = data.get("temperature") or inner.get("temperature")
                if st == "calibrated" and temp and float(temp) > 0:
                    self._cal_temperature = float(temp)
                    self._cal_status      = "calibrated"
                    self._cal_detail      = {
                        "temperature":   self._cal_temperature,
                        "ece_before":    inner.get("ece_before"),
                        "ece_after":     inner.get("ece_after"),
                        "brier_before":  inner.get("brier_before"),
                        "brier_after":   inner.get("brier_after"),
                    }
                    status["calibration_loaded"] = True
        except Exception as e:
            status["errors"].append(f"Calibration load failed: {e}")

        # Experiment ID from DB (SQLite, optional — non-fatal if /tmp or absent)
        try:
            import os as _os_inner
            from src.database.connection    import DatabaseManager
            from src.database.repositories  import list_experiments
            from src.database.schema        import init_db
            # SIGNALSCOPE_SQLITE_PATH allows Vercel to redirect to /tmp
            _sqlite_override = _os_inner.environ.get("SIGNALSCOPE_SQLITE_PATH")
            if _sqlite_override:
                db_path = Path(_sqlite_override)
            else:
                db_path = _PROJECT_ROOT / "data" / "signalscope.db"
            if db_path.exists():
                db = DatabaseManager(str(db_path))
                db.connect()
                init_db(db)
                exps = list_experiments(db)
                if exps:
                    self._experiment_id = dict(exps[0])["id"]
                    status["experiment_id"] = self._experiment_id
                db.close()
        except Exception:
            pass  # Non-fatal

        self._loaded = status["model_loaded"]
        return status

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def calibration_loaded(self) -> bool:
        return self._cal_status == "calibrated"

    # ── Core analysis ─────────────────────────────────────────────────────

    def analyze(
        self,
        image_bytes: bytes,
        filename: str = "upload",
        run_stress_test: bool = True,
        run_gradcam: bool = True,
        run_provenance: bool = True,
        save_to_db: bool = True,
    ) -> dict:
        """
        Full analysis pipeline.

        Returns a dict suitable for JSON serialization.
        Never raises — all failures are captured in result fields.
        """
        t_start = time.perf_counter()

        result: dict = {
            "prediction":               None,
            "label":                    None,
            "raw_probability":          None,
            "calibrated_probability":   None,
            "calibration_status":       self._cal_status,
            "calibration_temperature":  self._cal_temperature,
            "calibration_detail":       self._cal_detail or None,
            "stress_test":              None,
            "reliability":              None,
            "gradcam":                  {"status": "not_run"},
            "provenance":               {"status": "not_run"},
            "model_architecture":       self._model_cfg.get("architecture", "unknown"),
            "model_version":            f"{self._model_cfg.get('architecture','?')}_v1",
            "checkpoint_epoch":         self._ckpt_meta.get("epoch", -1) + 1,
            "experiment_id":            self._experiment_id,
            "image_hash":               None,
            "processing_time_ms":       None,
            "error":                    None,
            "warnings":                 [],
        }

        if not self._loaded:
            result["error"] = "Model not loaded. Call load() first."
            return result

        # ── Parse and hash image ──────────────────────────────────────────
        try:
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result["image_hash"] = hashlib.sha256(image_bytes).hexdigest()
        except Exception as e:
            result["error"] = f"Cannot decode image: {e}"
            return result

        # ── Baseline inference ────────────────────────────────────────────
        try:
            tensor = self._transform(image_pil).unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits = self._model(tensor)
                if logits.shape[1] >= 2:
                    probs = torch.softmax(logits, dim=1)
                    raw_prob = probs[0, 1].item()
                else:
                    raw_prob = torch.sigmoid(logits[0, 0]).item()
        except Exception as e:
            result["error"] = f"Inference failed: {e}"
            return result

        result["raw_probability"] = round(raw_prob, 6)

        # ── Calibration ───────────────────────────────────────────────────
        cal_prob = None
        if self._cal_status == "calibrated" and self._cal_temperature:
            try:
                import numpy as np
                _eps      = 1e-7
                _p        = float(np.clip(raw_prob, _eps, 1.0 - _eps))
                logit     = float(np.log(_p / (1.0 - _p)))
                scaled    = logit / self._cal_temperature
                cal_prob  = float(1.0 / (1.0 + np.exp(-scaled)))
                result["calibrated_probability"] = round(cal_prob, 6)
            except Exception as e:
                result["warnings"].append(f"Calibration failed: {e}")

        # ── Prediction label ──────────────────────────────────────────────
        prob_for_pred = cal_prob if cal_prob is not None else raw_prob
        result["prediction"] = "synthetic" if prob_for_pred >= 0.5 else "real"
        if prob_for_pred >= 0.75:
            result["label"] = "Likely AI-Generated"
        elif prob_for_pred >= 0.5:
            result["label"] = "Possibly AI-Generated"
        elif prob_for_pred <= 0.25:
            result["label"] = "Likely Real"
        else:
            result["label"] = "Uncertain"

        # ── Stress testing ────────────────────────────────────────────────
        if run_stress_test:
            try:
                from src.robustness import ProbeRunner, StressTestEngine, StressTestConfig, BUILTIN_PROBES
                runner = ProbeRunner(
                    model=self._model,
                    val_transform=self._transform,
                    device=self._device,
                    calibration_temperature=self._cal_temperature,
                    calibration_status=self._cal_status,
                )
                engine = StressTestEngine(probe_runner=runner, config=StressTestConfig())
                st     = engine.run(image_pil)
                result["stress_test"] = st.to_dict()

                # Build reliability summary from stress test
                n_obs   = st.probes_successful + 1  # +1 for baseline
                all_probs = [prob_for_pred] + [
                    (p.calibrated_probability if p.calibrated_probability is not None else p.raw_probability)
                    for p in st.probe_results if p.success and p.raw_probability is not None
                ]
                result["reliability"] = {
                    "observation_count":    n_obs,
                    "mean_probability":     st.mean_probability,
                    "std_probability":      st.std_probability,
                    "stability":            st.stability,
                    "evidence_sufficient":  st.evidence_sufficient,
                    "stop_reason":          st.stop_reason,
                    "threshold_configured": st.threshold_configured,
                    "status": (
                        "threshold_not_configured" if not st.threshold_configured
                        else "stable" if (st.stability is not None and st.stability > 0.8)
                        else "unstable" if (st.stability is not None and st.stability <= 0.8)
                        else "insufficient_evidence"
                    ),
                }
            except Exception as e:
                result["warnings"].append(f"Stress test failed: {e}")
                result["stress_test"] = {"status": "failed", "error": str(e)}

        if result.get("reliability") is None:
            result["reliability"] = {
                "observation_count":    1,
                "mean_probability":     None,
                "std_probability":      None,
                "stability":            None,
                "evidence_sufficient":  False,
                "stop_reason":          "stress_test_not_run",
                "threshold_configured": False,
                "status":               "insufficient_evidence",
            }

        # ── Grad-CAM ──────────────────────────────────────────────────────
        if run_gradcam:
            try:
                from src.gradcam import GradCAM
                gc = GradCAM(model=self._model, device=self._device, target_class=1)
                gc_result = gc.generate(image_pil, self._transform)
                result["gradcam"] = gc_result.to_dict()
                # Attach overlay bytes (base64) for API
                overlay_bytes = gc_result.overlay_bytes(fmt="PNG")
                if overlay_bytes:
                    import base64
                    result["gradcam"]["overlay_b64"] = base64.b64encode(overlay_bytes).decode()
                heatmap_bytes = gc_result.heatmap_bytes(fmt="PNG")
                if heatmap_bytes:
                    import base64
                    result["gradcam"]["heatmap_b64"] = base64.b64encode(heatmap_bytes).decode()
            except Exception as e:
                result["gradcam"] = {"status": "failed", "error": str(e)}
                result["warnings"].append(f"Grad-CAM failed: {e}")

        # ── Provenance ────────────────────────────────────────────────────
        if run_provenance:
            try:
                # Provenance needs a temp file since it reads EXIF from path
                import tempfile, os
                suffix = Path(filename).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(image_bytes)
                    tmp_path = Path(f.name)
                try:
                    from src.provenance import ProvenanceAnalyser
                    prov = ProvenanceAnalyser()
                    prov_result = prov.analyse(tmp_path)
                    result["provenance"] = prov_result.to_dict()
                finally:
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
            except Exception as e:
                result["provenance"] = {"status": "failed", "error": str(e)}
                result["warnings"].append(f"Provenance analysis failed: {e}")

        # ── DB persistence (Supabase first → SQLite with UUID) ────────────
        analysis_id          = None
        supabase_analysis_id = None
        if save_to_db:
            # Supabase first — so we have the UUID to store in SQLite
            try:
                supabase_analysis_id = self._persist_supabase(result)
            except Exception as e:
                result["warnings"].append(f"Supabase persistence failed: {e}")
            # SQLite — stores supabase_analysis_id for feedback linkage
            try:
                analysis_id = self._persist_sqlite(result, supabase_analysis_id=supabase_analysis_id)
            except Exception as e:
                result["warnings"].append(f"Local DB persistence failed: {e}")

        result["analysis_id"]          = analysis_id
        result["supabase_analysis_id"] = supabase_analysis_id
        result["processing_time_ms"]   = round((time.perf_counter() - t_start) * 1000, 1)
        return result

    def _persist_sqlite(self, result: dict, supabase_analysis_id: Optional[str] = None) -> Optional[int]:
        """Persist analysis result to local SQLite. Returns analysis_id or None."""
        from src.database.connection   import DatabaseManager
        from src.database.schema       import init_db
        from src.database.repositories import insert_analysis_run

        import os as _os_persist
        _sqlite_override = _os_persist.environ.get("SIGNALSCOPE_SQLITE_PATH")
        if _sqlite_override:
            db_path = Path(_sqlite_override)
        else:
            db_path = _PROJECT_ROOT / "data" / "signalscope.db"
        db = DatabaseManager(str(db_path))
        db.connect()
        init_db(db)

        # Migration: ensure supabase_analysis_id column exists
        try:
            conn = db.connect()
            conn.execute("ALTER TABLE analysis_runs ADD COLUMN supabase_analysis_id TEXT")
            conn.commit()
        except Exception:
            pass  # column already exists

        st = result.get("stress_test") or {}
        probe_results_json = json.dumps(st.get("probe_results", []))

        analysis_id = insert_analysis_run(
            db,
            experiment_id=self._experiment_id,
            image_hash=result.get("image_hash"),
            model_version=result.get("model_version"),
            raw_probability=result.get("raw_probability"),
            calibrated_probability=result.get("calibrated_probability"),
            calibration_status=result.get("calibration_status"),
            reliability_status=result.get("reliability", {}).get("status"),
            observation_count=result.get("reliability", {}).get("observation_count"),
            mean_probability=result.get("reliability", {}).get("mean_probability"),
            std_probability=result.get("reliability", {}).get("std_probability"),
            probe_results=probe_results_json,
            processing_time_ms=result.get("processing_time_ms"),
        )

        # Back-fill the Supabase UUID if available
        if analysis_id and supabase_analysis_id:
            try:
                conn2 = db.connect()
                conn2.execute(
                    "UPDATE analysis_runs SET supabase_analysis_id = ? WHERE id = ?",
                    (supabase_analysis_id, analysis_id),
                )
                conn2.commit()
            except Exception:
                pass

        db.close()
        return analysis_id


    def _persist_supabase(self, result: dict) -> Optional[str]:
        """
        Persist analysis to Supabase (when configured).
        Returns Supabase UUID or None if Supabase not configured/failed.
        """
        from src.database.supabase_repo import supabase_repo
        if not supabase_repo.ready:
            return None

        st  = result.get("stress_test") or {}
        rel = result.get("reliability") or {}
        pv  = result.get("provenance")  or {}

        sb_id = supabase_repo.insert_analysis(
            image_hash=result.get("image_hash") or ("0" * 64),
            model_version=result.get("model_version"),
            prediction=result.get("prediction"),
            raw_probability=result.get("raw_probability"),
            calibrated_probability=result.get("calibrated_probability"),
            calibration_status=result.get("calibration_status", "not_calibrated"),
            reliability_status=rel.get("status"),
            stability=st.get("stability"),
            probes_run=st.get("probes_run", 0),
            processing_time_ms=result.get("processing_time_ms"),
        )

        if sb_id:
            # Probe results
            probes = st.get("probe_results", [])
            if probes:
                supabase_repo.insert_probe_results(sb_id, probes)

            # Provenance
            exif_d = pv.get("exif") or {}
            c2pa_d = pv.get("c2pa") or {}
            supabase_repo.insert_provenance(
                sb_id,
                exif_status=exif_d.get("status", "unknown"),
                c2pa_status=c2pa_d.get("status", "unknown"),
                summary=pv.get("summary"),
            )

        return sb_id


# ── Module-level singleton ────────────────────────────────────────────────────
analyzer = SignalScopeAnalyzer()

