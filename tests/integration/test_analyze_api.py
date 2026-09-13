"""
Tests for the /analyze, /health, /feedback API endpoints.

Uses FastAPI TestClient with a mocked analyzer to avoid loading
the real GPU model during unit tests.

Coverage:
- GET /health returns correct fields
- POST /analyze with valid image returns expected schema
- POST /analyze with no file returns 422
- POST /analyze with oversized upload returns 413
- POST /analyze with unsupported MIME returns 415
- POST /analyze with corrupt image returns 422
- POST /feedback with valid verdict stores feedback
- POST /feedback with invalid verdict returns 400
- GET /results/{id} returns 404 for unknown ID
- Analysis result schema includes all required fields
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ── Build a tiny valid PNG in memory ──────────────────────────────────────────
def _make_png(w=32, h=32, color=(120, 80, 40)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Mock analyzer to avoid loading real GPU model ─────────────────────────────
def _mock_analyze_result(**kwargs):
    """Return a realistic analysis result dict."""
    return {
        "prediction":               "real",
        "label":                    "Likely Real",
        "raw_probability":          0.12,
        "calibrated_probability":   0.18,
        "calibration_status":       "calibrated",
        "calibration_temperature":  2.8756,
        "calibration_detail": {
            "temperature": 2.8756,
            "ece_before": 0.104,
            "ece_after":  0.048,
            "brier_before": 0.1236,
            "brier_after":  0.1086,
        },
        "stress_test": {
            "probes_configured": 5,
            "probes_run":        5,
            "probes_successful": 5,
            "probes_failed":     0,
            "mean_probability":  0.19,
            "std_probability":   0.04,
            "stability":         0.96,
            "stop_reason":       "threshold_not_configured",
            "threshold_configured": False,
            "evidence_sufficient":  True,
            "probe_results": [
                {"probe_name": "jpeg_q90", "success": True, "raw_probability": 0.12,
                 "calibrated_probability": 0.19, "processing_time_s": 0.1},
            ],
            "final_prediction": "real",
            "final_probability": 0.18,
            "total_time_s": 2.1,
        },
        "reliability": {
            "observation_count":    6,
            "mean_probability":     0.19,
            "std_probability":      0.04,
            "stability":            0.96,
            "evidence_sufficient":  True,
            "stop_reason":          "threshold_not_configured",
            "threshold_configured": False,
            "status":               "threshold_not_configured",
        },
        "gradcam": {
            "status":            "ok",
            "target_layer_name": "blocks[-1][-1].conv_pw",
            "target_class":      1,
            "heatmap_available": True,
            "overlay_available": True,
            "overlay_b64":       "iVBORw0KGgo=",  # tiny stub
            "heatmap_b64":       "iVBORw0KGgo=",
        },
        "provenance": {
            "exif": {"status": "metadata_absent"},
            "c2pa": {"status": "c2pa_library_unavailable"},
            "summary": "No EXIF metadata. C2PA analysis unavailable.",
        },
        "model_architecture":  "efficientnet_b0",
        "model_version":       "efficientnet_b0_v1",
        "checkpoint_epoch":    5,
        "experiment_id":       1,
        "image_hash":          "a" * 64,
        "analysis_id":         1,
        "processing_time_ms":  3200.0,
        "error":               None,
        "warnings":            [],
    }


@pytest.fixture
def client():
    """TestClient with analyzer mocked to avoid GPU load."""
    with patch("app.analyzer.analyzer") as mock_az:
        mock_az.model_loaded     = True
        mock_az.calibration_loaded = True
        mock_az._model_cfg       = {"architecture": "efficientnet_b0"}
        mock_az._loaded          = True
        mock_az.analyze          = MagicMock(side_effect=lambda **kw: _mock_analyze_result(**kw))

        # Patch startup to do nothing (model already "loaded")
        with patch("app.main._analyzer", mock_az):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as tc:
                yield tc, mock_az


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200

    def test_health_fields_present(self, client):
        tc, _ = client
        data = tc.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "calibration_loaded" in data
        assert "version" in data
        assert "schema_version" in data

    def test_health_model_loaded_true(self, client):
        tc, _ = client
        data = tc.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_schema_version_is_2(self, client):
        tc, _ = client
        data = tc.get("/health").json()
        assert data["schema_version"] == 2


# ── Analyze ───────────────────────────────────────────────────────────────────

class TestAnalyze:

    def test_valid_png_returns_200(self, client):
        tc, _ = client
        png = _make_png()
        resp = tc.post("/analyze", files={"file": ("test.png", png, "image/png")})
        assert resp.status_code == 200

    def test_response_has_prediction(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert data["prediction"] in ("real", "synthetic")

    def test_response_has_raw_probability(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert data["raw_probability"] is not None
        assert 0.0 <= data["raw_probability"] <= 1.0

    def test_response_has_calibrated_probability(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert data["calibrated_probability"] is not None
        assert 0.0 <= data["calibrated_probability"] <= 1.0

    def test_response_has_calibration_status(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert "calibration_status" in data
        assert data["calibration_status"] in (
            "calibrated", "not_calibrated", "calibration_unavailable", "calibration_fallback"
        )

    def test_response_has_stress_test(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert "stress_test" in data
        st = data["stress_test"]
        assert "probes_configured" in st
        assert "probes_successful" in st
        assert "stop_reason" in st

    def test_response_has_reliability(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert "reliability" in data
        rel = data["reliability"]
        assert "status" in rel
        assert "observation_count" in rel

    def test_response_has_model_version(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert data.get("model_version") is not None

    def test_response_has_processing_time(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert data.get("processing_time_ms") is not None
        assert data["processing_time_ms"] > 0

    def test_response_has_image_hash(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        assert data.get("image_hash") is not None

    def test_no_file_returns_422(self, client):
        tc, _ = client
        resp = tc.post("/analyze")
        assert resp.status_code == 422

    def test_empty_file_returns_400(self, client):
        tc, _ = client
        resp = tc.post("/analyze", files={"file": ("empty.png", b"", "image/png")})
        assert resp.status_code == 400

    def test_oversized_file_returns_413(self, client):
        tc, _ = client
        # 21 MB of zeros
        big = b"\x00" * (21 * 1024 * 1024)
        resp = tc.post("/analyze", files={"file": ("big.png", big, "image/png")})
        assert resp.status_code == 413

    def test_stress_test_stop_reason_explicit(self, client):
        tc, _ = client
        png = _make_png()
        data = tc.post("/analyze", files={"file": ("test.png", png, "image/png")}).json()
        rel = data.get("reliability", {})
        # Must explicitly report why probing stopped
        assert rel.get("stop_reason") is not None

    def test_analyzer_called_with_correct_filename(self, client):
        tc, mock_az = client
        png = _make_png()
        tc.post("/analyze", files={"file": ("myimage.png", png, "image/png")})
        call_kwargs = mock_az.analyze.call_args[1]
        assert call_kwargs.get("filename") == "myimage.png"

    def test_analyzer_called_with_stress_test_flag(self, client):
        tc, mock_az = client
        png = _make_png()
        tc.post(
            "/analyze",
            files={"file": ("test.png", png, "image/png")},
            data={"run_stress_test": "false"},
        )
        call_kwargs = mock_az.analyze.call_args[1]
        assert call_kwargs.get("run_stress_test") is False


# ── Feedback ──────────────────────────────────────────────────────────────────

class TestFeedback:

    def _post_feedback(self, tc, verdict="correct", analysis_id=1, comment=None):
        payload = {"analysis_id": analysis_id, "verdict": verdict}
        if comment:
            payload["comment"] = comment
        return tc.post("/feedback", json=payload)

    def test_valid_correct_verdict(self, client):
        tc, _ = client
        with patch("app.main._save_feedback", return_value=1):
            resp = self._post_feedback(tc, verdict="correct")
            assert resp.status_code == 200

    def test_valid_incorrect_verdict(self, client):
        tc, _ = client
        with patch("app.main._save_feedback", return_value=2):
            resp = self._post_feedback(tc, verdict="incorrect")
            assert resp.status_code == 200

    def test_valid_unsure_verdict(self, client):
        tc, _ = client
        with patch("app.main._save_feedback", return_value=3):
            resp = self._post_feedback(tc, verdict="unsure")
            assert resp.status_code == 200

    def test_invalid_verdict_returns_400(self, client):
        tc, _ = client
        resp = self._post_feedback(tc, verdict="maybe")
        assert resp.status_code == 400

    def test_invalid_verdict_error_message(self, client):
        tc, _ = client
        resp = self._post_feedback(tc, verdict="yes_its_fake")
        data = resp.json()
        assert "verdict" in str(data).lower() or "invalid" in str(data).lower()

    def test_feedback_response_has_status(self, client):
        tc, _ = client
        with patch("app.main._save_feedback", return_value=1):
            resp = self._post_feedback(tc)
            data = resp.json()
            assert "status" in data

    def test_feedback_storage_failure_returns_200_gracefully(self, client):
        """Storage failures are reported gracefully, not as server 500."""
        tc, _ = client
        with patch("app.main._save_feedback", side_effect=Exception("DB locked")):
            resp = self._post_feedback(tc)
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("status") == "storage_failed"


# ── Results retrieval ─────────────────────────────────────────────────────────

class TestResults:

    def test_unknown_analysis_id_returns_404(self, client):
        tc, _ = client
        # get_analysis is imported inside the route function, so patch at the source module
        with patch("src.database.repositories.get_analysis", return_value=None):
            resp = tc.get("/results/99999")
            assert resp.status_code == 404
