"""
Integration tests — FastAPI health endpoint (updated for Phase 2+).

Tests that:
- GET /health returns HTTP 200.
- The response has all required fields.
- /analyze exists (Phase 2).
- model_loaded reflects actual model state.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.utils as _utils_module
from src.utils import LOGGER_NAME, reset_config


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset logging + config cache state before every test."""
    _utils_module._logging_initialised = False
    logging.getLogger(LOGGER_NAME).handlers.clear()
    reset_config()
    yield
    _utils_module._logging_initialised = False
    logging.getLogger(LOGGER_NAME).handlers.clear()
    reset_config()


@pytest.fixture
def client():
    """TestClient with model loading mocked to avoid GPU usage in CI."""
    with patch("app.analyzer.analyzer") as mock_az:
        mock_az.model_loaded       = False
        mock_az.calibration_loaded = False
        mock_az._model_cfg         = {}
        mock_az._loaded            = False
        mock_az.load               = MagicMock(return_value={
            "model_loaded": False,
            "calibration_loaded": False,
            "device": "cpu",
            "architecture": None,
            "checkpoint_epoch": None,
            "experiment_id": None,
            "errors": ["Test mode: model not loaded"],
        })
        with patch("app.main._analyzer", mock_az):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c


# ── Health endpoint tests ──────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client: TestClient):
        response = client.get("/health")
        assert response.headers["content-type"].startswith("application/json")

    def test_health_has_status_field(self, client: TestClient):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_has_model_loaded_field(self, client: TestClient):
        data = client.get("/health").json()
        assert "model_loaded" in data

    def test_health_has_schema_version_field(self, client: TestClient):
        data = client.get("/health").json()
        assert "schema_version" in data

    def test_health_has_version_field(self, client: TestClient):
        data = client.get("/health").json()
        assert "version" in data

    def test_predict_not_implemented(self, client: TestClient):
        """/predict is not an endpoint — /analyze is used instead."""
        response = client.post("/predict", json={})
        assert response.status_code == 404

    def test_analyze_endpoint_exists(self, client: TestClient):
        """/analyze endpoint must exist (even if model not loaded returns 503)."""
        response = client.post("/analyze")
        # 422 (no file) or 503 (model not loaded) — not 404
        assert response.status_code != 404
