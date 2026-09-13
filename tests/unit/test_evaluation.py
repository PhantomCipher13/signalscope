"""
Phase 1 unit tests — Evaluation metrics.

Tests metric computation on known synthetic label/probability arrays.
No real model or training required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compute_metrics, format_metrics_report


class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        probs = np.array([0.0, 0.1, 0.9, 1.0], dtype=np.float32)
        m = compute_metrics(labels, probs)
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["f1"] == 1.0

    def test_random_predictions_auc_near_half(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=200).astype(np.int32)
        probs = rng.uniform(0, 1, size=200).astype(np.float32)
        m = compute_metrics(labels, probs)
        # Random predictions → ROC-AUC near 0.5 (not exactly)
        assert 0.35 <= m["roc_auc"] <= 0.65

    def test_output_keys_present(self):
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        probs = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
        m = compute_metrics(labels, probs)
        for key in ("roc_auc", "accuracy", "precision", "recall", "f1",
                    "false_positive_rate", "tp", "fp", "tn", "fn",
                    "n_samples", "n_real", "n_synthetic", "threshold"):
            assert key in m, f"Missing metric key: {key}"

    def test_n_samples_correct(self):
        labels = np.array([0, 0, 1, 1, 0], dtype=np.int32)
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3], dtype=np.float32)
        m = compute_metrics(labels, probs)
        assert m["n_samples"] == 5
        assert m["n_real"] == 3
        assert m["n_synthetic"] == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([]), np.array([]))

    def test_single_class_does_not_crash(self):
        labels = np.array([0, 0, 0, 0], dtype=np.int32)
        probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        # Should not raise; ROC-AUC will be NaN
        m = compute_metrics(labels, probs)
        assert np.isnan(m["roc_auc"])

    def test_threshold_effect(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        probs = np.array([0.6, 0.7, 0.6, 0.7], dtype=np.float32)
        # threshold 0.5 → all predicted synthetic → FP=2
        m_low = compute_metrics(labels, probs, threshold=0.5)
        # threshold 0.8 → all predicted real → TP=0
        m_high = compute_metrics(labels, probs, threshold=0.8)
        assert m_low["fp"] > m_high["fp"]

    def test_format_metrics_report_returns_string(self):
        labels = np.array([0, 1], dtype=np.int32)
        probs = np.array([0.2, 0.8], dtype=np.float32)
        m = compute_metrics(labels, probs)
        report = format_metrics_report(m, split="val")
        assert isinstance(report, str)
        assert "ROC-AUC" in report
