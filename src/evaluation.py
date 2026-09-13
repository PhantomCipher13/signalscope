"""
SignalScope — src/evaluation.py
================================
Evaluation utilities: metrics computation from model predictions.

Produces: ROC-AUC, accuracy, precision, recall, F1, confusion matrix,
false-positive rate. Raw predictions are preserved for later calibration.

IMPORTANT: Never use the final unseen-generator test split for
checkpoint selection, calibration, or threshold tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("signalscope.evaluation")


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute binary classification metrics.

    Parameters
    ----------
    labels:
        Ground-truth labels, shape (N,), values in {0, 1}.
    probs:
        Predicted synthetic probability, shape (N,), values in [0, 1].
    threshold:
        Decision threshold for hard predictions.

    Returns
    -------
    dict with keys: roc_auc, accuracy, precision, recall, f1,
                    false_positive_rate, tp, fp, tn, fn, n_samples,
                    n_real, n_synthetic, threshold.
    """
    from sklearn.metrics import (  # lazy
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)

    if len(labels) == 0:
        raise ValueError("Labels array is empty — cannot compute metrics.")
    if len(np.unique(labels)) < 2:
        logger.warning(
            "Only one class present in labels. "
            "ROC-AUC is undefined and will be reported as NaN."
        )

    preds = (probs >= threshold).astype(np.int32)

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = float("nan")

    accuracy = float(accuracy_score(labels, preds))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")

    return {
        "roc_auc": round(roc_auc, 6),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_samples": len(labels),
        "n_real": int((labels == 0).sum()),
        "n_synthetic": int((labels == 1).sum()),
        "threshold": threshold,
    }


def evaluate_dataloader(
    model,
    dataloader,
    device,
    return_predictions: bool = True,
) -> dict:
    """Run evaluation over a DataLoader and return metrics + raw outputs.

    Parameters
    ----------
    model:
        Trained model (already on *device*, in eval mode).
    dataloader:
        DataLoader yielding (images, labels).
    device:
        Compute device.
    return_predictions:
        If True, include raw label/prob arrays in output for calibration.

    Returns
    -------
    dict with keys: metrics (dict), labels (np.ndarray), probs (np.ndarray)
    """
    import torch  # lazy

    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)

            num_classes = logits.shape[-1]
            if num_classes == 1:
                probs = torch.sigmoid(logits[:, 0]).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    labels_arr = np.array(all_labels, dtype=np.int32)
    probs_arr = np.array(all_probs, dtype=np.float32)

    metrics = compute_metrics(labels_arr, probs_arr)

    result: dict = {"metrics": metrics}
    if return_predictions:
        result["labels"] = labels_arr
        result["probs"] = probs_arr

    return result


def save_predictions(
    output_path: str | Path,
    labels: np.ndarray,
    probs: np.ndarray,
    split: str = "val",
) -> None:
    """Save raw labels and probabilities to a NumPy .npz file.

    These raw outputs are needed for Phase 2 temperature calibration.

    Parameters
    ----------
    output_path:
        Output .npz file path.
    labels:
        Ground-truth labels, shape (N,).
    probs:
        Raw model synthetic probabilities, shape (N,).
        NOT calibrated — calibration is Phase 2.
    split:
        Split name for logging.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, labels=labels, probs=probs, split=np.array([split]))
    logger.info("Raw predictions saved: %s (%d samples)", output_path, len(labels))


def format_metrics_report(metrics: dict, split: str = "val") -> str:
    """Format metrics into a human-readable string."""
    lines = [
        f"\n--- Evaluation Results ({split}) ---",
        f"  Samples:           {metrics['n_samples']:>8d}",
        f"  Real:              {metrics['n_real']:>8d}",
        f"  Synthetic:         {metrics['n_synthetic']:>8d}",
        f"  Threshold:         {metrics['threshold']:>8.3f}",
        f"",
        f"  ROC-AUC:           {metrics['roc_auc']:>8.4f}",
        f"  Accuracy:          {metrics['accuracy']:>8.4f}",
        f"  Precision:         {metrics['precision']:>8.4f}",
        f"  Recall:            {metrics['recall']:>8.4f}",
        f"  F1:                {metrics['f1']:>8.4f}",
        f"  False-positive:    {metrics['false_positive_rate']:>8.4f}",
        f"",
        f"  Confusion matrix:",
        f"    TP={metrics['tp']}  FP={metrics['fp']}",
        f"    FN={metrics['fn']}  TN={metrics['tn']}",
        f"---",
    ]
    return "\n".join(lines)
