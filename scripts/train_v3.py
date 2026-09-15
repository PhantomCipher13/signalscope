#!/usr/bin/env python3
"""
SignalScope — scripts/train.py
================================
Training entry point for the baseline real-vs-synthetic detector.

Requirements before running:
  1. Dataset manifest must exist (run scripts/create_manifest.py first).
  2. Set dataset_root in configs/default.yaml or pass --manifest.
  3. Ensure ML dependencies are installed.

Usage:
    python scripts/train.py
    python scripts/train.py --manifest data/splits/manifest.csv
    python scripts/train.py --epochs 5 --batch-size 16 --device cpu
    python scripts/train.py --debug   # 2 epochs, 20 samples/split

Scientific integrity:
  - Checkpoint selection uses ONLY validation metrics.
  - The test split is never evaluated during training.
  - Raw predictions are saved for Phase 2 calibration.
  - All metrics are measured from real model predictions; none are fabricated.

PYTHON: C:\\Users\\Admin\\AppData\\Local\\python-embed\\python.exe
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import load_manifest, filter_split, SignalScopeDataset, make_dataloaders
from src.detector import create_model_from_config, resolve_device, save_checkpoint
from src.evaluation import evaluate_dataloader, format_metrics_report, save_predictions
from src.preprocessing import build_transforms_from_config
from src.utils import get_config, get_logger, seed_from_config, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the SignalScope baseline detector.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Path to train manifest CSV (overrides config).")
    p.add_argument("--val-manifest", type=Path, default=None,
                   help="Path to separate validation manifest CSV. If not given, uses 'val' split from --manifest.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of training epochs (overrides config).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size (overrides config).")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (overrides config).")
    p.add_argument("--device", type=str, default=None,
                   help="Device: auto | cpu | cuda | dml (overrides config).")
    p.add_argument("--architecture", type=str, default=None,
                   help="Model architecture override, e.g. efficientnet_b0, efficientnet_b3 (overrides config).")
    p.add_argument("--input-size", type=int, default=None,
                   help="Input image size override, e.g. 224, 300 (overrides config).")
    p.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"],
                   help="Optimizer override: adamw (default) or sgd. SGD avoids AdamW lerp fallback on DirectML.")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Checkpoint output directory (overrides config).")
    p.add_argument("--checkpoint-name", type=str, default=None,
                   help="Checkpoint filename (overrides config default 'best_checkpoint.pt').")
    p.add_argument("--experiment-name", type=str, default=None,
                   help="Experiment name for database recording.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (overrides config).")
    p.add_argument("--no-db", action="store_true",
                   help="Skip database persistence (useful for quick tests).")
    p.add_argument("--resume-from", type=str, default=None, help="Checkpoint to warm-start from")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: 2 epochs, 20 samples per split.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_dir=PROJECT_ROOT / log_cfg.get("log_dir", "logs"),
        filename=log_cfg.get("filename", "signalscope.log"),
    )
    logger = get_logger("train")

    # ── Seed ────────────────────────────────────────────────────────────────
    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        seed = args.seed
    else:
        seed_from_config(cfg)
        seed = cfg.get("seed", {}).get("value", 42)

    # ── Resolve training config ─────────────────────────────────────────────
    train_cfg = cfg.get("training", {})
    epochs = args.epochs or train_cfg.get("epochs", 20)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = train_cfg.get("pin_memory", False)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    best_metric_key = train_cfg.get("best_metric", "roc_auc")
    checkpoint_dir = args.checkpoint_dir or Path(
        train_cfg.get("checkpoint_dir", "models")
    )
    checkpoint_name = (
        args.checkpoint_name
        or train_cfg.get("checkpoint_name", "best_checkpoint.pt")
    )
    history_path = Path(train_cfg.get("history_path", "outputs/training_history.csv"))
    device_pref = args.device or train_cfg.get("device", "auto")

    exp_cfg = cfg.get("experiment", {})
    experiment_name = args.experiment_name or f"cifake_baseline_{int(time.time())}"
    dataset_name = exp_cfg.get("dataset", "CIFAKE")
    generator = exp_cfg.get("generator", "stable_diffusion_v1.4")
    evaluation_type = exp_cfg.get("evaluation_type", "in_distribution")

    # Primary manifest (train split)
    manifest_path = args.manifest or Path(
        exp_cfg.get("manifests", {}).get("train", "data/manifests/train.csv")
    )
    # Separate validation manifest if provided
    val_manifest_path = args.val_manifest or None
    if val_manifest_path is None:
        val_manifest_override = exp_cfg.get("manifests", {}).get("validation")
        if val_manifest_override:
            val_manifest_path = Path(val_manifest_override)

    if args.debug:
        epochs = 2
        batch_size = 8
        logger.info("DEBUG mode: epochs=2, batch_size=8")

    # ── CIFAKE 32×32 Limitation Warning + Architecture override ──────────────
    model_cfg = cfg.get("model", {})
    # Apply CLI overrides to model_cfg before using it anywhere
    if args.architecture:
        model_cfg = dict(model_cfg)  # don't mutate global cfg
        model_cfg["architecture"] = args.architecture
        logger.info("Architecture override: %s", args.architecture)
    if args.input_size:
        model_cfg = dict(model_cfg)
        model_cfg["input_size"] = args.input_size
        logger.info("Input size override: %d", args.input_size)

    input_size = model_cfg.get("input_size", 300)
    arch_name  = model_cfg.get("architecture", "efficientnet_b3")

    print("\n" + "="*60)
    print("SCIENTIFIC LIMITATION NOTICE")
    print("="*60)
    print(f"  CIFAKE images are 32×32 pixels (CIFAR-10 resolution).")
    print(f"  Model: {arch_name}, input size: {input_size}×{input_size}.")
    print(f"  Images will be upscaled {input_size/32:.1f}× — this introduces")
    print(f"  interpolation artefacts and means the model is NOT learning")
    print(f"  from natural high-res real-world image statistics.")
    print(f"  Results on CIFAKE test set are IN-DISTRIBUTION only.")
    print(f"  Do NOT extrapolate to real-world high-resolution deployment.")
    print("="*60 + "\n")

    # ── Check manifest ───────────────────────────────────────────────────────
    if not manifest_path.exists():
        logger.error(
            "Manifest not found: %s\n"
            "  Run: python scripts/create_manifest.py --root <dataset_dir>",
            manifest_path,
        )
        print(f"\n[BLOCKED] Manifest missing: {manifest_path}")
        sys.exit(1)

    # ── Device ──────────────────────────────────────────────────────────────
    device = resolve_device(device_pref)

    # ── Transforms (using effective input_size) ──────────────────────────────
    # Temporarily patch cfg model.input_size so build_transforms_from_config picks it up
    _cfg_patched = dict(cfg)
    _cfg_patched["model"] = model_cfg
    train_transform = build_transforms_from_config(_cfg_patched, mode="train")
    val_transform   = build_transforms_from_config(_cfg_patched, mode="val")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    logger.info("Loading manifest: %s", manifest_path)
    all_samples = load_manifest(manifest_path)
    train_samples = filter_split(all_samples, "train")

    # If a separate val manifest is provided, load it independently
    if val_manifest_path and val_manifest_path.exists():
        logger.info("Loading validation manifest: %s", val_manifest_path)
        val_all = load_manifest(val_manifest_path)
        # Accept any split label (or all rows if manifest is already a pure val set)
        val_by_label = filter_split(val_all, "validation")
        if not val_by_label:
            val_samples = val_all  # entire separate manifest is the val set
        else:
            val_samples = val_by_label
    else:
        val_samples = filter_split(all_samples, "val")

    if args.debug:
        import random as _rnd
        _rnd.seed(seed)
        train_samples = _rnd.sample(train_samples, min(20, len(train_samples)))
        val_samples = _rnd.sample(val_samples, min(20, len(val_samples)))
        logger.info("DEBUG: using %d train, %d val samples", len(train_samples), len(val_samples))

    if not train_samples:
        logger.error("Training split is empty. Check manifest.")
        sys.exit(1)
    if not val_samples:
        logger.error("Validation split is empty. Check manifest.")
        sys.exit(1)

    from torch.utils.data import DataLoader
    train_ds = SignalScopeDataset(train_samples, transform=train_transform)
    val_ds = SignalScopeDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    logger.info(
        "Dataset: train=%d val=%d | batch_size=%d | device=%s",
        len(train_ds), len(val_ds), batch_size, device,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    # Use patched model_cfg (which may have CLI architecture/input_size overrides)
    from src.detector import create_model
    num_classes = model_cfg.get("num_classes", 2)
    dropout_rate = model_cfg.get("dropout", 0.3)
    model = create_model(
        architecture=model_cfg.get("architecture", "efficientnet_b3"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    # num_classes=2 → CrossEntropyLoss; num_classes=1 → BCEWithLogitsLoss
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────────────────────
    # SGD preferred for DirectML — AdamW has a lerp CPU-fallback on DML
    optimizer_choice = args.optimizer or train_cfg.get("optimizer", "adamw")
    if optimizer_choice == "sgd":
        from torch.optim import SGD
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        logger.info("Optimizer: SGD (momentum=0.9)")
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info("Optimizer: AdamW")

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    scheduler_name = train_cfg.get("lr_scheduler", "cosine")
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    history_path = PROJECT_ROOT / history_path
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric_value = -float("inf")
    history_rows: list[dict] = []

    logger.info(
        "Training: epochs=%d, lr=%.1e, weight_decay=%.1e",
        epochs, lr, weight_decay,
    )
    print(f"\n=== SignalScope Training ===")
    print(f"  Epochs:    {epochs}")
    print(f"  Batch:     {batch_size}")
    print(f"  LR:        {lr}")
    print(f"  Device:    {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)

            if num_classes == 1:
                loss = criterion(logits.squeeze(1), labels.float())
            else:
                loss = criterion(logits, labels)

            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        # ── Validation ────────────────────────────────────────────────────
        val_result = evaluate_dataloader(model, val_loader, device, return_predictions=True)
        val_metrics = val_result["metrics"]
        val_loss = _compute_val_loss(
            model, val_loader, device, criterion, num_classes
        )

        current_metric = val_metrics.get(best_metric_key, 0.0)
        is_best = current_metric > best_metric_value

        print(
            f"  Epoch {epoch+1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_{best_metric_key}={current_metric:.4f} | "
            f"time={epoch_time:.1f}s"
            + (" [BEST]" if is_best else "")
        )
        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_%s=%.4f%s",
            epoch + 1, epochs, train_loss, val_loss,
            best_metric_key, current_metric,
            " [BEST]" if is_best else "",
        )

        if is_best:
            best_metric_value = current_metric
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                epoch=epoch,
                val_metrics=val_metrics,
                train_cfg={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "device": str(device),
                },
                model_cfg=model_cfg,
                seed=seed,
                optimizer_state=optimizer.state_dict(),
            )
            # Save raw val predictions for Phase 2 calibration
            save_predictions(
                output_path=PROJECT_ROOT / "outputs" / "val_predictions.npz",
                labels=val_result["labels"],
                probs=val_result["probs"],
                split="val",
            )

        # ── History row ────────────────────────────────────────────────────
        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "is_best": is_best,
            **{f"val_{k}": round(v, 6) for k, v in val_metrics.items()
               if k not in ("tp", "fp", "tn", "fn", "n_samples", "n_real",
                             "n_synthetic", "threshold")},
        }
        history_rows.append(row)

    # ── Save training history ──────────────────────────────────────────────
    if history_rows:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history_rows[0].keys())
            writer.writeheader()
            writer.writerows(history_rows)
        logger.info("Training history saved: %s", history_path)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n=== Training Complete ===")
    print(f"  Best val {best_metric_key}: {best_metric_value:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  History:    {history_path}")
    print()
    print("  NOTE: The test split was NOT evaluated during training.")
    print("  Run scripts/evaluate.py to evaluate on the in-distribution test split.")

    # ── Database persistence ───────────────────────────────────────────────
    training_duration_s = sum(r.get("time", 0) for r in history_rows) if history_rows else None
    best_epoch_found = None
    for r in history_rows:
        if r.get("is_best"):
            best_epoch_found = r["epoch"]

    experiment_id = None
    if not args.no_db:
        try:
            from src.database.connection import DatabaseManager, _resolve_db_path
            from src.database.schema import init_db
            from src.database.repositories import (
                insert_experiment, update_experiment, insert_metrics_batch
            )
            db_path = _resolve_db_path()
            db = DatabaseManager(db_path)
            db.connect()
            init_db(db)

            # Gather best val metrics for persistence
            # Use the LAST row marked is_best (= the final best checkpoint),
            # NOT the first — all-improving runs mark every epoch is_best.
            best_val_metrics = {}
            if history_rows:
                best_candidates = [r for r in history_rows if r.get("is_best")]
                best_row = best_candidates[-1] if best_candidates else history_rows[-1]
                for k, v in best_row.items():
                    if k.startswith("val_") and k not in ("val_loss",):
                        clean_name = k[4:]  # strip 'val_' prefix
                        if isinstance(v, (int, float)):
                            best_val_metrics[clean_name] = v

            aug_cfg = cfg.get("augmentation", {})
            experiment_id = insert_experiment(
                db,
                experiment_name=experiment_name,
                dataset=dataset_name,
                model_name=model_cfg.get("architecture", "efficientnet_b3"),
                seed=seed,
                model_version=f"{model_cfg.get('architecture', 'efficientnet_b3')}_v1",
                architecture=model_cfg.get("architecture", "efficientnet_b3"),
                checkpoint_path=str(checkpoint_path),
                train_manifest=str(manifest_path),
                val_manifest=str(val_manifest_path) if val_manifest_path else None,
                epochs_trained=epochs,
                best_epoch=best_epoch_found,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=weight_decay,
                scheduler=scheduler_name if "scheduler_name" in dir() else "cosine",
                augmentation_config=aug_cfg,
                generator=generator,
                notes=(
                    f"CIFAKE baseline. Images upscaled from 32x32 to {input_size}x{input_size}. "
                    f"Evaluation type: {evaluation_type}. "
                    f"NOT representative of real-world high-resolution deployment."
                ),
            )

            if best_val_metrics:
                insert_metrics_batch(
                    db, experiment_id, "validation",
                    best_val_metrics,
                    evaluation_type=evaluation_type,
                )

            db.close()
            print(f"\n  [DB] Experiment persisted: id={experiment_id}")
            print(f"  [DB] Database: {db_path}")
        except Exception as db_err:
            print(f"\n  [DB WARNING] Could not persist to database: {db_err}")
            print("  Training results are still saved to checkpoint and history files.")

    if experiment_id is not None:
        print(f"\n  Experiment ID: {experiment_id}")


def _compute_val_loss(model, loader, device, criterion, num_classes: int) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if num_classes == 1:
                loss = criterion(logits.squeeze(1), labels.float())
            else:
                loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


if __name__ == "__main__":
    main()
