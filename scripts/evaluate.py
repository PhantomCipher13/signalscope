#!/usr/bin/env python3
"""
SignalScope — scripts/evaluate.py
===================================
Evaluate a trained checkpoint on a specified data split.

Usage:
    python scripts/evaluate.py --checkpoint models/best_checkpoint.pt \\
                                --manifest data/manifests/validation.csv

    python scripts/evaluate.py --checkpoint models/best_checkpoint.pt \\
                                --manifest data/manifests/test_indistribution.csv \\
                                --split-name test_indistribution \\
                                --evaluation-type in_distribution

    python scripts/evaluate.py --checkpoint models/best_checkpoint.pt \\
                                --manifest data/manifests/validation.csv \\
                                --experiment-id 1

SCIENTIFIC INTEGRITY:
    - The in-distribution test (test_indistribution.csv) must NEVER be called
      "unseen_generator" evaluation — it uses the same SD v1.4 generator as training.
    - --evaluation-type defaults to 'in_distribution' for all CIFAKE splits.
    - The test split MUST NOT be used for threshold tuning or calibration.
    - Results are stored in the database under the correct evaluation_type.

PYTHON: C:\\Python311\\python.exe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_manifest, filter_split, SignalScopeDataset
from src.detector import load_checkpoint, resolve_device
from src.evaluation import evaluate_dataloader, format_metrics_report, save_predictions
from src.preprocessing import build_val_transforms
from src.utils import get_config, get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a SignalScope checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the checkpoint .pt file.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Manifest CSV path. Can be a standalone split CSV or combined manifest.")
    p.add_argument("--split", type=str, default=None,
                   help="Filter manifest to this split value (e.g. 'train', 'validation', 'test'). "
                        "If None, uses all rows in the manifest as the evaluation set.")
    p.add_argument("--split-name", type=str, default=None,
                   help="Human-readable name for this split in output files and DB "
                        "(e.g. 'validation', 'test_indistribution'). Defaults to --split.")
    p.add_argument("--evaluation-type", type=str, default="in_distribution",
                   choices=["in_distribution", "unseen_generator"],
                   help="Evaluation type for DB recording. CIFAKE is always 'in_distribution'.")
    p.add_argument("--experiment-id", type=int, default=None,
                   help="Experiment ID to associate metrics with in the database.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save-predictions", action="store_true",
                   help="Save raw predictions .npz for calibration.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Where to write predictions and metrics JSON.")
    p.add_argument("--no-db", action="store_true",
                   help="Skip database persistence.")
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
    logger = get_logger("evaluate")

    # Determine human-readable split name for output files / DB
    split_name = args.split_name or args.split or "eval"

    # Safety: warn if evaluating test split before calibration is done
    if "test" in split_name.lower():
        print(
            "\n[CAUTION] Evaluating on the TEST split.\n"
            "  This should be done only ONCE after all model selection is complete.\n"
            "  Do NOT use test results to make further model decisions or tune thresholds.\n"
            f"  Evaluation type: {args.evaluation_type}\n"
        )
        if args.evaluation_type == "unseen_generator":
            print(
                "[WARNING] evaluation_type='unseen_generator' specified for test split.\n"
                "  If this is CIFAKE, this is INCORRECT — CIFAKE test set uses the same\n"
                "  Stable Diffusion v1.4 generator as training. Use 'in_distribution'.\n"
            )

    # ── Load model ────────────────────────────────────────────────────────
    device = resolve_device(args.device or cfg.get("training", {}).get("device", "auto"))
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, ckpt_meta = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    logger.info(
        "Checkpoint: epoch=%d, architecture=%s",
        ckpt_meta.get("epoch", -1),
        ckpt_meta.get("model_config", {}).get("architecture", "unknown"),
    )

    # ── Load data ─────────────────────────────────────────────────────────
    exp_cfg = cfg.get("experiment", {})
    if args.manifest:
        manifest_path = args.manifest
    else:
        # Default: use validation manifest from config
        manifest_path = Path(exp_cfg.get("manifests", {}).get("validation", "data/manifests/validation.csv"))

    logger.info("Loading manifest: %s", manifest_path)
    all_samples = load_manifest(manifest_path)

    if args.split:
        split_samples = filter_split(all_samples, args.split)
        if not split_samples:
            # Try direct — maybe the whole manifest IS the split
            split_samples = all_samples
    else:
        # No filter — evaluate on all rows in manifest
        split_samples = all_samples

    logger.info("Evaluating on %d samples (split_name=%s)", len(split_samples), split_name)

    if not split_samples:
        print(f"[ERROR] No samples found. Check manifest and --split argument.")
        sys.exit(1)

    model_cfg = ckpt_meta.get("model_config", {})
    image_size = model_cfg.get("input_size", 256) if model_cfg else 256
    transform = build_val_transforms(image_size=image_size)
    ds = SignalScopeDataset(split_samples, transform=transform)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Evaluate ─────────────────────────────────────────────────────────
    result = evaluate_dataloader(model, loader, device, return_predictions=True)
    metrics = result["metrics"]

    print(format_metrics_report(metrics, split=split_name))

    # ── Save metrics JSON ─────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"metrics_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split_name,
                "evaluation_type": args.evaluation_type,
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": ckpt_meta.get("epoch", -1),
                "manifest": str(manifest_path),
                "n_samples": metrics.get("n_samples"),
                "metrics": metrics,
            },
            f, indent=2,
        )
    print(f"\nMetrics saved: {metrics_path}")

    # ── Save raw predictions ──────────────────────────────────────────────
    if args.save_predictions or "val" in split_name:
        pred_path = output_dir / f"{split_name}_predictions.npz"
        save_predictions(pred_path, result["labels"], result["probs"], split_name)
        print(f"Raw predictions saved: {pred_path}")
        if "val" in split_name:
            print("  (Use these for Phase 2 temperature calibration — val split only.)")

    # ── Database persistence ──────────────────────────────────────────────
    if not args.no_db and args.experiment_id is not None:
        try:
            from src.database.connection import DatabaseManager, _resolve_db_path
            from src.database.schema import init_db
            from src.database.repositories import insert_metrics_batch

            db_path = _resolve_db_path()
            db = DatabaseManager(db_path)
            db.connect()
            init_db(db)

            # Only store numeric metrics
            numeric_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and k not in ("tp", "fp", "tn", "fn")
            }
            insert_metrics_batch(
                db, args.experiment_id, split_name,
                numeric_metrics,
                evaluation_type=args.evaluation_type,
            )
            db.close()
            print(f"\n[DB] Metrics persisted for experiment_id={args.experiment_id}, split={split_name}")
        except Exception as db_err:
            print(f"\n[DB WARNING] Could not persist metrics: {db_err}")

    elif args.experiment_id is None and not args.no_db:
        print("\n[DB] No --experiment-id specified — metrics not persisted to database.")
        print("     Re-run with --experiment-id <id> to save to DB.")


if __name__ == "__main__":
    main()
