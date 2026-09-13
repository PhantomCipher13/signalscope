#!/usr/bin/env python3
"""
SignalScope — scripts/stress_test.py
======================================
Run the full adaptive stress-test pipeline on a single image.

Usage:
    python scripts/stress_test.py --image <path>
    python scripts/stress_test.py --image <path> --json
    python scripts/stress_test.py --image <path> --no-calibration

Pipeline:
    original image
    → detector (baseline probability)
    → calibrated probability (if calibration available)
    → JPEG Q90 probe
    → JPEG Q70 probe
    → JPEG Q50 probe
    → resize 75% probe
    → resize 50% probe
    → reliability aggregation (mean, std, stability)
    → structured result

Scientific constraints:
    - Each probe is generated from the ORIGINAL image independently.
    - No probe chains another probe's output.
    - Failed probes are reported explicitly — probability is never fabricated.
    - Stability is None when < 2 successful probes.
    - Reliability thresholds are reported as threshold_not_configured if null.
    - This result is NOT a benchmark metric.
    - CIFAKE-trained model may not generalise to other distributions.

PYTHON: C:\\Python311\\python.exe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT  = PROJECT_ROOT / "models" / "fast_baseline_checkpoint.pt"
DEFAULT_CALIBRATION = PROJECT_ROOT / "outputs" / "calibration_exp1.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SignalScope adaptive stress-test: runs 5 probes on a single image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image",       type=Path, required=True,
                   help="Path to the input image.")
    p.add_argument("--checkpoint",  type=Path, default=None,
                   help=f"Checkpoint .pt file. Default: {DEFAULT_CHECKPOINT}")
    p.add_argument("--calibration", type=Path, default=None,
                   help=f"Calibration JSON. Default: {DEFAULT_CALIBRATION}")
    p.add_argument("--device",      type=str, default="auto",
                   help="Device: auto | cpu | cuda | dml.")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip calibration even if calibration file exists.")
    p.add_argument("--json",        action="store_true",
                   help="Output result as JSON.")
    return p.parse_args()


def _load_calibration(path: Path) -> tuple[float | None, str]:
    """Return (temperature, status). Both None/'not_calibrated' if unavailable."""
    if not path.exists():
        return None, "not_calibrated"
    try:
        import json as _json
        data = _json.loads(path.read_text(encoding="utf-8"))
        inner = data.get("calibration_result", data)
        status = inner.get("status", data.get("calibration_status", ""))
        temp   = data.get("temperature") or inner.get("temperature")
        if status == "calibrated" and temp and float(temp) > 0:
            return float(temp), "calibrated"
        return None, f"unavailable ({status})"
    except Exception as e:
        return None, f"load_error ({e})"


def _validate_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"File not found: {path}"
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported format: {path.suffix!r}"
    if path.stat().st_size == 0:
        return False, "File is empty."
    try:
        from PIL import Image
        img = Image.open(path)
        img.verify()
        return True, ""
    except Exception as e:
        return False, f"Cannot open image: {e}"


def _print_stress_result(result: dict, image_path: Path) -> None:
    """Pretty-print the stress-test result to stdout."""
    err = result.get("error")
    if err:
        print(f"\n[ERROR] {err}")
        return

    base_raw  = result.get("baseline_raw_probability")
    base_cal  = result.get("baseline_calibrated_probability")
    cal_st    = result.get("baseline_calibration_status", "unknown")
    cal_t     = result.get("calibration_temperature")
    pred      = (result.get("final_prediction") or "unknown").upper()
    final_p   = result.get("final_probability")

    n_cfg     = result.get("probes_configured", 0)
    n_run     = result.get("probes_run", 0)
    n_ok      = result.get("probes_successful", 0)
    n_fail    = result.get("probes_failed", 0)

    mean_p    = result.get("mean_probability")
    std_p     = result.get("std_probability")
    stability = result.get("stability")
    stop      = result.get("stop_reason", "unknown")
    thresh_ok = result.get("threshold_configured", False)
    evidence  = result.get("evidence_sufficient", False)
    t_s       = result.get("total_time_s", 0)
    probes    = result.get("probe_results", [])

    print("\n" + "="*60)
    print("  SignalScope — Adaptive Stress Test")
    print("="*60)
    print(f"  Image: {image_path.name}")
    print()

    # Baseline
    print("  ── Baseline ──────────────────────────────────────")
    print(f"  Raw probability (synthetic):        {base_raw:.4f}")
    if base_cal is not None:
        print(f"  Calibrated probability (synthetic): {base_cal:.4f}  (T={cal_t:.4f})")
    else:
        print(f"  Calibration:                        {cal_st}")
    print()

    # Prediction
    prob_disp = f"{final_p:.4f}" if final_p is not None else "N/A"
    print(f"  ┌─ FINAL PREDICTION: {pred}  (p={prob_disp})")
    print()

    # Probe summary
    print("  ── Probe Summary ─────────────────────────────────")
    print(f"  Configured: {n_cfg}  |  Run: {n_run}  |  OK: {n_ok}  |  Failed: {n_fail}")
    if mean_p is not None:
        print(f"  Mean probability across probes: {mean_p:.4f}")
    if std_p is not None:
        print(f"  Std  deviation:                 {std_p:.4f}")
    if stability is not None:
        print(f"  Stability (1 - std):            {stability:.4f}")
    else:
        print(f"  Stability:                      insufficient evidence (< 2 successful probes)")
    print(f"  Evidence sufficient (≥2 probes): {evidence}")
    print()

    # Reliability
    print("  ── Reliability ───────────────────────────────────")
    if thresh_ok:
        print(f"  Threshold configured: YES")
        print(f"  Stop reason:          {stop}")
    else:
        print(f"  Threshold configured: NO")
        print(f"  Status:               threshold_not_configured")
        print(f"  All {n_run} probes run deterministically (no early stopping).")
    print()

    # Per-probe breakdown
    print("  ── Per-Probe Results ─────────────────────────────")
    for pr in probes:
        name    = pr.get("probe_name", "?")
        ok      = pr.get("success", False)
        raw_p   = pr.get("raw_probability")
        cal_p   = pr.get("calibrated_probability")
        err_msg = pr.get("error", "")
        pt      = pr.get("processing_time_s", 0)
        if ok:
            prob_str = f"raw={raw_p:.4f}"
            if cal_p is not None:
                prob_str += f"  cal={cal_p:.4f}"
            print(f"  [{name:12s}] ✓  {prob_str}  ({pt:.2f}s)")
        else:
            print(f"  [{name:12s}] ✗  FAILED: {err_msg}")
    print()
    print(f"  Total time: {t_s:.2f}s")
    print()
    print("  DISCLAIMER: This is a development stress-test tool.")
    print("  Results are NOT benchmark metrics.")
    print("  CIFAKE-trained model — in-distribution only.")
    print("="*60 + "\n")


def main() -> None:
    args  = parse_args()
    ckpt  = args.checkpoint  or DEFAULT_CHECKPOINT
    cal_f = args.calibration or DEFAULT_CALIBRATION

    result: dict = {"error": None}

    # ── Validate image ────────────────────────────────────────────────────
    ok, msg = _validate_image(args.image)
    if not ok:
        result["error"] = msg
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n[ERROR] {msg}")
        sys.exit(1)

    if not ckpt.exists():
        result["error"] = f"Checkpoint not found: {ckpt}"
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n[ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    # ── Load calibration ──────────────────────────────────────────────────
    cal_temperature, cal_status = (None, "not_calibrated") if args.no_calibration \
                                  else _load_calibration(cal_f)

    # ── Load model ────────────────────────────────────────────────────────
    import torch
    from PIL import Image

    from src.detector    import load_checkpoint, resolve_device
    from src.preprocessing import build_val_transforms
    from src.robustness  import (
        BUILTIN_PROBES, ProbeRunner, StressTestEngine, StressTestConfig,
    )

    device = resolve_device(args.device)

    try:
        model, ckpt_meta = load_checkpoint(ckpt, device=device)
        model.eval()
    except Exception as e:
        result["error"] = f"Failed to load checkpoint: {e}"
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n[ERROR] {e}")
        sys.exit(1)

    model_cfg  = ckpt_meta.get("model_config", {})
    input_size = model_cfg.get("input_size", 224)
    transform  = build_val_transforms(image_size=input_size)

    # ── Build runner + engine ─────────────────────────────────────────────
    runner = ProbeRunner(
        model=model,
        val_transform=transform,
        device=device,
        calibration_temperature=cal_temperature,
        calibration_status=cal_status,
    )
    engine = StressTestEngine(probe_runner=runner, config=StressTestConfig())

    # ── Run ───────────────────────────────────────────────────────────────
    try:
        image = Image.open(args.image).convert("RGB")
        st_result = engine.run(image)
    except Exception as e:
        result["error"] = f"Stress test failed: {e}"
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n[ERROR] {e}")
        sys.exit(1)

    # ── Build output dict ─────────────────────────────────────────────────
    output = st_result.to_dict()
    output["calibration_temperature"]       = cal_temperature
    output["baseline_calibration_status"]   = cal_status
    output["model_architecture"]            = model_cfg.get("architecture", "unknown")
    output["checkpoint_epoch"]              = ckpt_meta.get("epoch", -1) + 1
    output["image_path"]                    = str(args.image)
    output["error"]                         = None

    if args.json:
        print(json.dumps(output, indent=2, default=str))
    else:
        _print_stress_result(output, args.image)


if __name__ == "__main__":
    main()
