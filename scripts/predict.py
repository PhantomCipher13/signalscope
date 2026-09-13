#!/usr/bin/env python3
"""
SignalScope — scripts/predict.py
===================================
Single-image inference with calibrated probability and reliability status.

Usage:
    python scripts/predict.py --image <path>
    python scripts/predict.py --image <path> --checkpoint models/fast_baseline_checkpoint.pt
    python scripts/predict.py --image <path> --json

Output fields (where available):
    prediction            : real | synthetic
    raw_probability       : uncalibrated model output P(synthetic)
    calibrated_probability: temperature-scaled P(synthetic), if calibration loaded
    calibration_status    : calibrated | not_calibrated | unavailable
    calibration_temperature
    model_architecture
    model_version
    checkpoint
    checkpoint_epoch
    input_size
    processing_time_s
    image_path
    image_size_original   : (W, H) of the source image
    warnings              : list of any non-fatal issues

SCIENTIFIC DISCLAIMER:
    - Manual inference results are NOT benchmark metrics.
    - Do NOT compare manual predictions to validation/test statistics.
    - This is a development/testing tool only.
    - CIFAKE-trained model may not generalise to other image distributions.

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

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
DEFAULT_CHECKPOINT   = PROJECT_ROOT / "models" / "fast_baseline_checkpoint.pt"
DEFAULT_CALIBRATION  = PROJECT_ROOT / "outputs" / "calibration_exp1.json"
MAX_IMAGE_DIM        = 8192  # refuse absurdly large images
MIN_IMAGE_DIM        = 4     # refuse degenerate images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SignalScope — single-image real-vs-synthetic inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", type=Path, required=True,
                   help="Path to the input image file.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help=f"Checkpoint .pt file. Default: {DEFAULT_CHECKPOINT}")
    p.add_argument("--calibration", type=Path, default=None,
                   help=f"Calibration JSON file. Default: {DEFAULT_CALIBRATION}")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto | cpu | cuda | dml.")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip calibration even if file exists.")
    p.add_argument("--json", action="store_true",
                   help="Print result as JSON (machine-readable).")
    return p.parse_args()


def _validate_image(path: Path) -> tuple[bool, str, dict]:
    """Return (ok, error_message, image_info)."""
    if not path.exists():
        return False, f"File not found: {path}", {}
    if not path.is_file():
        return False, f"Not a file: {path}", {}
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported format: {path.suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        ), {}
    if path.stat().st_size == 0:
        return False, f"File is empty: {path}", {}
    if path.stat().st_size > 100 * 1024 * 1024:  # 100 MB hard cap
        return False, f"File too large ({path.stat().st_size / 1e6:.1f} MB). Max 100 MB.", {}

    try:
        from PIL import Image
        img = Image.open(path)
        img.verify()  # check for corruption
        img = Image.open(path)  # reopen after verify
        w, h = img.size
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            return False, f"Image too small: {w}×{h}. Minimum {MIN_IMAGE_DIM}px.", {}
        if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
            return False, f"Image too large: {w}×{h}. Maximum {MAX_IMAGE_DIM}px.", {}
        mode = img.mode
        return True, "", {"width": w, "height": h, "mode": mode}
    except Exception as e:
        return False, f"Cannot open image: {e}", {}


def _load_calibration(cal_path: Path) -> dict | None:
    """Return calibration dict or None if unavailable."""
    if not cal_path.exists():
        return None
    try:
        with cal_path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    # ── Resolve paths ─────────────────────────────────────────────────────
    checkpoint_path = args.checkpoint or DEFAULT_CHECKPOINT
    cal_path        = args.calibration or DEFAULT_CALIBRATION

    result: dict = {
        "image_path":             str(args.image),
        "checkpoint":             str(checkpoint_path),
        "prediction":             None,
        "raw_probability":        None,
        "calibrated_probability": None,
        "calibration_status":     "not_calibrated",
        "calibration_temperature": None,
        "model_architecture":     None,
        "model_version":          None,
        "checkpoint_epoch":       None,
        "input_size":             None,
        "image_size_original":    None,
        "processing_time_s":      None,
        "warnings":               [],
        "error":                  None,
    }

    # ── Validate image ────────────────────────────────────────────────────
    ok, err_msg, img_info = _validate_image(args.image)
    if not ok:
        result["error"] = err_msg
        _print_result(result, args.json)
        sys.exit(1)

    result["image_size_original"] = [img_info.get("width"), img_info.get("height")]
    if img_info.get("mode") not in ("RGB", "RGBA", "L"):
        result["warnings"].append(
            f"Unusual image mode: {img_info.get('mode')!r}. Converting to RGB."
        )

    # ── Validate checkpoint ───────────────────────────────────────────────
    if not checkpoint_path.exists():
        result["error"] = f"Checkpoint not found: {checkpoint_path}"
        _print_result(result, args.json)
        sys.exit(1)

    # ── Load calibration (optional) ───────────────────────────────────────
    cal_data = None
    if not args.no_calibration:
        cal_data = _load_calibration(cal_path)
        if cal_data is None:
            result["warnings"].append(
                f"Calibration file not found or unreadable: {cal_path}. "
                "Reporting raw probability only."
            )

    # ── Device ────────────────────────────────────────────────────────────
    from src.detector import resolve_device, load_checkpoint
    from src.preprocessing import build_val_transforms

    device = resolve_device(args.device)

    # ── Load model ────────────────────────────────────────────────────────
    try:
        model, ckpt_meta = load_checkpoint(checkpoint_path, device=device)
        model.eval()
    except Exception as e:
        result["error"] = f"Failed to load checkpoint: {e}"
        _print_result(result, args.json)
        sys.exit(1)

    model_cfg = ckpt_meta.get("model_config", {})
    input_size = model_cfg.get("input_size", 224)
    result["model_architecture"] = model_cfg.get("architecture", "unknown")
    result["model_version"]      = f"{result['model_architecture']}_v1"
    result["checkpoint_epoch"]   = ckpt_meta.get("epoch", -1) + 1  # 0-indexed → 1-indexed
    result["input_size"]         = input_size

    # ── Preprocess & Inference ────────────────────────────────────────────
    try:
        import torch
        from PIL import Image

        transform = build_val_transforms(image_size=input_size)
        img = Image.open(args.image).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)
                raw_prob_synthetic = probs[0, 1].item()
            else:
                raw_prob_synthetic = torch.sigmoid(logits[0, 0]).item()

    except Exception as e:
        result["error"] = f"Inference failed: {e}"
        _print_result(result, args.json)
        sys.exit(1)

    result["raw_probability"] = round(raw_prob_synthetic, 6)
    result["prediction"]      = "synthetic" if raw_prob_synthetic >= 0.5 else "real"

    # ── Apply calibration ─────────────────────────────────────────────────
    if cal_data:
        # calibration_exp1.json stores results nested under calibration_result
        cal_inner = cal_data.get("calibration_result", cal_data)
        cal_status = cal_inner.get("status", cal_data.get("calibration_status", ""))
        temperature = cal_data.get("temperature") or cal_inner.get("temperature")
        if cal_status == "calibrated" and temperature and temperature > 0:
            try:
                import numpy as np
                # Temperature scaling on logits
                # We re-derive logit from probability then scale
                raw_logit = float(np.log(raw_prob_synthetic / (1.0 - raw_prob_synthetic + 1e-12)))
                scaled_logit = raw_logit / temperature
                cal_prob = float(1.0 / (1.0 + np.exp(-scaled_logit)))
                result["calibrated_probability"]  = round(cal_prob, 6)
                result["calibration_status"]      = "calibrated"
                result["calibration_temperature"] = round(temperature, 4)
                # Override prediction label using calibrated probability
                result["prediction"] = "synthetic" if cal_prob >= 0.5 else "real"
            except Exception as e:
                result["warnings"].append(f"Calibration application failed: {e}. Using raw probability.")
                result["calibration_status"] = "application_error"
        else:
            result["calibration_status"] = f"unavailable ({cal_status})"
    else:
        result["calibration_status"] = "not_calibrated"

    # ── Timing ────────────────────────────────────────────────────────────
    result["processing_time_s"] = round(time.perf_counter() - t_start, 3)

    _print_result(result, args.json)


def _print_result(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return

    err = result.get("error")
    if err:
        print(f"\n[ERROR] {err}")
        if result.get("warnings"):
            for w in result["warnings"]:
                print(f"[WARNING] {w}")
        return

    pred    = (result.get("prediction") or "unknown").upper()
    raw_p   = result.get("raw_probability")
    cal_p   = result.get("calibrated_probability")
    cal_st  = result.get("calibration_status", "unknown")
    cal_t   = result.get("calibration_temperature")
    arch    = result.get("model_architecture", "unknown")
    ver     = result.get("model_version", "unknown")
    epoch   = result.get("checkpoint_epoch", "?")
    ckpt    = result.get("checkpoint", "unknown")
    isize   = result.get("input_size", "?")
    orig    = result.get("image_size_original")
    t_s     = result.get("processing_time_s", 0)

    orig_str = f"{orig[0]}×{orig[1]}" if orig else "unknown"

    print("\n" + "="*56)
    print("  SignalScope — Image Analysis Result")
    print("="*56)
    print(f"  Image:        {result.get('image_path','')}")
    print(f"  Original size: {orig_str}")
    print(f"  Model input:  {isize}×{isize}")
    print()
    print(f"  ┌─ PREDICTION:  {pred}")
    if cal_p is not None:
        print(f"  ├─ Calibrated probability (synthetic): {cal_p:.4f}")
        print(f"  ├─ Raw probability (synthetic):        {raw_p:.4f}")
        print(f"  └─ Temperature (T={cal_t:.4f}):       calibration applied")
    else:
        print(f"  ├─ Raw probability (synthetic): {raw_p:.4f}")
        print(f"  └─ Calibration: {cal_st}")
    print()
    print(f"  Model:        {arch} ({ver})")
    print(f"  Checkpoint:   {Path(ckpt).name} (epoch {epoch})")
    print(f"  Time:         {t_s:.3f}s")
    print()

    if result.get("warnings"):
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    ⚠  {w}")
        print()

    print("  DISCLAIMER: This is a development inference tool.")
    print("  Raw and calibrated probabilities are from a CIFAKE-trained model.")
    print("  Results are IN-DISTRIBUTION only. Do not claim generalisation.")
    print("  This result is NOT a scientific benchmark measurement.")
    print("="*56 + "\n")


if __name__ == "__main__":
    main()
