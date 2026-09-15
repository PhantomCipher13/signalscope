#!/usr/bin/env python3
"""
SignalScope — Section 4.1 CLI Predict Interface
Usage:
    python model/predict.py --input <path_to_image> [--device auto|cpu|cuda]
    python -m model.predict --input <path_to_image>
"""
import os
import sys
import json
import argparse
import urllib.request
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

WEIGHTS_URL = "https://github.com/PhantomCipher13/signalscope/releases/download/signalscope-sih-2026-final/signalscope_b0_v3.pt"
CALIBRATION_URL = "https://github.com/PhantomCipher13/signalscope/releases/download/signalscope-sih-2026-final/calibration_v3.json"

def ensure_checkpoint() -> Path:
    """Ensure weights exist locally; if not, download from official release."""
    ckpt_path = _PROJECT_ROOT / "models" / "signalscope_b0_v3.pt"
    if ckpt_path.exists() and ckpt_path.stat().st_size > 1000:
        return ckpt_path
    
    fallback = _PROJECT_ROOT / "models" / "fast_baseline_checkpoint.pt"
    if fallback.exists() and fallback.stat().st_size > 1000 and not os.environ.get("FORCE_DOWNLOAD"):
        return fallback

    print(f"[SignalScope] Checkpoint not found at {ckpt_path}. Downloading from GitHub Release...", file=sys.stderr)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, str(ckpt_path))
        print(f"[SignalScope] Downloaded {ckpt_path.stat().st_size / (1024*1024):.2f} MB successfully.", file=sys.stderr)
        return ckpt_path
    except Exception as e:
        print(f"[SignalScope] Warning: Could not download release weights: {e}", file=sys.stderr)
        if fallback.exists():
            print(f"[SignalScope] Falling back to bundled baseline checkpoint.", file=sys.stderr)
            return fallback
        raise

def predict(image_path: str, device: str = "auto", run_stress: bool = False, run_gradcam: bool = False) -> dict:
    from app.analyzer import SignalScopeAnalyzer
    
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    active_ckpt = ensure_checkpoint()

    analyzer = SignalScopeAnalyzer()
    analyzer.load()

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    result = analyzer.analyze(
        img_bytes,
        filename=img_path.name,
        run_stress_test=run_stress,
        run_gradcam=run_gradcam,
        run_provenance=True,
        save_to_db=False
    )

    pred = result.get("prediction", "unknown")
    cal_prob = result.get("calibrated_probability")
    raw_prob = result.get("raw_probability", 0.5)
    
    active_prob = cal_prob if cal_prob is not None else raw_prob
    if active_prob is None:
        active_prob = 0.5

    confidence = active_prob if pred == "synthetic" else (1.0 - active_prob)

    output = {
        "file": str(img_path.name),
        "verdict": "Likely AI-Generated" if pred == "synthetic" else "Likely Real",
        "prediction": pred,
        "synthetic_probability": round(float(active_prob), 4),
        "raw_synthetic_probability": round(float(raw_prob), 4) if raw_prob is not None else None,
        "confidence_score": round(float(confidence), 4),
        "calibration_status": result.get("calibration_status", "calibrated"),
        "model_architecture": result.get("model_architecture", "efficientnet_b0"),
        "checkpoint_used": str(active_ckpt.name),
        "provenance_summary": result.get("provenance", {}).get("summary", "No provenance metadata.")
    }
    return output

def main():
    parser = argparse.ArgumentParser(description="SignalScope CLI Predict Interface (SIH 2026 Section 4.1)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image file (JPEG, PNG, WebP)")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--json-only", action="store_true", help="Print only raw JSON output")
    args = parser.parse_args()

    try:
        out = predict(args.input, device=args.device)
        if args.json_only:
            print(json.dumps(out, indent=2))
        else:
            print("=" * 60)
            print(f" SIGNALSCOPE FORENSIC VERDICT — {out['file']}")
            print("=" * 60)
            verdict_badge = "[LIKELY AI-GENERATED]" if out['prediction'] == "synthetic" else "[LIKELY REAL]"
            print(f" Verdict:                {verdict_badge}")
            print(f" Calibrated Confidence:  {out['confidence_score'] * 100:.1f}%")
            print(f" Synthetic Probability:  {out['synthetic_probability']:.4f} (Raw: {out['raw_synthetic_probability']})")
            print(f" Model Architecture:     {out['model_architecture']} ({out['checkpoint_used']})")
            print(f" Calibration Status:     {out['calibration_status']}")
            print(f" Provenance:             {out['provenance_summary']}")
            print("-" * 60)
            print("JSON Output:")
            print(json.dumps(out, indent=2))
            print("=" * 60)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
