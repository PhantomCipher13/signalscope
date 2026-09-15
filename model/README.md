# SignalScope Model Interface (SIH 2026 Section 4.1 & 7.1)

This directory contains the required training and inference interfaces for SignalScope.

---

## 1. Quick Inference (<10 seconds)

Run inference on any image using the standardized predict CLI:

```bash
# Basic usage
python model/predict.py --input path/to/image.jpg

# JSON-only output for automated pipeline integration
python model/predict.py --input path/to/image.jpg --json-only

# Force CPU inference
python model/predict.py --input path/to/image.jpg --device cpu
```

### Python API Usage

```python
from model.predict import predict

result = predict("path/to/image.jpg")
print(result["verdict"])                # 'Likely AI-Generated' or 'Likely Real'
print(result["confidence_score"])       # 0.9667 (96.7%)
print(result["synthetic_probability"])  # Calibrated probability
```

---

## 2. Model Weights & Architecture

- **Architecture:** `EfficientNet-B0` (ImageNet pretrained backbone via `timm`, 4,010,110 parameters)
- **Production Checkpoint:** `signalscope_b0_v3.pt` (48.5 MB)
- **Official Weights Release Link:** [GitHub Release v3](https://github.com/PhantomCipher13/signalscope/releases/download/signalscope-sih-2026-final/signalscope_b0_v3.pt)
- **Automatic Weight Retrieval:** If `signalscope_b0_v3.pt` is not present locally upon calling `predict.py`, the script automatically downloads the official release weights.

---

## 3. Training & Reproduction

To reproduce the V3 model training on 97,000 unique CIFAKE images:

```bash
python model/train.py \
  --manifest data/manifests/train_97k.csv \
  --val-manifest data/manifests/fast_validation.csv \
  --resume-from models/signalscope_b0_v2.pt \
  --output-dir experiments/cifake_97k_v3 \
  --epochs 1 --lr 0.005 --batch-size 32
```
