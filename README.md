# SignalScope — Telling Real From Synthetic in the Age of Generative Media

> **SIH 2026 Submission** · [Demo Video](#6-demo-video--deployed-app) · [Deployed App](https://signalscope-kappa.vercel.app) · [Model Report](#model-report)

SignalScope is a professional AI-image forensics system that detects AI-generated images with calibrated confidence, visual explanations, robustness probing, and provenance metadata analysis.

---

## 1. Modules Built

### Core Module
- **Binary Real-vs-AI-Generated Classification** — EfficientNet-B0 fine-tuned on CIFAKE, returning calibrated probability + discrete verdict.

### Bonus Modules
| Module | Status | Details |
|--------|--------|---------|
| **A — Explainability** | ✅ Completed | Grad-CAM highlights regions that drove the prediction |
| **B — Robustness / Calibration** | ✅ Completed | Temperature scaling (validation-only), 5 JPEG/resize stress probes |
| **C — Provenance / Metadata** | ✅ Completed | EXIF + C2PA Content Credential extraction |
| **D — Reliability Scoring** | ✅ Completed | Probe stability aggregated into a reliability verdict |
| **Unseen-Generator Split** | ⚠️ Not Completed | Blocked by network constraints (see Limitations) |

---

## 2. Setup & Run Instructions

> A judge should be able to reproduce a prediction in under 10 minutes.

### Prerequisites
- Python 3.11
- Git

### Quick Start

```bash
# 1. Clone
git clone https://github.com/PhantomCipher13/signalscope.git
cd signalscope

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server (model loads automatically)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. Open in browser
#    http://localhost:8000
#    Upload any image → click Analyze → see full forensic report
```

### Reproduce a Single Prediction via API

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/your/image.jpg"
```

Expected response fields: `prediction`, `raw_probability`, `calibrated_probability`, `calibration_status`, `robustness_probes`, `grad_cam`, `provenance`, `analysis_id`.

### Verify Health

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true,"calibration_loaded":true,"architecture":"efficientnet_b0"}
```

### Run Tests

```bash
pytest tests/     # 344 tests, all pass
```

### Reproduce Training (V3)

```bash
# Requires CIFAKE dataset at data/raw/CIFAKE/
python scripts/train_v3.py \
  --manifest data/manifests/train_97k.csv \
  --val-manifest data/manifests/fast_validation.csv \
  --resume-from models/signalscope_b0_v2.pt \
  --output-dir experiments/cifake_97k_v3 \
  --epochs 1 --lr 0.005 --batch-size 32
```

---

## 3. Dataset

| Split | Images | Source |
|-------|--------|--------|
| Training (V3) | 97,000 | CIFAKE native train split (100k − 3k validation hold-out) |
| Validation | 3,000 | CIFAKE native train split (held out; `fast_validation.csv`) |
| Test (final, used once) | 20,000 | CIFAKE native test split (`test_indistribution.csv`) |

**CIFAKE Dataset**
- **Source:** [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) / [Bird & Lotfi, 2023](https://arxiv.org/abs/2303.08854)
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **Real images:** CIFAR-10 photographs
- **Synthetic images:** Stable Diffusion v1.4 generated at 32×32 px
- **Balance:** Perfectly balanced — 50% real, 50% synthetic in all splits
- **Resolution note:** Native 32×32 px images are upscaled to 224×224 for EfficientNet-B0 inference

No additional private or proprietary datasets were used.

---

## 4. Reported Metrics

> ⚠️ All metrics below are from the **V3 final model** evaluated **once** on the untouched 20,000-image native CIFAKE test split. The test set was not used for model selection, calibration, or threshold tuning.

### Primary Metrics — V3 (SIH 2026 Submission)

| Metric | Value |
|--------|-------|
| **Overall ROC-AUC** | **0.9799** |
| **Macro-F1** | **0.9226** |
| Accuracy | 91.92% |
| Precision | 88.53% |
| Recall | 96.32% |
| False-Positive Rate | 12.48% |
| Threshold | 0.50 |

### Confusion Matrix (V3 · 20,000-image test set)

```
                  Predicted REAL   Predicted SYNTHETIC
Actual REAL           8,752              1,248   (FP)
Actual SYNTHETIC        368   (FN)       9,632
```

### Unseen-Generator Split AUC

Not available. Evaluation was attempted on GenImage++ (FLUX, SD3) but blocked by network constraints. This metric **cannot be reported** and is explicitly disclosed as a limitation.

### Progression (same 20k test set)

| Model | ROC-AUC | Macro-F1 | Accuracy | FPR |
|-------|---------|----------|----------|-----|
| Baseline (V1) | 0.9316 | 0.8375 | 84.63% | 9.93% |
| V2 (20k train) | 0.9484 | 0.8474 | 82.58% | 31.54% |
| **V3 (97k train) ← submitted** | **0.9799** | **0.9226** | **91.92%** | **12.48%** |

---

## 5. Architecture, Robustness & Calibration

### Model Architecture
- **Backbone:** EfficientNet-B0 (`timm`, ImageNet pretrained)
- **Head:** 2-class softmax (real / ai-generated)
- **Input:** 224×224 RGB
- **Parameters:** 4,010,110 total / trainable
- **Dropout:** 0.30 on the classification head

### Training (V3)
- **Training images:** 97,000 (unique; CIFAKE native train − 3k validation)
- **Warm-start:** From V2 checkpoint (`models/signalscope_b0_v2.pt`)
- **Optimizer:** AdamW · lr=0.005 · weight_decay=0.01
- **Epochs:** 1 (Epoch 2 deliberately not run)
- **Batch size:** 32

### Calibration
- **Method:** Temperature Scaling (validation predictions only — zero test contamination)
- **Temperature T:** 0.9804 (near-unity — model was already well-calibrated)
- **ECE:** 0.0602 → 0.0601 (marginal improvement)
- **Brier Score:** 0.0669 → 0.0669 (neutral)

### Robustness Probes (5 probes per image)

| Probe | Transform |
|-------|-----------|
| JPEG Q=90 | Light compression |
| JPEG Q=70 | Moderate compression |
| JPEG Q=50 | Heavy compression |
| Resize 75% | Downscale then restore |
| Resize 50% | Aggressive downscale then restore |

Probe results are aggregated into a **Reliability Score** (stable / unstable / insufficient evidence).

### Explainability
- **Method:** Grad-CAM on the final convolutional layer
- **Output:** Heatmap overlay showing which image regions drove the prediction
- **Caveat:** Grad-CAM shows model attention, not proof of manipulation

### Provenance
- EXIF metadata extraction (camera, GPS, software)
- C2PA Content Credential detection (where present)
- Explicit policy: absence of metadata is **not** treated as evidence of AI generation

### Known Limitations
- **Resolution Bias:** Trained on upscaled 32×32 images. Performance on native high-resolution images is unknown and likely lower.
- **Generator Bias:** Trained only on Stable Diffusion v1.4 outputs. Cross-generator generalization (FLUX, DALL-E, Midjourney, SD3) has **not** been demonstrated.
- **In-distribution only:** All reported metrics are in-distribution on CIFAKE. Do not extrapolate AUC 0.9799 to real-world deployment without further validation.
- **Unseen-generator split:** Not evaluated — infrastructure constraints prevented download of the required evaluation dataset.
- **Metadata spoofing:** A motivated adversary can strip or forge EXIF/C2PA metadata; provenance analysis is supplementary evidence only.

---

## 6. Demo Video & Deployed App

| Resource | Link |
|----------|------|
| 🎬 **Demo Video** | *(Add your YouTube/Drive link here before submission)* |
| 🌐 **Deployed App (Vercel)** | https://signalscope-kappa.vercel.app |
| 🔗 **GitHub Repository** | https://github.com/PhantomCipher13/signalscope |
| 🏷️ **Submission Tag** | [`signalscope-sih-2026-final`](https://github.com/PhantomCipher13/signalscope/tree/signalscope-sih-2026-final) |

---

## Model Report

**Task:** Binary real-vs-AI-generated image classification. Bonus modules: Grad-CAM explainability, temperature-scaled calibration, robustness probing (5 JPEG/resize transforms), provenance metadata analysis.

**Data & Split:**
Source: CIFAKE (CC BY 4.0) — CIFAR-10 real + Stable Diffusion v1.4 synthetic, 32×32 px.
Train: 97,000 images (CIFAKE native train split minus 3k validation hold-out).
Validation: 3,000 images (calibration fitting only — zero test leakage).
Test: 20,000 images (CIFAKE native test split — evaluated once at submission).

**Model / Approach:**
Backbone: EfficientNet-B0 (timm, ImageNet pretrained), warm-started from V2.
Optimizer: AdamW (lr=0.005, weight_decay=0.01), 1 epoch, batch size 32.
Calibration: Temperature Scaling on validation predictions only (T=0.9804).

**Metric & Result (V3, 20k test set):**
- Overall ROC-AUC: **0.9799**
- Unseen-generator-split AUC: **Not available** (explicitly disclosed)
- Macro-F1: **0.9226** · Accuracy: 91.92% · FPR: 12.48% at threshold 0.50
- Confusion matrix: TP=9,632 · FP=1,248 · TN=8,752 · FN=368

**Baseline Comparison:**
- Provided baseline (V1): AUC 0.9316, F1 0.8375
- V3 vs baseline: AUC +0.0483, F1 +0.0851
- V3 vs V2 (20k→97k training): AUC +0.0315, F1 +0.0752, Accuracy +9.34%, FPR reduced by 19.06 pp

**Limitations:**
Trained exclusively on Stable Diffusion v1.4 outputs at 32×32 native resolution. Cross-generator and high-resolution generalization are unvalidated. Unseen-generator split AUC is not reported.

---

## Repository Structure

```
signalscope/
├── app/                    # FastAPI backend (analyzer, main, database)
├── src/                    # ML modules (robustness, gradcam, provenance)
├── scripts/                # train_v3.py, calibrate.py, evaluate.py
├── model/                  # Production checkpoint + calibration
│   ├── signalscope_b0_v3.pt        (48 MB — final submitted model)
│   └── calibration_v3.json
├── experiments/
│   └── cifake_97k_v3/      # Metrics, calibration, experiment artifacts
├── data/manifests/         # train_97k.csv, fast_validation.csv, test_indistribution.csv
├── frontend/               # HTML/CSS/JS web interface
├── tests/                  # 344 unit tests (pytest)
├── report/                 # One-page model report
├── configs/                # model.yaml, default.yaml
├── requirements.txt
└── README.md
```

---

## Team

Built during Internal Hackathon (September 2026).
