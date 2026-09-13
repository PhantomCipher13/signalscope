# SignalScope

**Telling Real From Synthetic in the Age of Generative Media**

SIH 2026 · Problem Statement PS-1702 · AI/ML Track

---

## Overview

SignalScope is a forensic analysis system for detecting AI-generated images. It combines a calibrated neural network detector with robustness probing, gradient-based explanation (Grad-CAM), and image provenance/metadata analysis to give a multi-evidence verdict on any uploaded photograph.

The system is designed to be scientifically honest: it reports calibrated probabilities rather than binary labels, exposes uncertainty, and clearly communicates its limitations.

---

## Live Demo

**Public URL:** [https://signalscope.vercel.app](https://signalscope.vercel.app)

**API docs:** Available at `/docs` on the backend server.

---

## System Architecture

```
Browser (frontend/index.html)
        │
        │  HTTP POST /analyze, /feedback, GET /health
        ▼
FastAPI backend (app/main.py)
        │
        ├── EfficientNet-B0 detector (src/detector.py)
        ├── Temperature calibration (src/calibration/)
        ├── Robustness probes / stress test (src/robustness/)
        ├── Grad-CAM (src/gradcam.py)
        ├── Provenance / EXIF (src/provenance.py)
        ├── Reliability engine (src/reliability/)
        └── Persistence
             ├── Supabase (cloud, production)
             └── SQLite (local fallback)
```

**Deployment:**
- Frontend: Vercel (static hosting)
- Backend: Self-hosted FastAPI server (CPU inference)
- Database: Supabase (PostgreSQL)

---

## ML Approach

### Model

- **Architecture:** EfficientNet-B0 (via `timm`)
- **Input:** 224×224 RGB images, ImageNet normalisation
- **Output:** Binary classification — Real (0) vs Synthetic (1)
- **Dropout:** 0.3 for regularisation

### Training Dataset

- **CIFAKE** (Bird & Lotfi, 2023)
  - 60,000 real images (CIFAR-10)
  - 60,000 synthetic images (Stable Diffusion v1.4)
  - 80/10/10 train/validation/test split

### Training Configuration

- Optimiser: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Epochs: 5 (rapid baseline)
- Augmentation: RandomHorizontalFlip, ColorJitter, RandomRotation(10°)
- Mixed precision: disabled (CPU/DirectML compatibility)

### Calibration

Temperature scaling was applied after training to reduce overconfidence:

| Metric | Before calibration | After calibration |
|--------|-------------------|-------------------|
| ECE    | 0.1040            | 0.0480            |
| Brier  | 0.1236            | 0.1086            |

Temperature: **T = 2.8756** (fitted on validation set)

### Evaluation Results (In-Distribution)

| Metric | Validation | Test (in-distribution) |
|--------|------------|------------------------|
| ROC-AUC | **0.9316** | **0.9319** |
| Accuracy | 84.63% | 85.01% |
| Precision | 88.86% | — |
| Recall | 79.20% | — |
| F1 | 0.8375 | — |
| FPR | 9.93% | — |

> ⚠️ **Scientific Limitation:** These results are **in-distribution** (Stable Diffusion v1.4 only). The model has not been evaluated on DALL-E, Midjourney, Firefly, or other generators. Real-world performance on unseen generators is unknown. True generalisation evaluation is planned for a subsequent training phase.

---

## Features

### 1. Calibrated Detection
The raw model probability is temperature-scaled to produce a well-calibrated confidence score. Values near 50% indicate high uncertainty.

### 2. Robustness Probes
Five transformation probes re-run inference under common real-world image degradation:

| Probe | Transformation |
|-------|---------------|
| JPEG Q90 | Light JPEG compression |
| JPEG Q70 | Medium JPEG compression |
| JPEG Q50 | Heavy JPEG compression |
| Resize 75% | Downscale to 75% then back |
| Resize 50% | Downscale to 50% then back |

Stability score = fraction of probes agreeing with the baseline prediction.

### 3. Grad-CAM Explanation
Class Activation Mapping highlights which image regions most influenced the prediction. This is a model explanation aid — not proof of manipulation.

### 4. Provenance Analysis
EXIF metadata is extracted and displayed. The system explicitly notes that **absence of metadata is not evidence of AI generation**.

### 5. Reliability Engine
Aggregates baseline + probe results to assess prediction consistency. Reports `stable`, `unstable`, or `insufficient_evidence` based on observed variation.

### 6. Structured Feedback
Testers can submit `correct`, `incorrect`, or `unsure` verdicts. Feedback is stored in Supabase and linked to the analysis record for post-test evaluation. **Feedback does not trigger automatic retraining.**

---

## API Reference

### `GET /health`
Returns model load status and calibration status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "calibration_loaded": true,
  "architecture": "efficientnet_b0",
  "version": "1.0.0",
  "schema_version": 1
}
```

### `POST /analyze`
Upload an image for full forensic analysis.

**Form fields:**
- `file` (required): Image file (JPEG/PNG/WebP/BMP, max 20MB)
- `run_stress_test` (optional, default true): Run robustness probes
- `run_gradcam` (optional, default true): Generate Grad-CAM
- `run_provenance` (optional, default true): Extract EXIF

**Response:** Full analysis JSON including prediction, calibrated probability, probe results, Grad-CAM base64, provenance, reliability status.

### `POST /feedback`
Submit tester feedback for an analysis.

```json
{
  "analysis_id": 42,
  "verdict": "incorrect",
  "comment": "This is a real photo — model predicted synthetic incorrectly"
}
```

### `GET /results/{analysis_id}`
Retrieve a previous analysis by ID.

### `GET /admin/feedback`
Protected endpoint. Requires `X-Admin-Token` header or `?token=` query parameter.

---

## Installation & Local Development

### Prerequisites
- Python 3.11+
- ~4 GB RAM (for model loading)
- GPU optional (CPU inference supported)

### Setup

```bash
git clone https://github.com/PhantomCipher13/signalscope.git
cd signalscope

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials and ADMIN_TOKEN
```

### Running locally

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: http://localhost:8000

### Running tests

```bash
python -m pytest tests/ -v
```

---

## Deployment

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes (for cloud) | Supabase project URL |
| `SUPABASE_KEY` | Yes (for cloud) | Supabase anon/publishable key |
| `ADMIN_TOKEN` | Yes | Token for /admin endpoints |
| `SIGNALSCOPE_DEVICE` | No | `auto`/`cpu`/`cuda` (default: auto) |
| `SIGNALSCOPE_DB_BACKEND` | No | `supabase` or `sqlite` |

**Never commit `.env` to version control.**

### Vercel (Frontend)

1. Fork/clone this repository to GitHub
2. Connect GitHub repo to Vercel
3. Vercel detects `vercel.json` — no additional configuration needed
4. The frontend builds automatically from `frontend/`
5. API rewrites in `vercel.json` proxy to the backend

### Backend Server

Run the FastAPI server on any machine with Python 3.11+:

```bash
# For production with public tunnel (no domain):
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
cloudflared tunnel --url http://localhost:8000
```

Update `vercel.json` rewrite destinations to point to your backend URL.

---

## Security

- All secrets via environment variables — never in source code
- Upload validation: MIME type + file size enforced server-side
- Admin endpoints protected with bearer token authentication
- No stack traces exposed in API responses
- No image binaries stored in database
- CORS configured for cross-origin frontend access

---

## Supabase Schema

Tables: `analyses`, `probe_results`, `provenance`, `feedback`

View: `feedback_with_analysis`

See `docs/supabase_schema.sql` for DDL.

---

## Known Limitations

1. **In-distribution only.** Trained on Stable Diffusion v1.4; unseen-generator performance is unknown.
2. **Compression artefacts.** JPEG-compressed real images may score higher AI probability.
3. **Very small images.** Images smaller than 32×32 are upscaled, which may affect results.
4. **Cold start.** First request after server restart takes 5–10 seconds for model loading.
5. **CPU inference.** Without GPU, full analysis (including Grad-CAM and 5 probes) takes 5–30 seconds.
6. **No C2PA support.** Content credentials (C2PA) library is not currently installed.

---

## Tomorrow's Plan (14 September)

- Inspect and validate chosen Hugging Face multi-generator dataset
- Create proper train/validation/hold-out splits
- Fine-tune EfficientNet-B0 on multi-generator data
- Evaluate on unseen-generator hold-out set
- Recalibrate with temperature scaling
- Freeze and deploy best checkpoint
- Update benchmark results in documentation

---

## Reproducibility

```bash
# Exact evaluation on CIFAKE test set:
python scripts/evaluate.py --checkpoint models/fast_baseline_checkpoint.pt --split test

# Temperature calibration:
python scripts/calibrate.py --checkpoint models/fast_baseline_checkpoint.pt

# Full stress test:
python scripts/stress_test.py --image path/to/image.jpg
```

All evaluation scripts are in `scripts/`. No held-out test data is included in the repository (too large), but the CIFAKE dataset is publicly available at [https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).

---

## Citation

If using SignalScope or its evaluation methodology in research:

> SignalScope (2026). *Telling Real From Synthetic in the Age of Generative Media.* SIH 2026 Submission.
>
> Bird, J.J. & Lotfi, A. (2023). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access.

---

## Team

SIH 2026 · Internal Hackathon Submission
