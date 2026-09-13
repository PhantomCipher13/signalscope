# SignalScope — Phase 1 Developer Notes

## Overview

Phase 1 implements the baseline ML pipeline: dataset manifest generation,
preprocessing, model training, evaluation, and single-image inference.

This document describes how to use the Phase 1 tools.

---

## Environment

**Python executable** (Windows — DirectML-compatible torch build):
```
C:\Python311\python.exe
```

> **Note on torch DLLs**: The original `AppData\Local\python-embed\` location
> was blocked by a Windows Application Control (WDAC) policy that prevents
> loading unsigned DLLs. The working fix was:
> 1. Copy python-embed to `C:\Python311\`
> 2. Install `torch-directml` (Microsoft's signed DirectML build of PyTorch)
>    instead of the standard CPU wheel.
>
> `torch-directml` installs `torch 2.4.1` with DLLs that pass WDAC checks.
> For production GPU training on Colab T4, use the standard CUDA build.

**Run scripts from the project root** (`signalscope/`):
```powershell
$py = "C:\Python311\python.exe"
```

---

## Dataset Placement

Place your dataset under a directory with **real** and **fake** (or equivalent) subdirectories:

```
<dataset_root>/
  real/          ← real images (.jpg, .png, .bmp, .webp)
  fake/          ← AI-generated images
```

Or nested structures:
```
<dataset_root>/
  real/
  synthetic/
    midjourney/
    stablediffusion/
```

The label is inferred from directory names containing keywords like:
- **Real**: `real`, `genuine`, `authentic`, `original`
- **Synthetic**: `fake`, `ai`, `synthetic`, `generated`, `gan`, `diffusion`

Update `configs/default.yaml` → `dataset.real_dir_keywords` / `synthetic_dir_keywords` if your dataset uses different naming.

---

## Step-by-Step Workflow

### 1. Inspect the dataset

```powershell
& $py scripts/inspect_dataset.py --root <dataset_root>
& $py scripts/inspect_dataset.py --root <dataset_root> --check-images
```

Review:
- Total image counts
- Class balance (real vs synthetic)
- Generator hints (directory names)
- Duplicate filenames
- Any unknown-label warnings

### 2. Create the manifest

```powershell
& $py scripts/create_manifest.py --root <dataset_root>
```

Options:
```
--output data/splits/manifest.csv   (default)
--train 0.70 --val 0.15 --test 0.15 (default fractions)
--seed 42                            (default seed)
```

> **Scientific integrity**: If generator metadata is available (visible in
> subdirectory names), manually verify the `generator` column before training.
> For true unseen-generator evaluation, ensure no generator appears in both
> train and test splits.

### 3. Train the model

```powershell
& $py scripts/train.py
```

Options:
```
--manifest data/splits/manifest.csv
--epochs 20          (default from config)
--batch-size 32
--lr 0.0001
--device auto        (auto | cpu | cuda)
--debug              (2 epochs, 20 samples — for pipeline testing)
```

**The test split is never evaluated during training.**

Best checkpoint is saved to `models/best_checkpoint.pt` based on validation ROC-AUC.

Training history is saved to `outputs/training_history.csv`.

Raw validation predictions are saved to `outputs/val_predictions.npz`
(required for Phase 2 temperature calibration).

### 4. Evaluate the model

```powershell
# Evaluate on validation split (safe during development)
& $py scripts/evaluate.py --checkpoint models/best_checkpoint.pt --split val

# Evaluate on test split (ONCE only, after all tuning is complete)
& $py scripts/evaluate.py --checkpoint models/best_checkpoint.pt --split test
```

> **CAUTION**: Only use `--split test` once, after all checkpoint selection,
> calibration, and threshold tuning is complete on validation data.

### 5. Single-image prediction

```powershell
& $py scripts/predict.py --image <path_to_image> --checkpoint models/best_checkpoint.pt
```

Output:
```
Prediction:            SYNTHETIC
Synthetic probability:  0.8734
Real probability:       0.1266

IMPORTANT: The probability above is the raw model output.
It is NOT a calibrated confidence score.
Calibration (temperature scaling) is implemented in Phase 2.
```

---

## Reproducibility

- All splits are deterministic for a fixed `--seed`.
- Seed is set in `configs/default.yaml → seed.value`.
- The seed is recorded in every checkpoint.
- Training config is recorded in every checkpoint.

---

## Checkpoint Contents

Each checkpoint (`models/best_checkpoint.pt`) contains:
- `model_state_dict` — model weights
- `model_config` — architecture, num_classes, dropout_rate
- `train_config` — epochs, batch_size, learning_rate, etc.
- `epoch` — best epoch index (0-based)
- `val_metrics` — validation metrics at the best checkpoint
- `seed` — random seed

---

## Known Limitations

1. **No dataset included**: A real dataset must be provided before training
   can proceed. See Dataset Placement above.

2. **Raw probability, not calibrated confidence**: Phase 1 outputs are
   uncalibrated model probabilities. Calibration (temperature scaling)
   is implemented in Phase 2.

3. **No reliability analysis**: Multi-version stability testing is Phase 2.

4. **No Grad-CAM**: Visual explanations are Phase 2.

5. **No provenance analysis**: EXIF/C2PA is Phase 2.

6. **No frontend**: The web interface is a later phase.

7. **CPU-only environment**: The current setup uses PyTorch CPU.
   For GPU training, install the CUDA variant of PyTorch from pytorch.org.

---

## Running Tests

```powershell
# All tests
& $py -m pytest -v

# Unit tests only (fast)
& $py -m pytest tests/unit/ -v

# Integration tests (includes mini training pipeline)
& $py -m pytest tests/integration/ -v

# Environment check
& $py scripts/check_environment.py
```
