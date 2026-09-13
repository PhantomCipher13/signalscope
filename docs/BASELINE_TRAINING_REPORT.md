# SignalScope — CIFAKE Baseline Training Report

> [!WARNING]  
> **This is a template. Values marked `[PENDING]` will be filled in automatically by `scripts/post_training_pipeline.py` after training completes.**
>
> **Do NOT replace `[PENDING]` values manually with invented numbers.**

---

## 1. Training Configuration

| Parameter | Value |
|---|---|
| Dataset | CIFAKE |
| Training split | `data/manifests/train.csv` — 85,000 images |
| Validation split | `data/manifests/validation.csv` — 15,000 images |
| Architecture | EfficientNet-B3 (pretrained on ImageNet) |
| Input size | 300×300 (upscaled from 32×32) |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 1e-4 (AdamW) |
| Weight decay | 1e-2 |
| Dropout | 0.30 |
| Seed | 42 |
| Device | CPU |
| Label convention | 0 = real, 1 = synthetic/AI-generated |
| Experiment name | `cifake_baseline_eb3_e10` |

---

## 2. Training Results

> [!NOTE]
> Training is currently running. Results will be filled in when training completes and `post_training_pipeline.py` is executed.

| Metric | Value |
|---|---|
| Best epoch | `[PENDING]` |
| Final training loss | `[PENDING]` |
| Best validation loss | `[PENDING]` |
| Best validation ROC-AUC | `[PENDING]` |
| Best validation accuracy | `[PENDING]` |
| Best validation F1 | `[PENDING]` |
| Checkpoint path | `models/best_checkpoint.pt` |
| Training duration | `[PENDING]` |
| Database experiment ID | `[PENDING]` |

---

## 3. Validation Evaluation (In-Distribution)

**Split:** `data/manifests/validation.csv`  
**Evaluation type:** `in_distribution`  
**Note:** Same Stable Diffusion v1.4 generator as training — this is NOT a generalisation test.

| Metric | Value |
|---|---|
| ROC-AUC | `[PENDING]` |
| Accuracy | `[PENDING]` |
| F1 | `[PENDING]` |
| Precision | `[PENDING]` |
| Recall | `[PENDING]` |
| N samples | 15,000 |
| N real | 7,500 |
| N synthetic | 7,500 |

---

## 4. CIFAKE In-Distribution Test Evaluation

**Split:** `data/manifests/test_indistribution.csv`  
**Evaluation type:** `in_distribution`

> [!IMPORTANT]
> **This is an IN-DISTRIBUTION test only.**
>
> The CIFAKE test set uses the **same Stable Diffusion v1.4 generator** as the training set.  
> It measures whether the model has learned useful features — it does **NOT** measure generalisation to unseen generators.  
> This split must **never** be called `unseen_generator` evaluation.  
> No generalisation claims may be derived from this result.

| Metric | Value |
|---|---|
| ROC-AUC | `[PENDING]` |
| Accuracy | `[PENDING]` |
| F1 | `[PENDING]` |
| Precision | `[PENDING]` |
| Recall | `[PENDING]` |
| N samples | 20,000 |
| N real | 10,000 |
| N synthetic | 10,000 |

---

## 5. Temperature Scaling Calibration

**Fitted on:** Validation predictions only  
**Calibration script:** `scripts/calibrate.py`

> [!IMPORTANT]
> Temperature was fitted on the **validation set only**.
> The test set was not used for calibration.
> ECE and Brier values are **measured results**, not performance targets.

| Metric | Value |
|---|---|
| Calibration status | `[PENDING]` |
| Temperature (T) | `[PENDING]` |
| ECE before calibration | `[PENDING]` |
| ECE after calibration | `[PENDING]` |
| Brier score before | `[PENDING]` |
| Brier score after | `[PENDING]` |
| Calibration file | `outputs/calibration_exp<id>.json` |
| Fitted on (n samples) | 15,000 |

---

## 6. Database Persistence

| Item | Status |
|---|---|
| Experiment record | `[PENDING]` |
| Validation metrics | `[PENDING]` |
| Test metrics | `[PENDING]` |
| Calibration record | `[PENDING]` |
| Schema version | 2 |

---

## 7. Sanity Checks

| Check | Status |
|---|---|
| ROC-AUC > 0.50 (not random) | `[PENDING]` |
| No NaN/Inf metrics | `[PENDING]` |
| Evaluation not 0 samples | `[PENDING]` |
| Test evaluation_type = in_distribution | `[PENDING]` |
| Checkpoint size > 1MB | `[PENDING]` |
| No suspicious findings reported | `[PENDING]` |

---

## 8. Known Limitations

| Limitation | Impact |
|---|---|
| CIFAKE images are 32×32 upscaled to 300×300 | Model learns upscaling artefacts, not high-res texture patterns |
| CIFAKE only uses SD v1.4 for synthetic images | Limited diversity of synthetic generators |
| No unseen-generator dataset available locally | Generalisation cannot be measured |
| CPU training | Very slow; no GPU acceleration |
| CIFAKE test ≠ unseen generator test | All CIFAKE splits share the same generator |

---

## 9. What This Evaluation Does NOT Establish

- **Generalisation to unseen generators** — not tested, no dataset available.
- **Real-world deployment performance** — CIFAKE is a controlled benchmark, not real-world images.
- **Calibration quality** — ECE is informative but thresholds are not yet configured.
- **Optimal threshold** — confidence thresholds must be selected from validation data, not test data.

---

## 10. Next Steps After This Baseline

1. Obtain a multi-generator dataset (GenImage, UnbiasedGenImage, or RAID image corpus).
2. Evaluate `best_checkpoint.pt` on unseen generators with `--evaluation-type unseen_generator`.
3. Run reliability probing (Phase 4) to get multi-observation stability estimates.
4. Tune confidence and stability thresholds on validation data.
