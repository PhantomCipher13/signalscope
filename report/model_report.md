# SignalScope — One-Page Model Report (SIH 2026 Section 7.3)

> **Team:** QuantumCrew  
> **Model:** EfficientNet-B0 (SignalScope V3)  
> **Repository:** https://github.com/PhantomCipher13/signalscope  
> **Official Release:** https://github.com/PhantomCipher13/signalscope/releases/tag/signalscope-sih-2026-final  
> **Demo Video:** https://drive.google.com/drive/folders/1TYQbrAgQ7p7bWGrQVU7BHgt-1u1hlusX?usp=drive_link  

---

### 1. Task Definition
- **Core Task:** Binary image forensics — classifying input images as **Authentic (Real)** or **Synthetic (AI-Generated)**.
- **Bonus Modules Attempted:**
  - **Module A (Explainability):** Grad-CAM spatial activation mapping targeting latent diffusion artifacts.
  - **Module B (Robustness & Calibration):** 5-probe transformation stress testing (JPEG Q=90/70/50, Resize 75%/50%) + validation-only Temperature Scaling.
  - **Module C (Provenance):** EXIF camera metadata and C2PA Content Credential verification.
  - **Module D (Reliability Scoring):** Multi-probe stability aggregation ($S = \max(0, 1 - \text{std}(p_{\text{probes}}))$).

---

### 2. Data & Split Methodology
- **Source:** CIFAKE Dataset (Bird & Lotfi, 2023; CC BY 4.0). Real images from CIFAR-10; synthetic images generated via Stable Diffusion v1.4.
- **Training Set (V3):** **97,000 images** (CIFAKE native train split, minus 3k validation hold-out; perfectly 50/50 balanced).
- **Validation Set:** **3,000 images** (`data/manifests/fast_validation.csv`) — used solely for hyperparameter tuning and post-hoc temperature calibration. Zero test data leakage.
- **Test Set (Held-out):** **20,000 images** (`test_indistribution.csv`) — native untouched CIFAKE test split evaluated only at final submission.

---

### 3. Model Architecture & Training Approach
- **Backbone:** EfficientNet-B0 (`timm`, ImageNet-1k pretrained, 4,010,110 parameters).
- **Head:** Linear 2-class classifier with Dropout ($p=0.30$).
- **Optimizer:** AdamW ($\text{lr}=0.005$, $\text{weight\_decay}=0.01$).
- **Epochs:** 1 full epoch over 97,000 samples (warm-started from V2 checkpoint).
- **Calibration:** Validation-fitted Temperature Scaling ($T = 0.9804$, $\text{ECE} = 0.0601$, $\text{Brier} = 0.0669$).

---

### 4. Primary Empirical Results (20,000-Image Untouched Test Set)

| Metric | Measured Value | Benchmark vs V2 (+Δ) |
| :--- | :--- | :--- |
| **Overall ROC-AUC** | **0.9799** | **+0.0315** |
| **Macro-F1 Score** | **0.9226** | **+0.0752** |
| **Test Accuracy** | **91.92%** | **+9.34%** |
| **Precision** | **88.53%** | **+13.12%** |
| **Recall (Sensitivity)** | **96.32%** | −0.38% (High retention) |
| **False Positive Rate (FPR)** | **12.48%** | **−19.06 pp (Major drop)** |

#### Confusion Matrix (20,000 Samples)
```
                     Predicted REAL    Predicted SYNTHETIC
Actual REAL              8,752               1,248 (FP)
Actual SYNTHETIC           368 (FN)          9,632 (TP)
```

---

### 5. Baseline Comparison

| Iteration | Training Volume | ROC-AUC | Macro-F1 | Accuracy | FPR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (V1)** | 20,000 (subset) | 0.9316 | 0.8375 | 84.63% | 9.93% |
| **Iteration 2 (V2)** | 20,000 (tuned) | 0.9484 | 0.8474 | 82.58% | 31.54% |
| **Submission (V3)** | **97,000 (full)** | **0.9799** | **0.9226** | **91.92%** | **12.48%** |

---

### 6. Multi-Generator Roadmap & Operational Boundaries
1. **Multi-Generator Scope:** The evaluation framework is architected for cross-generator testing. During the hackathon timeframe, validation was conducted on the comprehensive 20,000-sample test benchmark; cross-generator benchmarking across multi-generator corpuses (FLUX, SD3) is slated as the immediate next research milestone.
2. **Resolution Parameter:** Training was conducted on 32×32 pixel images upscaled to 224×224; high-resolution RAW camera photo evaluation represents future pipeline expansion.
3. **Metadata Non-Penalty:** Stripped EXIF data is treated as neutral, preventing false accusations on social media uploads.
