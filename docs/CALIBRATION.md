# SignalScope — Calibration Documentation

**Module:** `src/calibration/`  
**Script:** `scripts/calibrate.py`  
**Phase:** 3

---

## 1. Why Calibration Exists

A neural network classifier can be **accurate** yet **poorly calibrated**. A model that outputs probability 0.95 should be correct ~95% of the time across all samples where it outputs 0.95. If the model is overconfident (e.g., always outputs 0.95 regardless of true correctness), the raw probability is **not a reliable uncertainty estimate**.

SignalScope returns calibrated probabilities because:

- Users must be able to interpret the probability as a meaningful likelihood, not just a ranking score.
- Downstream decisions (abstention, reliability classification) depend on the probability being trustworthy.
- Scientific integrity requires acknowledging when probabilities are not calibrated.

---

## 2. Temperature Scaling

SignalScope uses **post-hoc temperature scaling** — the simplest and most robust calibration technique for neural networks [Guo et al., 2017].

### How it works

Temperature scaling adds a single learned scalar $T > 0$ that divides the logit before the softmax:

$$
\hat{p}_{\text{calibrated}} = \sigma\!\left(\frac{z}{T}\right)
$$

where $z$ is the raw logit and $\sigma$ is the sigmoid function.

- $T = 1.0$: no change (identity)
- $T > 1.0$: softens predictions toward 0.5 (reduces overconfidence)
- $T < 1.0$: sharpens predictions away from 0.5 (rarely beneficial)

$T$ is a scalar — it does **not** change the model's predictions (argmax is unchanged) — it only affects the probability values.

### Fitting procedure

Temperature $T$ is fitted by minimising the **negative log-likelihood (NLL)** on the validation set:

$$
T^* = \arg\min_T \; -\sum_{i=1}^{N} \left[ y_i \log \sigma\!\left(\frac{z_i}{T}\right) + (1-y_i) \log\!\left(1 - \sigma\!\left(\frac{z_i}{T}\right)\right) \right]
$$

Optimisation uses `scipy.optimize.minimize_scalar` with bounds $T \in [0.1, 5.0]$. If scipy is unavailable, a grid search over 50 points in $[0.1, 5.0]$ is used as fallback (status = `calibration_fallback`).

### Fitting rules (ENFORCED)

> [!IMPORTANT]
> Temperature **must only be fitted on the validation set**.
> 
> - **Never fit on the test set.** The test set is for final one-shot evaluation only.
> - **Never fit on the unseen-generator set.** CIFAKE has no unseen-generator split.
> - Once fitted, $T$ is **frozen** before any test-set evaluation.

The `calibrate.py` script enforces this: if the predictions file has `split='test'`, the script aborts with an error.

---

## 3. Calibration Status

Every probability-producing result in SignalScope carries an **explicit calibration status**. The system never silently treats an uncalibrated probability as calibrated.

| Status | Meaning |
|---|---|
| `not_calibrated` | Model outputs raw softmax probability. No temperature scaling applied. |
| `calibrated` | Temperature scaling was successfully fitted and applied. |
| `calibration_fallback` | Grid-search fallback used (scipy unavailable). Applied but less precise. |
| `calibration_unavailable` | Fitting failed or insufficient data. Raw probability returned unchanged. |

**Rule:** If you see `calibrated_probability=None` and `calibration_status=not_calibrated`, the system is correctly reporting that calibration has not been performed. This is honest, not a bug.

---

## 4. Expected Calibration Error (ECE)

ECE measures how well the model's confidence aligns with its actual accuracy, aggregated across probability bins.

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where:
- $M$ is the number of bins (default 15, configurable via `ece_n_bins` in `reliability.yaml`)
- $B_m$ is the set of samples in bin $m$
- $\text{acc}(B_m)$ is the fraction of correct predictions in that bin
- $\text{conf}(B_m)$ is the mean predicted probability in that bin
- $N$ is total samples

**ECE is in $[0, 1]$. Lower is better. Perfect calibration → ECE = 0.**

Empty bins are skipped (not penalised). This is standard practice to avoid edge effects.

### ECE after temperature scaling

The calibration script reports ECE before and after scaling. A well-calibrated model should show a reduction in ECE after scaling.

> [!WARNING]
> **ECE thresholds are NOT configured yet.**
> 
> `reliability.yaml` has `ece_threshold: null`.
> The system will NOT classify a model as "well-calibrated" or "poorly calibrated" based on ECE alone until validation experiments establish a meaningful threshold.
> ECE reduction after scaling is informative but not a pass/fail criterion at this stage.

---

## 5. Brier Score

The Brier score is the mean squared error between predicted probabilities and true labels:

$$
\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

- Range: $[0, 1]$. Lower is better.
- Random classifier (p=0.5 on balanced data) → Brier = 0.25.
- Perfect predictor → Brier = 0.0.
- Worst predictor (always wrong) → Brier = 1.0.

Brier score is a proper scoring rule that jointly penalises both calibration and resolution (sharpness).

> [!NOTE]
> `brier_threshold: null` in `reliability.yaml`. Like ECE, thresholds will be set from validation evidence only.

---

## 6. How to Run Calibration

### Step 1: Train and collect validation predictions

Training automatically saves `outputs/val_predictions.npz` when `--save-predictions` is used with `evaluate.py`, or during training when validation is enabled.

### Step 2: Fit temperature

```powershell
C:\Python311\python.exe scripts/calibrate.py `
    --predictions outputs/val_predictions.npz `
    --experiment-id 1
```

Output:
```
=== SignalScope — Temperature Scaling Calibration ===
  Fitting temperature on VALIDATION predictions only.

  Loaded 15000 samples from: outputs/val_predictions.npz
  Split label in file: 'validation'

  Calibration status: calibrated
  Temperature (T):    1.234567

  ECE before:         0.082341
  ECE after:          0.041209
  Brier before:       0.115320
  Brier after:        0.088740

  ECE change:         -0.041132 (improved)

  REMINDER: These are measured results, NOT performance targets.
```

### Step 3: Use calibrated probabilities

```python
from src.calibration import TemperatureScaler

scaler = TemperatureScaler.load("outputs/calibration_exp1.json")
calibrated_probs, status = scaler.transform_with_status(raw_probs)
# status == CalibrationStatus.CALIBRATED
```

---

## 7. Calibration File Format

Saved as JSON (not binary):

```json
{
  "temperature": 1.234567,
  "is_fitted": true,
  "_fitted_on_split": "validation",
  "calibration_result": {
    "temperature": 1.234567,
    "status": "calibrated",
    "fitted_on": "validation",
    "n_samples_fitted": 15000,
    "ece_before": 0.082341,
    "ece_after": 0.041209,
    "brier_before": 0.115320,
    "brier_after": 0.088740
  }
}
```

---

## 8. Database Persistence

Calibration records are stored in the `calibration_records` table:

```sql
SELECT temperature, calibration_status, ece_before, ece_after
FROM calibration_records
WHERE experiment_id = 1;
```

**Scientific integrity enforced at DB layer:**
- `temperature=NULL` is the default (not fabricated).
- `fitted_on_split='test'` is REJECTED with a `ValueError`.
- ECE and Brier values are NULL until actually computed.

---

## 9. Integration with Analysis Pipeline

```
Image Input
    ↓
Detector (EfficientNet-B3)
    ↓ raw logits
TemperatureScaler.transform()
    ↓ calibrated_probability + CalibrationStatus
ReliabilityEngine
    ↓ ReliabilityResult
API Response (AnalysisResponse)
```

When the model is not yet calibrated, `calibrated_probability=None` and `calibration_status='not_calibrated'` are returned explicitly — the raw `baseline_probability` is still available.

---

## 10. Current Limitations

| Limitation | Status |
|---|---|
| Calibration not yet fitted | Awaiting real trained model output |
| ECE/Brier thresholds not set | Require validation experiments |
| Only scalar temperature scaling | Multi-class calibration (e.g. Platt, isotonic) deferred to later phase |
| Single-class datasets affect ECE | CIFAKE is balanced — this is acceptable |
| 32×32 upscaled images | CIFAKE limitation — probabilities reflect upscaled statistics |

---

## 11. References

- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.*
- Brier, G.W. (1950). *Verification of forecasts expressed in terms of probability.*
