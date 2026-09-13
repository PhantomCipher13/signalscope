# SignalScope — Reliability Engine Documentation

**Module:** `src/reliability/`  
**Phase:** 3

---

## 1. What the Reliability Engine Does

The SignalScope reliability engine addresses a core problem in AI detection: **a single model prediction, no matter how confident, cannot self-verify its own consistency**.

A sophisticated forgery might fool the model reliably. A borderline real image might get inconsistent scores under slight transformations. The reliability engine measures this consistency directly.

For a given input image, the engine:

1. Observes the baseline model probability (always available).
2. When probe results are also available (Phase 4+), collects probabilities across controlled image transformations.
3. Aggregates all observations to compute mean, standard deviation, and a stability score.
4. Classifies the overall reliability of the prediction.

---

## 2. Observation Count and Insufficient Evidence

> [!IMPORTANT]
> **One observation is not enough to establish transformation stability.**

When only the baseline prediction is available (`observation_count=1`):

```
stability         = None
stability_status  = insufficient_evidence
reliability_status = insufficient_evidence
note = "Only one observation. Cannot assess transformation stability."
```

**Do NOT interpret `stability=None` as stable.** It means the engine has insufficient data to make any stability claim.

This is distinct from a model outputting `std=0` for a single value. Mathematically, $\text{std}([x]) = 0$ is true, but it carries no information about how the system behaves under perturbation. The engine explicitly rejects this interpretation.

---

## 3. Stability Computation

When multiple observations are available ($n \geq 2$):

### Standard deviation

$$
\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(p_i - \bar{p})^2}
$$

This is the **sample standard deviation** (ddof=1), appropriate when treating observations as a sample from a broader distribution of possible transformations.

> [!NOTE]
> Population std (ddof=0) would underestimate variance for small samples. Using sample std is more conservative and scientifically appropriate.

### Stability score

$$
S = \text{clip}(1 - \sigma, \; 0, \; 1)
$$

- $S$ near 1.0 → predictions are very consistent across observations (potentially stable).
- $S$ near 0.0 → predictions vary widely (potentially unstable).

> [!WARNING]
> $S = 1 - \sigma$ is a **simple heuristic**, not a formally validated stability measure. It provides an intuitive summary but should not be treated as a rigorous scientific quantity without validation. Future phases may replace or augment this with more principled uncertainty estimates.

### When stability is None

| Condition | stability | stability_status |
|---|---|---|
| `observation_count == 1` | `None` | `insufficient_evidence` |
| `observation_count < minimum_observations` (configured) | `None` | `insufficient_evidence` |
| `minimum_observations` not configured | `None` | `unavailable` |
| `observation_count >= minimum_observations` | `float in [0,1]` | `computed` |

---

## 4. Reliability Status

The engine produces one of five reliability statuses:

| Status | Meaning |
|---|---|
| `insufficient_evidence` | Too few observations to assess stability. |
| `stable` | $\sigma < \text{stability\_threshold}$ AND $n \geq \text{minimum\_observations}$. |
| `unstable` | $\sigma \geq \text{stability\_threshold}$ AND $n \geq \text{minimum\_observations}$. |
| `unavailable` | Engine could not produce a result. |
| `threshold_not_configured` | Thresholds have not been set from validation data yet. |

---

## 5. Threshold Policy

> [!CAUTION]
> **All reliability thresholds are currently `null` (not configured).**

In `configs/reliability.yaml`:

```yaml
reliability:
  minimum_observations: null    # e.g. 5 — set after validation
  stability_threshold: null     # e.g. 0.05 — set after validation
```

Until validation experiments determine appropriate thresholds:

- `reliability_status` will be `threshold_not_configured` when probes are run.
- `stability` will be `None` for single observations (regardless of thresholds).
- No abstention decisions are made.

**Why?** Choosing arbitrary thresholds (e.g. `std < 0.10`) without validation evidence would result in miscalibrated reliability claims. The system is designed to be scientifically honest about what it doesn't know.

---

## 6. Reliability Result Structure

```python
@dataclass
class ReliabilityResult:
    baseline_probability: float        # Raw model output (always present)
    calibrated_probability: float | None  # None if calibration unavailable
    calibration_status: str            # Explicit: never silently uncalibrated
    observation_count: int             # 1 = baseline only
    probabilities: list[float]         # All observed values
    mean_probability: float | None     # None for empty observation set
    standard_deviation: float | None   # None when n < 2
    stability: float | None            # None when insufficient evidence
    stability_status: StabilityStatus
    reliability_status: ReliabilityStatus
    reliability_note: str              # Human-readable explanation
    threshold_configured: bool
```

The result serialises directly to the API `ReliabilitySchema`:

```json
{
  "reliability": {
    "observation_count": 1,
    "mean_probability": 0.82,
    "standard_deviation": null,
    "stability": null,
    "stability_status": "insufficient_evidence",
    "status": "insufficient_evidence",
    "note": "Only one observation. Cannot assess transformation stability.",
    "threshold_configured": false
  }
}
```

---

## 7. Engine API

### Single observation (baseline only)

```python
from src.reliability import ReliabilityEngine

engine = ReliabilityEngine()  # no config = thresholds unconfigured

result = engine.analyze_single(
    baseline_probability=0.85,
    calibration_status="calibrated",
    calibrated_probability=0.79,
)

assert result.stability is None          # insufficient evidence
assert result.observation_count == 1
assert result.reliability_status.value == "insufficient_evidence"
```

### Multiple observations (with probes)

```python
result = engine.analyze_probes(
    baseline_probability=0.85,
    probe_probabilities=[0.87, 0.84, 0.86, 0.83],
    calibration_status="calibrated",
    calibrated_probability=0.79,
)

# With unconfigured thresholds:
assert result.reliability_status.value == "threshold_not_configured"
assert result.standard_deviation is not None   # computed from 5 observations
```

### With configured thresholds (post-validation)

```python
engine = ReliabilityEngine(config={
    "minimum_observations": 3,
    "stability_threshold": 0.05,
})

result = engine.analyze_probes(0.85, [0.87, 0.84, 0.86, 0.83])
# If std < 0.05 and n >= 3:
#   result.reliability_status == ReliabilityStatus.STABLE
```

---

## 8. Database Persistence

Reliability results are stored as columns on `analysis_runs`:

| Column | Type | Notes |
|---|---|---|
| `calibrated_probability` | REAL | NULL if not calibrated |
| `reliability_status` | TEXT | e.g. `insufficient_evidence` |
| `reliability_note` | TEXT | Human-readable explanation |
| `observation_count` | INTEGER | Default 0 |
| `mean_probability` | REAL | NULL until computed |
| `std_probability` | REAL | NULL until computed |
| `stability` | REAL | NULL when insufficient evidence — NOT 0 or 1 |

Updating an analysis with reliability results:

```python
from src.database.repositories import update_analysis_reliability

update_analysis_reliability(
    db, analysis_id=1,
    reliability_status="insufficient_evidence",
    observation_count=1,
    # stability intentionally omitted → stays NULL in DB
)
```

---

## 9. Anti-Patterns (Forbidden by Design)

The following are explicitly prevented by the engine:

| Anti-pattern | Why forbidden |
|---|---|
| `stability=1.0` for single observation | `std([x])=0` proves nothing about consistency |
| Treating `stability=None` as stable | None means unknown, not good |
| Hard-coding `stability_threshold=0.1` | Value is arbitrary without validation evidence |
| Using test-set probes for threshold tuning | Test data must remain held-out |
| Claiming `reliable` without minimum_observations | Statistical noise dominates small samples |

---

## 10. Integration with Future Phases

```
Phase 3 (current):
  baseline_probability
      ↓
  TemperatureScaler → calibrated_probability
      ↓
  ReliabilityEngine.analyze_single() → observation_count=1, stability=None

Phase 4 (Adaptive Stress Testing):
  baseline + probe_probabilities[n]
      ↓
  ReliabilityEngine.analyze_probes() → stability computed when n >= minimum_observations

Phase 5 (Provenance):
  ReliabilityResult + C2PA/EXIF metadata → holistic confidence assessment
```

The reliability engine is decoupled from the specific model architecture (EfficientNet-B3). It accepts any sequence of probability floats.

---

## 11. Configuration Reference

All reliability parameters live in [`configs/reliability.yaml`](../configs/reliability.yaml):

```yaml
reliability:
  n_versions: 10                 # probes per image (Phase 4+)
  minimum_observations: null     # set from validation
  stability_threshold: null      # set from validation

  abstention:
    confidence_margin: null      # set from validation
    stability_threshold: null    # set from validation

  adaptive:
    enabled: true
    max_probes: 15
    early_stop_margin: null      # set from validation
    fallback_to_fixed: true

calibration:
  ece_n_bins: 15
  ece_threshold: null
  brier_threshold: null
```

---

## 12. Known Limitations

| Limitation | Notes |
|---|---|
| All thresholds unconfigured | Awaiting validation experiments post-training |
| Phase 4 probes not yet implemented | Engine accepts probe_probabilities but probes not generated |
| `S = 1 - std` is a heuristic | Validated alternatives (MC Dropout, Deep Ensembles) deferred |
| CIFAKE images are 32×32 | Probe transformation stats may differ from natural HD images |
| Single-class evaluation | CIFAKE test cannot establish unseen-generator reliability |
