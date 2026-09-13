# SignalScope — Database Documentation

**Location:** `data/signalscope.db` (SQLite, never committed to git)

---

## 1. Why SQLite?

SignalScope uses SQLite as its initial database backend for these reasons:

| Property | SQLite | PostgreSQL/Supabase |
|---|---|---|
| Infrastructure required | None — file-based | Requires running server |
| SIH reproducibility | ✅ Clone and run | ❌ Requires server setup |
| Local development | ✅ Zero config | ❌ Installation required |
| Python stdlib | ✅ (`sqlite3` built-in) | ❌ (`psycopg2` install) |
| Migration path | ✅ via abstraction layer | — |
| Concurrent writes | Limited (WAL mode used) | Full concurrent support |

For a hackathon prototype and SIH demo, SQLite is the correct choice. The abstraction layer in `src/database/` means the backend can be replaced with PostgreSQL without rewriting any application logic.

---

## 2. What the Database Stores

| Category | Stored | Example values |
|---|---|---|
| Experiment configuration | ✅ | architecture, seed, LR, batch_size, manifest paths |
| Experiment metrics | ✅ | roc_auc=0.87, accuracy=0.85, split=validation |
| Checkpoint paths | ✅ | `models/best_checkpoint.pt` (path reference only) |
| Analysis results | ✅ | image_hash, synthetic_probability, prediction |
| Reliability probe results | ✅ (Phase 2+) | transformation_name, probability |
| Provenance metadata | ✅ (Phase 2+) | c2pa_status, exif_status |
| Generator / evaluation identity | ✅ | generator=stable_diffusion_v1.4, evaluation_type=in_distribution |

---

## 3. What the Database Does NOT Store

> **NEVER stored in the database:**

| Prohibited content | Why |
|---|---|
| **Image binary data** | Images can be 500 KB–5 MB each; storing them would make the DB unmanageable |
| **Model weights** | Checkpoints are large binary blobs; stored in `models/` via filesystem |
| **Fabricated metrics** | Calibration, uncertainty, reliability are `NULL` until those modules exist |
| **Unseen-generator results from CIFAKE** | CIFAKE has only SD v1.4 — calling it "unseen-generator" would be scientifically dishonest |
| **Raw dataset images** | The `data/raw/` directory is ignored by git; images never enter the database |

Image identity is tracked via **SHA-256 hash** (`image_hash` column in `analysis_runs`). The same image can be matched across repeated analyses without storing any binary data.

---

## 4. Schema Overview

```
schema_version          (version tracking for future migrations)
│
├── experiments         (one row per training run)
│     └── experiment_metrics   (many metrics per experiment, per split)
│
└── analysis_runs       (one row per image inference)
      ├── probe_results        (many reliability probes per analysis)
      └── provenance_records   (one C2PA/EXIF record per analysis)
```

### Table: `experiments`

Stores one record per training run. Model weights are **not** stored — only a reference path.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `experiment_name` | TEXT | Human-readable name (e.g. `cifake_baseline_eb3_e10`) |
| `dataset` | TEXT | e.g. `CIFAKE` |
| `model_name` | TEXT | e.g. `efficientnet_b3` |
| `seed` | INTEGER | Random seed for reproducibility |
| `train_manifest` | TEXT | Path to training split CSV |
| `val_manifest` | TEXT | Path to validation split CSV |
| `test_manifest` | TEXT | Path to test split CSV |
| `epochs_trained` | INTEGER | |
| `best_epoch` | INTEGER | Epoch with best validation metric |
| `batch_size` | INTEGER | |
| `learning_rate` | REAL | |
| `weight_decay` | REAL | |
| `scheduler` | TEXT | e.g. `cosine` |
| `checkpoint_path` | TEXT | Path to `.pt` file (**never** binary) |
| `generator` | TEXT | e.g. `stable_diffusion_v1.4` |
| `augmentation_config` | TEXT | JSON string of augmentation settings |
| `training_duration_s` | REAL | |
| `notes` | TEXT | Limitations, methodology notes |
| `created_at` | TEXT | ISO 8601 timestamp |

### Table: `experiment_metrics`

One row per metric per split per experiment.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `experiment_id` | INTEGER FK | References `experiments(id)` |
| `split_name` | TEXT | e.g. `validation`, `test_indistribution` |
| `evaluation_type` | TEXT | **`in_distribution`** or `unseen_generator` |
| `metric_name` | TEXT | e.g. `roc_auc`, `accuracy`, `f1` |
| `metric_value` | REAL | |

> ⚠️ **Scientific integrity enforced:** CIFAKE test split must always be recorded with `evaluation_type = 'in_distribution'`. The database layer raises a `ValueError` if an invalid type is specified.

### Table: `analysis_runs`

One record per image inference request. Never stores image binary.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `image_hash` | TEXT | **SHA-256** of input image (64-char hex) |
| `image_path` | TEXT | Optional local path (NOT binary data) |
| `model_version` | TEXT | |
| `experiment_id` | INTEGER FK | Optional — which experiment produced this model |
| `baseline_probability` | REAL | Raw model synthetic probability (uncalibrated) |
| `prediction` | TEXT | `real` or `synthetic` |
| `predicted_class` | INTEGER | `0` = real, `1` = synthetic |
| `confidence` | REAL | Same as probability until calibration is done |
| `calibration_status` | TEXT | `not_calibrated` (Phase 1) / `calibrated` (Phase 2+) |
| `uncertainty_status` | TEXT | `not_computed` (Phase 1) / `computed` (Phase 2+) |
| `stability` | REAL | NULL until reliability engine is active |
| `probes_run` | INTEGER | 0 until adaptive probing is implemented |
| `stop_reason` | TEXT | NULL until adaptive probing |
| `processing_time_ms` | REAL | |

### Table: `probe_results`

Populated by the reliability engine (Phase 2+). Empty until then.

| Column | Notes |
|---|---|
| `analysis_id` | FK → `analysis_runs(id)` CASCADE DELETE |
| `transformation_name` | e.g. `jpeg_compression`, `gaussian_noise` |
| `transformation_params` | JSON string |
| `probability` | Synthetic probability after transformation |
| `success` | 0/1 |
| `error_message` | If probe failed |

### Table: `provenance_records`

Populated by the provenance module (Phase 2+). Empty until then.

| Column | Notes |
|---|---|
| `analysis_id` | FK → `analysis_runs(id)` CASCADE DELETE |
| `metadata_available` | 0=no EXIF/C2PA found, 1=found |
| `c2pa_status` | `not_checked`, `not_found`, `found_valid`, `found_tampered` |
| `exif_status` | `not_checked`, `not_found`, `found` |
| `exif_fields` | JSON string of relevant EXIF fields |

---

## 5. How Database Initialisation Works

The database is initialised automatically when first used:

```python
from src.database import get_db, init_db

db = get_db()   # creates data/signalscope.db if it doesn't exist
db.init()       # runs CREATE TABLE IF NOT EXISTS for all 5 tables
```

`init_db()` is **idempotent** — safe to call multiple times. It uses `CREATE TABLE IF NOT EXISTS` and only inserts the schema version row once.

The path is read from `configs/default.yaml`:
```yaml
database:
  path: "data/signalscope.db"
```

In tests, always use a temporary path:
```python
db = DatabaseManager(tmp_path / "test.db")
```

**Never use the production DB in tests.**

---

## 6. How Experiments Are Recorded

Training automatically records to the database unless `--no-db` is passed:

```powershell
python scripts/train.py --manifest data/manifests/train.csv \
                        --val-manifest data/manifests/validation.csv \
                        --epochs 10 --experiment-name "my_experiment"
```

After training completes, the experiment and validation metrics are inserted. The returned `experiment_id` can be used with `evaluate.py`:

```powershell
python scripts/evaluate.py --checkpoint models/best_checkpoint.pt \
                            --manifest data/manifests/test_indistribution.csv \
                            --split-name test_indistribution \
                            --evaluation-type in_distribution \
                            --experiment-id 1
```

---

## 7. How to Query the Database

```python
from src.database import get_db
from src.database.repositories import list_experiments, get_metrics

db = get_db()

# List all experiments
for exp in list_experiments(db):
    print(exp["id"], exp["experiment_name"], exp["created_at"])

# Get metrics for experiment 1
metrics = get_metrics(db, experiment_id=1, split_name="validation")
for m in metrics:
    print(m["metric_name"], m["metric_value"], m["evaluation_type"])
```

---

## 8. Future PostgreSQL / Supabase Migration

The `DatabaseManager` class in `src/database/connection.py` wraps all SQL execution. To switch to PostgreSQL:

1. Add `psycopg2` (or `psycopg3`) to `requirements.txt`
2. Update `DatabaseManager.connect()` to use `psycopg2.connect(dsn)` instead of `sqlite3.connect(path)`
3. Replace SQLite-specific syntax: `datetime('now')` → `NOW()`, `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY`, `?` placeholders → `%s`
4. Update `configs/default.yaml`: `backend: "postgresql"`
5. Add a `DATABASE_URL` environment variable for the connection string

All repository functions (`repositories.py`) remain unchanged — they only call `db.execute()`, `db.fetchone()`, etc., which are backend-agnostic.

---

## 9. Scientific Integrity Guarantees

The database layer enforces several scientific integrity rules at write time:

| Rule | Enforcement |
|---|---|
| Image binary data never stored | `insert_analysis()` accepts only `image_hash` (SHA-256 string), not binary |
| CIFAKE test = in_distribution | `insert_metric()` validates `evaluation_type` against allowed values; 'unseen_generator' must never be used for CIFAKE |
| Calibration status honest | `calibration_status` must be `'not_calibrated'` or `'calibrated'` — no other values accepted |
| Uncertainty status honest | `uncertainty_status` must be `'not_computed'` or `'computed'` |
| No fabricated confidence | `stability`, `stop_reason`, `probes_run` default to NULL/0 and are only filled by real implementation |

---

## 10. Git Hygiene

The `.gitignore` contains:

```gitignore
# SQLite database — never commit
data/signalscope.db
data/*.db
data/*.db-wal
data/*.db-shm
*.db
*.sqlite
*.sqlite3
```

The schema and all Python source files **are** committed. The database file itself is not.
