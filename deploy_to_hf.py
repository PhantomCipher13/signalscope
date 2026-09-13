"""
SignalScope — Deploy to Hugging Face Spaces (Gradio SDK, free tier)
====================================================================
Reads HF_TOKEN from environment / .env — never accepts it as a CLI arg.

Usage:
    python deploy_to_hf.py [username]

    username defaults to "Kingo101" if not provided.

Environment variables required (set in .env or shell, never commit):
    HF_TOKEN  — Hugging Face write token (hf_...)

This script:
1. Creates or verifies the Space Kingo101/signalscope (sdk=gradio, free)
2. Uploads all required files (code, model weights, frontend, configs)
3. Sets Space secrets (SUPABASE_URL, SUPABASE_KEY, ADMIN_TOKEN, etc.)
4. Prints the live URL

Security:
    - Token is read from env only, never printed or logged.
    - Supabase keys are set as Space secrets, never committed.
    - .env is never uploaded.
"""

import sys
import os
import io
from pathlib import Path

# ── Load .env for local credentials (HF_TOKEN, SUPABASE_*, ADMIN_TOKEN) ──────
ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in environment or .env")
    print("  Set it with: set HF_TOKEN=hf_... (Windows) or export HF_TOKEN=hf_... (bash)")
    sys.exit(1)

HF_USERNAME = sys.argv[1].strip() if len(sys.argv) > 1 else "Kingo101"
SPACE_NAME  = "signalscope"
REPO_ID     = f"{HF_USERNAME}/{SPACE_NAME}"

from huggingface_hub import HfApi

# Never print the token
api = HfApi(token=HF_TOKEN)

# ── Step 1: Create Space (Gradio SDK — free tier) ─────────────────────────────
print(f"\n[1/5] Creating/verifying Space: {REPO_ID} (sdk=gradio)")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    print(f"    Space ready: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"    create_repo error: {e}")
    print("    If Space already exists this may be harmless — continuing.")

# ── Step 2: Upload files ──────────────────────────────────────────────────────
print(f"\n[2/5] Uploading files to Space...")

UPLOAD_DIRS = [
    ("src",      "src"),
    ("configs",  "configs"),
    ("frontend", "frontend"),
    ("app",      "app"),
    ("outputs",  "outputs"),
]

UPLOAD_FILES = [
    ("models/fast_baseline_checkpoint.pt", "models/fast_baseline_checkpoint.pt"),
    ("hf_app.py",             "hf_app.py"),
    ("requirements_cloud.txt","requirements_cloud.txt"),
    ("README.md",             "README.md"),
    (".gitignore",            ".gitignore"),
]

IGNORE = ["__pycache__", "*.pyc", ".pytest_cache", "*.egg-info",
          ".env", "*.log", "test_large.jpg", "scratch"]

def upload_dir(local_rel, remote):
    local = ROOT / local_rel
    if not local.exists():
        print(f"    SKIP (missing): {local_rel}/")
        return
    print(f"    Dir  {local_rel}/ → {remote}/")
    api.upload_folder(
        folder_path=str(local),
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo=remote,
        ignore_patterns=IGNORE,
    )

def upload_file(local_rel, remote):
    local = ROOT / local_rel
    if not local.exists():
        print(f"    SKIP (missing): {local_rel}")
        return
    mb = local.stat().st_size / (1024 * 1024)
    print(f"    File {local_rel} ({mb:.1f} MB) → {remote}")
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=REPO_ID,
        repo_type="space",
    )

for local_rel, remote in UPLOAD_DIRS:
    try:
        upload_dir(local_rel, remote)
    except Exception as e:
        print(f"    ERROR {local_rel}: {e}")

for local_rel, remote in UPLOAD_FILES:
    try:
        upload_file(local_rel, remote)
    except Exception as e:
        print(f"    ERROR {local_rel}: {e}")

# ── Step 3: data/ placeholder ─────────────────────────────────────────────────
print("\n[3/5] Creating data/ directory placeholder...")
try:
    api.upload_file(
        path_or_fileobj=io.BytesIO(b"# SignalScope runtime data directory\n"),
        path_in_repo="data/.gitkeep",
        repo_id=REPO_ID,
        repo_type="space",
    )
    print("    data/.gitkeep uploaded")
except Exception as e:
    print(f"    WARNING: {e}")

# ── Step 4: Space secrets (values from .env, never printed) ──────────────────
print("\n[4/5] Setting Space secrets...")

SECRETS = {
    "SUPABASE_URL":           os.environ.get("SUPABASE_URL", ""),
    "SUPABASE_KEY":           os.environ.get("SUPABASE_KEY", ""),
    "ADMIN_TOKEN":            os.environ.get("ADMIN_TOKEN", ""),
    "SIGNALSCOPE_DEVICE":     "cpu",
    "SIGNALSCOPE_DB_BACKEND": "supabase",
}

for key, value in SECRETS.items():
    if value:
        try:
            api.add_space_secret(repo_id=REPO_ID, key=key, value=value)
            print(f"    Secret set: {key}")
        except Exception as e:
            print(f"    WARNING {key}: {e}")
    else:
        print(f"    SKIP (empty): {key} — set manually in Space settings if needed")

# ── Step 5: Summary ───────────────────────────────────────────────────────────
print("\n[5/5] Upload complete!")
print()
print("=" * 60)
print(f"  Space:     https://huggingface.co/spaces/{REPO_ID}")
print(f"  App URL:   https://{HF_USERNAME.lower()}-{SPACE_NAME}.hf.space")
print(f"  Logs:      https://huggingface.co/spaces/{REPO_ID}/logs")
print()
print("  The Space is now BUILDING (~5-10 min on first build).")
print("  Once live, the app runs 24/7 without your laptop.")
print("=" * 60)
