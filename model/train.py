#!/usr/bin/env python3
"""
SignalScope — Model Training Wrapper (Section 7.1)
Wraps scripts/train_v3.py for reproducing the final EfficientNet-B0 V3 model.
"""
import sys
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

def main():
    script = _ROOT / "scripts" / "train_v3.py"
    args = [sys.executable, str(script)] + sys.argv[1:]
    sys.exit(subprocess.run(args).returncode)

if __name__ == "__main__":
    main()
