"""
run_pipeline_v4.py
------------------
Root-level wrapper for PINO-DR v4 (Turn-Focused).
Delegates execution to v4_turn_focused/run_pipeline_v4.py.

Usage:
  python run_pipeline_v4.py
  python run_pipeline_v4.py --step preprocess
  python run_pipeline_v4.py --step train
  python run_pipeline_v4.py --step eval
  python run_pipeline_v4.py --step report
  python run_pipeline_v4.py --ablation [A|B|C|D|E]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V4_DIR = ROOT / "v4_turn_focused"

if __name__ == "__main__":
    script = V4_DIR / "run_pipeline_v4.py"
    cmd = [sys.executable, str(script)] + sys.argv[1:]
    res = subprocess.run(cmd, cwd=str(V4_DIR))
    sys.exit(res.returncode)
