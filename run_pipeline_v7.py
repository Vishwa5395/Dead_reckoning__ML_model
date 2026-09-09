"""
run_pipeline_v7.py
------------------
Root-level entry point for PINO-DR v7 (Dual-Specialist Fine-Tuned Model).

Usage:
  python run_pipeline_v7.py
  python run_pipeline_v7.py --step split
  python run_pipeline_v7.py --step train
  python run_pipeline_v7.py --step eval
  python run_pipeline_v7.py --step report
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V7_DIR = ROOT / "v7_sept_model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V7_DIR) not in sys.path:
    sys.path.insert(0, str(V7_DIR))

from v7_sept_model.run_pipeline_v7 import main

if __name__ == "__main__":
    main()
