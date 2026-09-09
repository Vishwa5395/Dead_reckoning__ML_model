"""
Root Entrypoint for PINO-DR v6 (Smartphone-Only Intelligent Dead Reckoning).
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from v6_smartphone_idr.run_pipeline_v6 import main

if __name__ == "__main__":
    main()
