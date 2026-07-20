#!/usr/bin/env python3
"""Launch final-model prediction on unannotated ligand-receptor candidates."""

from __future__ import annotations

import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "proofccc_core"
sys.path.insert(0, str(CORE))

from run_high_confidence_prediction import main  # noqa: E402


if __name__ == "__main__":
    main()
