#!/usr/bin/env python3
"""Launch the final 20-repeat, five-fold PROOF-CCC training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "proofccc_core"
sys.path.insert(0, str(CORE))

from run_cv_fixed_deep_ftensemble import main  # noqa: E402


if __name__ == "__main__":
    main()
