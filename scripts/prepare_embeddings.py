#!/usr/bin/env python3
"""Thin launcher for the production ESM-C embedding workflow."""

from __future__ import annotations

import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "proofccc_core"
sys.path.insert(0, str(CORE))

from prepare_dataset_embeddings import main  # noqa: E402


if __name__ == "__main__":
    main()
