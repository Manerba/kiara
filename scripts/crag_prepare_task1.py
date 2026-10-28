#!/usr/bin/env python3
# ruff: noqa: I001
"""Prepare CRAG Task 1 as KIara-indexable documents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_prepare


if __name__ == "__main__":
    raise SystemExit(main_prepare(1))
