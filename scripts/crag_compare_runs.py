#!/usr/bin/env python3
# ruff: noqa: I001
"""Compare CRAG summary JSON files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_compare


if __name__ == "__main__":
    raise SystemExit(main_compare())
