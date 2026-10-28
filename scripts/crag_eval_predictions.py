#!/usr/bin/env python3
# ruff: noqa: I001
"""Lightweight local evaluation for KIara CRAG predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_eval


if __name__ == "__main__":
    raise SystemExit(main_eval())
