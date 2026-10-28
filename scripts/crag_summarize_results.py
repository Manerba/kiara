#!/usr/bin/env python3
# ruff: noqa: I001
"""Summarize CRAG answer/retrieval metrics with metadata breakdowns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_summarize


if __name__ == "__main__":
    raise SystemExit(main_summarize())
