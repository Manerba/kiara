#!/usr/bin/env python3
# ruff: noqa: I001
"""Run KIara RAG against a prepared CRAG Task 3 manifest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_run


if __name__ == "__main__":
    raise SystemExit(main_run(3))
