#!/usr/bin/env python3
# ruff: noqa: I001
"""Evaluate KIara CRAG predictions with a configurable CRAG-rubric LLM judge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_eval_official_judge


if __name__ == "__main__":
    raise SystemExit(main_eval_official_judge())
