#!/usr/bin/env python3
# ruff: noqa: I001
"""Analyze CRAG preparation and ingestion audit cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_analyze_audits


if __name__ == "__main__":
    raise SystemExit(main_analyze_audits())
