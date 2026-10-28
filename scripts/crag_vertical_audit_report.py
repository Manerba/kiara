#!/usr/bin/env python3
# ruff: noqa: I001
"""Build a vertical CRAG audit report from existing audit artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_vertical_audit_report


if __name__ == "__main__":
    raise SystemExit(main_vertical_audit_report())
