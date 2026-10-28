#!/usr/bin/env python3
# ruff: noqa: I001
"""Export KIara CRAG predictions as official local-evaluator inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import main_export_official


if __name__ == "__main__":
    raise SystemExit(main_export_official())
