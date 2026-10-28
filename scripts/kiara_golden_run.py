#!/usr/bin/env python3
"""Run KIara's curated Golden Set benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiara.benchmarks import golden as golden_module

evaluate_golden_set = golden_module.evaluate_golden_set
main = golden_module.main
parse_k_values = golden_module.parse_k_values


if __name__ == "__main__":
    main()
