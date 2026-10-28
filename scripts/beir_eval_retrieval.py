#!/usr/bin/env python3
"""Evaluate KIara retrieval on a prepared BEIR dataset."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiara.benchmarks import beir as beir_module

evaluate_beir_retrieval = beir_module.evaluate_beir_retrieval
main = beir_module.main
parse_k_values = beir_module.parse_k_values


if __name__ == "__main__":
    main()
