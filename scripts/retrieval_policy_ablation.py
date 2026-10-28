#!/usr/bin/env python3
"""Run Sprint 4 retrieval-policy ablations."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiara.benchmarks import retrieval_policy as retrieval_policy_module

main = retrieval_policy_module.main
run_retrieval_policy_matrix = retrieval_policy_module.run_retrieval_policy_matrix


if __name__ == "__main__":
    main()
