#!/usr/bin/env python3
"""Run Sprint-5 generator/prompt replay audit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiara.benchmarks import generator_prompt as generator_prompt_module

collect_goldctx_bad_cases = generator_prompt_module.collect_goldctx_bad_cases
main = generator_prompt_module.main
run_generator_prompt_audit = generator_prompt_module.run_generator_prompt_audit


if __name__ == "__main__":
    main()
