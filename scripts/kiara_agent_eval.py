#!/usr/bin/env python3
"""Dokumente ziehen, Referenzen einfrieren und den echten KIara-Agenten messen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.agent_retrieval import main

if __name__ == "__main__":
    main()
