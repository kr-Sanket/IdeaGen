#!/usr/bin/env python3
"""CLI Entrypoint for IdeaGen Controlled Multi-Paper Experiment Runner."""

import sys
from pathlib import Path

# Add project root and backend to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from experiments.runner import main

if __name__ == "__main__":
    main()
