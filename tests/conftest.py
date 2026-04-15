"""
conftest.py — shared pytest configuration and path setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the local src/ is on the path so imports work without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Shared resource path used by multiple test modules.
PIPE_CATALOGUE = _REPO_ROOT / "src" / "pythermonet" / "resources" / "pipe_catalogue.csv"
EXAMPLES_DIR   = _REPO_ROOT / "examples"
