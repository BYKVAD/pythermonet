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

from pythermonet.resources import PIPE_CATALOG_PATH as PIPE_CATALOG  # noqa: E402

EXAMPLES_DIR = _REPO_ROOT / "examples"
