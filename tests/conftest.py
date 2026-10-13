"""
conftest.py — shared pytest configuration and path setup.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the local src/ is on the path so imports work without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pythermonet.resources import PIPE_CATALOG_PATH as PIPE_CATALOG  # noqa: E402

EXAMPLES_DIR = _REPO_ROOT / "examples"
TESTS_DATA_DIR = _REPO_ROOT / "tests" / "data"

# Fixed header values so formatted reports compare deterministically against
# tests/data/report_*_expected.txt.
REPORT_TIMESTAMP = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=1)))
REPORT_VERSION = "0.0.0-test"

RESULTS_JSON_SECTIONS = {"metadata", "ground_field", "distribution_network", "system"}


def iter_results_entries(results: dict) -> list[tuple[str, dict]]:
    """All (key, entry) pairs of a results dict, skipping the metadata section."""
    entries = []
    sections = [results["ground_field"], results["system"],
                results["distribution_network"]["performance"],
                results["distribution_network"]["traces"]]
    for section in sections:
        entries.extend(section.items())
    return entries
