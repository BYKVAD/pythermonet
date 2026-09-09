from __future__ import annotations

import importlib.resources as ir
from pathlib import Path

PIPE_CATALOG_PATH: Path = Path(ir.files(__package__) / "pipe_catalog.csv")
