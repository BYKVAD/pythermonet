from __future__ import annotations

import importlib.resources as ir
from pathlib import Path

PIPE_CATALOG_PATH: Path = Path(ir.files(__package__) / "pipe_catalog.csv")
SETTINGS_TEMPLATE_BHE_PATH: Path = Path(
    ir.files(__package__) / "settings_template_bhe.json"
)
SETTINGS_TEMPLATE_HHE_PATH: Path = Path(
    ir.files(__package__) / "settings_template_hhe.json"
)
