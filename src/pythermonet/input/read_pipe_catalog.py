from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from pythermonet.core.annulus import Annulus
from pythermonet.resources import PIPE_CATALOG_PATH


_REQUIRED_COLS = [
    "outer_diameter_mm",
    "standard_dimension_ratio",
]


def read_pipe_catalog(path: str | Path | None = None) -> List[Annulus]:
    """
    Read pipe catalog from CSV (semicolon-separated) and return a list
    of Annulus objects.

    Parameters
    ----------
    path : str | Path | None
        Path to a pipe catalog CSV. If None, reads the catalog bundled
        with this package.

    Notes
    -----
    - Expects semicolon separator (';').
    - Converts outer diameter from mm to m.
    - Only geometric properties relevant for dimensioning are retained.
    """
    p = Path(path) if path is not None else PIPE_CATALOG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Pipe catalog not found: {p}")

    df = pd.read_csv(p, sep=";")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Pipe catalog is missing required columns: {missing}"
        )

    catalog: List[Annulus] = []

    for _, r in df.iterrows():
        catalog.append(
            Annulus(
                diameter_outer=float(r["outer_diameter_mm"]) / 1000.0,
                sdr=float(r["standard_dimension_ratio"]),
            )
        )

    return catalog
