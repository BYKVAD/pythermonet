from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from pythermonet.core.annulus import Annulus


_REQUIRED_COLS = [
    "outer_diameter_mm",
    "standard_dimension_ratio",
]


def read_pipe_catalogue(path: str | Path) -> List[Annulus]:
    """
    Read pipe catalogue from CSV (semicolon-separated) and return a list
    of Annulus objects.

    Notes
    -----
    - Expects semicolon separator (';').
    - Converts outer diameter from mm to m.
    - Only geometric properties relevant for dimensioning are retained.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pipe catalogue not found: {p}")

    df = pd.read_csv(p, sep=";")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Pipe catalogue is missing required columns: {missing}"
        )

    catalogue: List[Annulus] = []

    for _, r in df.iterrows():
        catalogue.append(
            Annulus(
                outer_diameter=float(r["outer_diameter_mm"]) / 1000.0,
                sdr=float(r["standard_dimension_ratio"]),
            )
        )

    return catalogue
