from __future__ import annotations
from typing import List
import pandas as pd

from pythermonet.core.annulus import Annulus


_REQUIRED_COLS = [
    "outer_diameter_mm",
    "standard_dimension_ratio",
]


def read_pipe_catalogue(df: pd.DataFrame) -> List[Annulus]:
    """
    Converts a pd.Dataframe to a List of Annulus. 

    Notes
    -----
    - Converts outer diameter from mm to m.
    - Only geometric properties relevant for dimensioning are retained.
    """
    
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
                outerDiameter=float(r["outer_diameter_mm"]) / 1000.0,
                SDR=float(r["standard_dimension_ratio"]),
            )
        )

    return catalogue