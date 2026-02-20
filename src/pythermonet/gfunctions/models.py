from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class GFunctionRequest:
    # Primære inputs
    times_s: np.ndarray               # sekunder (float64)
    boreholes: Any                    # pygfunction borehole objects eller dine egne, se adapter
    alpha_m2_s: float                 # soil thermal diffusivity

    # Konfiguration (reproducerbarhed)
    method: str = "claesson_javed"    # eller "equivalent", etc.
    boundary_condition: str = "UBWT"  # eller "UHTR" afhængigt af din anvendelse
    options: Optional[Dict[str, Any]] = None  # pygfunction options

@dataclass
class GFunctionSet:
    times_s: np.ndarray
    g_values: np.ndarray
    meta: Dict[str, Any]              # fx borehole count, spacing stats, method, commit hash
