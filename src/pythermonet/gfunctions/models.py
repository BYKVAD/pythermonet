from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class GFunctionRequest:
    # Primære inputs
    evaluation_times: np.ndarray               # [s]
    boreholes: Any                             # pygfunction borehole objects eller dine egne, se adapter
    thermal_diffusivity_soil: float            # [m²/s]
    # Konfiguration (reproducerbarhed)
    method: str = "claesson_javed"             # eller "equivalent", etc.
    boundary_condition: str = "UBWT"           # eller "UHTR" afhængigt af din anvendelse
    pygfunction_options: Optional[Dict[str, Any]] = None  # pygfunction options

@dataclass
class GFunctionSet:
    evaluation_times: np.ndarray
    g_values: np.ndarray
    metadata: Dict[str, Any]              # fx borehole count, spacing stats, method, commit hash
