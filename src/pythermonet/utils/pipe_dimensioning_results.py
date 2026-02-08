# pythermonet/components/pipe_dimensioning_result.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PipeDimensioningResult:
    doCooling: bool
    Q_PG_H: np.ndarray                 # m3/s per pipe group (per trace)
    Q_PG_C: np.ndarray | None          # m3/s per pipe group (per trace)
