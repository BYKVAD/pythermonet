from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pygfunction as gt

@dataclass(frozen=True)
class GFunctionResult:
    time_s: np.ndarray
    g: np.ndarray
    meta: Dict[str, Any]

def compute_gfunction_pygfunction(
    *,
    coords_xy_m: Sequence[Sequence[float]],
    time_s: Sequence[float],
    alpha_m2s: float,
    H_m: float,
    D_m: float,
    r_b_m: float,
    method: str = "equivalent",
    options: Optional[Dict[str, Any]] = None,
) -> GFunctionResult:
    xy = np.asarray(coords_xy_m, dtype=float)
    if xy.ndim != 2 or xy.shape[1] not in (2, 3):
        raise ValueError("coords_xy_m must be Nx2 or Nx3: [[x,y], ...] or [[x,y,z], ...].")

    t = np.asarray(time_s, dtype=float)
    if np.any(t <= 0):
        raise ValueError("time_s must be strictly positive.")
    if alpha_m2s <= 0:
        raise ValueError("alpha_m2s must be > 0.")
    if H_m <= 0 or r_b_m <= 0:
        raise ValueError("H_m and r_b_m must be > 0.")

    borefield = [
        gt.boreholes.Borehole(H=H_m, D=D_m, r_b=r_b_m, x=float(x), y=float(y))
        for x, y in xy[:, :2]
    ]

    options = options or {}
    gfunc = gt.gfunction.gFunction(
        borefield=borefield,
        alpha=alpha_m2s,
        time=t,
        method=method,
        options=options,
    )

    return GFunctionResult(
        time_s=t,
        g=np.asarray(gfunc.gFunc, dtype=float),
        meta={
            "n_boreholes": int(len(borefield)),
            "alpha_m2s": float(alpha_m2s),
            "H_m": float(H_m),
            "D_m": float(D_m),
            "r_b_m": float(r_b_m),
            "method": method,
            "options": options,
        },
    )

def compute_gfunction_infinite_medium(
    *,
    coords_xy_m: Sequence[Sequence[float]],
    time_s: Sequence[float],
    alpha_m2s: float,
    H_m: float,
    r_b_m: float,
    D_large_m: float = 100.0,
    method: str = "equivalent",
    options: Optional[Dict[str, Any]] = None,
) -> GFunctionResult:
    # Approximates infinite medium by pushing top depth far away.
    return compute_gfunction_pygfunction(
        coords_xy_m=coords_xy_m,
        time_s=time_s,
        alpha_m2s=alpha_m2s,
        H_m=H_m,
        D_m=D_large_m,
        r_b_m=r_b_m,
        method=method,
        options=options,
    )
