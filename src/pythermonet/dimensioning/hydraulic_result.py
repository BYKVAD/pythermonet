from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HydraulicResult:
    """
    Output from hydraulic pipe dimensioning.

    All per-trace arrays are indexed identically to network.trace_names.
    Cooling arrays are None when the system has no cooling mode.
    """

    network: object  # DistributionNetwork — kept for topology / geometry access

    # Installed pipe geometry (per trace)
    outer_diameter: np.ndarray   # [m]
    inner_diameter: np.ndarray   # [m]

    governing_mode: np.ndarray   # dtype=object, values: "heating" | "cooling" | "equal"

    # Volume flows (per trace)
    m3_s_heating: np.ndarray
    m3_s_cooling: np.ndarray | None

    # Reynolds numbers for installed network (per trace)
    Re_heating: np.ndarray
    Re_cooling: np.ndarray | None

    # Total pressure loss for installed network (per trace) [Pa]
    dp_heating: np.ndarray
    dp_cooling: np.ndarray | None
