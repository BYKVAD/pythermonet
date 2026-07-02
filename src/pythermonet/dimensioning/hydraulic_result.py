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
    pipe_outer_diameters: np.ndarray   # [m]
    pipe_inner_diameters: np.ndarray   # [m]

    governing_mode: np.ndarray   # dtype=object, values: "heating" | "cooling" | "equal"

    # Volume flows (per trace) [m³/s]
    peak_volume_flow_rate_heating: np.ndarray
    peak_volume_flow_rate_cooling: np.ndarray | None

    # Reynolds numbers for installed network (per trace)
    reynolds_numbers_heating: np.ndarray
    reynolds_numbers_cooling: np.ndarray | None

    # Total pressure loss for installed network (per trace) [Pa]
    pressure_losses_heating: np.ndarray
    pressure_losses_cooling: np.ndarray | None
