from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HydraulicResult:
    """
    Output from hydraulic pipe dimensioning.

    All per-trace arrays are indexed identically to network.names_trace.
    Cooling arrays are None when the system has no cooling mode.
    """

    network: object  # DistributionNetwork — kept for topology / geometry access

    # Installed pipe geometry (per trace)
    diameters_outer: np.ndarray   # [m]
    diameters_inner: np.ndarray   # [m]

    governing_mode: np.ndarray   # dtype=object, values: "heating" | "cooling" | "equal"

    # Volume flows (per trace) [m³/s]
    volume_flow_rates_peak_heating: np.ndarray
    volume_flow_rates_peak_cooling: np.ndarray | None

    # Reynolds numbers for installed network (per trace)
    reynolds_numbers_heating: np.ndarray
    reynolds_numbers_cooling: np.ndarray | None

    # Total pressure loss for installed network (per trace) [Pa]
    pressure_losses_heating: np.ndarray
    pressure_losses_cooling: np.ndarray | None
