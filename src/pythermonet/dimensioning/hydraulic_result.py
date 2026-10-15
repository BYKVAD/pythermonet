from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class HydraulicResult:
    """
    Output from hydraulic pipe dimensioning.

    All per-trace arrays are indexed identically to network.names_trace.
    Cooling arrays are None when the system has no cooling mode.
    Field metadata holds each quantity's unit and description (used by the
    results JSON writer).
    """

    network: object  # DistributionNetwork — kept for topology / geometry access

    # Installed pipe geometry (per trace)
    diameters_outer: np.ndarray = field(metadata={"unit": "m", "description": "Selected outer pipe diameter per trace"})
    diameters_inner: np.ndarray = field(metadata={"unit": "m", "description": "Inner pipe diameter per trace"})

    governing_mode: np.ndarray = field(metadata={"unit": "-", "description": "Mode that governed the diameter choice per trace (heating, cooling or equal)"})

    # Volume flows (per trace) [m³/s]
    volume_flow_rates_peak_heating: np.ndarray = field(metadata={"unit": "m^3/s", "description": "Peak brine volume flow per pipe of the trace, heating"})
    volume_flow_rates_peak_cooling: np.ndarray | None = field(metadata={"unit": "m^3/s", "description": "Peak brine volume flow per pipe of the trace, cooling"})

    # Mean brine velocities for installed network (per trace) [m/s]
    velocities_heating: np.ndarray = field(metadata={"unit": "m/s", "description": "Mean brine velocity at peak flow per trace, heating"})
    velocities_cooling: np.ndarray | None = field(metadata={"unit": "m/s", "description": "Mean brine velocity at peak flow per trace, cooling"})

    # Reynolds numbers for installed network (per trace)
    reynolds_numbers_heating: np.ndarray = field(metadata={"unit": "-", "description": "Reynolds number at peak flow per trace, heating"})
    reynolds_numbers_cooling: np.ndarray | None = field(metadata={"unit": "-", "description": "Reynolds number at peak flow per trace, cooling"})

    # Pressure loss per pipe length for installed network (per trace) [Pa/m]
    pressure_losses_per_length_heating: np.ndarray = field(metadata={"unit": "Pa/m", "description": "Pressure loss per metre of a single pipe, one direction (supply or return), at peak flow per trace, heating"})
    pressure_losses_per_length_cooling: np.ndarray | None = field(metadata={"unit": "Pa/m", "description": "Pressure loss per metre of a single pipe, one direction (supply or return), at peak flow per trace, cooling"})

    # Total pressure loss for installed network (per trace) [Pa]
    pressure_losses_heating: np.ndarray = field(metadata={"unit": "Pa", "description": "Pressure loss through one trace, supply and return pipe together (n_pipes_parallel x one-way length), at peak flow, heating"})
    pressure_losses_cooling: np.ndarray | None = field(metadata={"unit": "Pa", "description": "Pressure loss through one trace, supply and return pipe together (n_pipes_parallel x one-way length), at peak flow, cooling"})
