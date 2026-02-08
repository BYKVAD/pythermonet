# src/pythermonet/components/distribution_network.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pythermonet.components.pipe_infrastructure import PipeInfrastructure

@dataclass
class DistributionNetwork:
    infrastructure: PipeInfrastructure
    trace_names: list[str]
    hp_id_trace: list[np.ndarray]
    max_pressure_loss_trace: np.ndarray
    SDR: np.ndarray
    L_traces: np.ndarray
    N_traces: np.ndarray
    L_segments: np.ndarray

    # outputs from dimensioning
    dimensioned_pipe_diameter_heating: np.ndarray | None = None
    dimensioned_pipe_inner_diameter_heating: np.ndarray | None = None
    dimensioned_pipe_reynolds_number_heating: np.ndarray | None = None
    dimensioned_pipe_diameter_cooling: np.ndarray | None = None
    dimensioned_pipe_inner_diameter_cooling: np.ndarray | None = None
    dimensioned_pipe_reynolds_number_cooling: np.ndarray | None = None
    V_brine: float | None = None
