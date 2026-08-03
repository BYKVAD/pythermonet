# src/pythermonet/components/distribution_network.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.material import Material

@dataclass(frozen=True, slots=True)
class DistributionNetwork:
    pipe_infrastructure: PipeInfrastructure
    names_trace: list[str]
    heat_pump_ids_trace: list[np.ndarray]
    pressure_losses_max_trace: np.ndarray
    sdr: np.ndarray
    lengths_trace: np.ndarray
    counts_trace: np.ndarray

    material_pipe: Material | None = None  # shared pipe material — must be identical across all trace segments

    def __post_init__(self) -> None:
        segments = self.pipe_infrastructure.segments_trace

        if len(segments) == 0:
            raise ValueError("No segments_trace in infrastructure.")

        first = segments[0].material
        k0 = first.thermal_conductivity
        rho0 = first.density  # eller hvad din anden property hedder
        c0 = first.specific_heat

        for seg in segments[1:]:
            mat = seg.material
            if mat.thermal_conductivity != k0 or mat.density != rho0 or mat.specific_heat != c0:
                raise ValueError(
                    "All pipe materials must have identical thermal properties."
                )

        object.__setattr__(self, "material_pipe", first)
