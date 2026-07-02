# src/pythermonet/components/distribution_network.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.material import Material

@dataclass(frozen=True, slots=True)
class DistributionNetwork:
    infrastructure: PipeInfrastructure
    trace_names: list[str]
    heat_pump_id_trace: list[np.ndarray]
    trace_max_pressure_loss: np.ndarray
    sdr: np.ndarray
    trace_lengths: np.ndarray
    trace_counts: np.ndarray

    pipe_material: Material | None = None  # shared pipe material — must be identical across all trace segments

    def __post_init__(self) -> None:
        segments = self.infrastructure.trace_segments

        if len(segments) == 0:
            raise ValueError("No trace_segments in infrastructure.")

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

        object.__setattr__(self, "pipe_material", first)
