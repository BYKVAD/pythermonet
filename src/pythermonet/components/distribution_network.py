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
    hp_id_trace: list[np.ndarray]
    max_pressure_loss_trace: np.ndarray
    sdr: np.ndarray
    L_traces: np.ndarray
    N_traces: np.ndarray

    globalMaterial: Material | None = None  # sættes automatisk

    def __post_init__(self) -> None:
        segments = self.infrastructure.traceSegments

        if len(segments) == 0:
            raise ValueError("No traceSegments in infrastructure.")

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

        # sæt globalMaterial til første
        object.__setattr__(self, "globalMaterial", first)