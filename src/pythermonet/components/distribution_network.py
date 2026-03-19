# src/pythermonet/components/distribution_network.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.material import Material

@dataclass
class DistributionNetwork:
    infrastructure: PipeInfrastructure
    trace_names: list[str]
    hp_id_trace: list[np.ndarray]
    max_pressure_loss_trace: np.ndarray
    SDR: np.ndarray
    L_traces: np.ndarray
    N_traces: np.ndarray

    globalMaterial: Material | None = None  # sættes automatisk

    def __post_init__(self) -> None:
        segments = self.infrastructure.traceSegments

        if len(segments) == 0:
            raise ValueError("No traceSegments in infrastructure.")

        first = segments[0].material
        k0 = first.thermalCond
        rho0 = first.rho  # eller hvad din anden property hedder
        c0 = first.c

        for seg in segments[1:]:
            mat = seg.material
            if mat.thermalCond != k0 or mat.rho != rho0 or mat.c != c0:
                raise ValueError(
                    "All pipe materials must have identical thermal properties."
                )

        # sæt globalMaterial til første
        self.globalMaterial = first