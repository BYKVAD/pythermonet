from __future__ import annotations

from typing import Optional
import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.simulation.distribution_pipe_thermal import (
    PipeGroupThermalInput,
    PulseResponseSpec,
    PipeThermalResponseModeInput,
    DistributionPipeThermalInput,
    DistributionPipeThermalResult,
    compute_distribution_pipe_thermal_response,
)


def _build_pipe_groups_from_network(*, network) -> list[PipeGroupThermalInput]:
    L_oneway = network.L_traces
    SDR = network.SDR
    npp = network.infrastructure.NParallelPipes

    # vælg én af dem i caller; her bruger vi heating som default i det her simple snit
    Re_arr = network.dimensionedPipeReynoldsNumberHeating

    Do = np.array([seg.outerDiameter for seg in network.infrastructure.traceSegments], dtype=float)

    L_m = npp * L_oneway
    Di = Do * (1.0 - 2.0 / SDR)

    out: list[PipeGroupThermalInput] = []
    for i in range(len(L_m)):
        out.append(
            PipeGroupThermalInput(
                ID=i,
                L_m=float(L_m[i]),
                Di_m=float(Di[i]),
                Do_m=float(Do[i]),
                Re=float(Re_arr[i]),
                k_pipe_W_mK=float(network.globalMaterial.thermalCond),
                burial_depth_m=float(network.infrastructure.burialDepth),
                n_parallel_pipes=int(npp),
                n_traces=int(network.N_traces[i]),
                pipe_spacing_m=float(network.infrastructure.pipeDistance) if npp > 1 else None,
            )
        )
    return out


def run_distribution_pipe_thermal(
    *,
    network,
    brine: HeatCarrier,
    soil: Soil,
    heat_pumps,
    times_heat_s: np.ndarray,
    times_cool_s: Optional[np.ndarray] = None,
    T_brine_min_heat: float = 0.0,
    T_brine_max_cool: Optional[float] = None,
) -> DistributionPipeThermalResult:
    # --- Heating ---
    P_heat = heat_pumps.heating_ground_load_W

    Ti_heat = T_brine_min_heat
    To_heat = Ti_heat - heat_pumps.deltaT_sys_heat

    heating_mode = PipeThermalResponseModeInput(
        spec=PulseResponseSpec(times_s=times_heat_s, powers_W=P_heat),
        Ti_C=float(Ti_heat),
        To_C=float(To_heat),
    )

    pipe_groups = _build_pipe_groups_from_network(network=network)

    # --- Cooling (kun hvis der findes cooling load array) ---
    cooling_mode = None
    P_cool = getattr(heat_pumps, "cooling_ground_load_W", None)

    if P_cool is not None:
        Ti_cool = float(T_brine_max_cool)
        To_cool = Ti_cool + heat_pumps.deltaT_sys_cool

        cooling_mode = PipeThermalResponseModeInput(
            spec=PulseResponseSpec(times_s=times_cool_s, powers_W=P_cool),
            Ti_C=float(Ti_cool),
            To_C=float(To_cool),
        )

    inp = DistributionPipeThermalInput(
        brine=brine,
        soil=soil,
        T0=float(soil.surfaceTemp),
        surface_amp=float(soil.surfaceTempAmp),
        pipe_groups=pipe_groups,
        heating=heating_mode,
        cooling=cooling_mode,
    )

    return compute_distribution_pipe_thermal_response(inp)