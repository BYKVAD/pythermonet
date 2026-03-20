from __future__ import annotations

from typing import Optional
import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.dimensioning.borehole_length import apply_annual_balance
from pythermonet.simulation.distribution_pipe_thermal_model import (
    PipeGroup,
    ModeInput,
    DistributionPipeModel,
    ModeResult,
    compute_distribution_pipe_thermal_response,
)


def _build_pipe_groups_from_hydraulic(hydraulic: HydraulicResult, Re_arr: np.ndarray) -> list[PipeGroup]:
    network = hydraulic.network
    L_oneway = np.asarray(network.L_traces, dtype=float)
    npp = int(network.infrastructure.NParallelPipes)
    n_traces = np.asarray(network.N_traces, dtype=int)
    Re_arr = np.asarray(Re_arr, dtype=float)

    L_m = npp * L_oneway
    pipe_spacing_m = float(network.infrastructure.pipeDistance) if npp > 1 else None
    burial_depth_m = float(network.infrastructure.burialDepth)
    k_pipe_W_mK = float(network.globalMaterial.thermalCond)

    out: list[PipeGroup] = []
    for i in range(len(L_m)):
        out.append(
            PipeGroup(
                ID=i,
                L_m=float(L_m[i]),
                Di_m=float(hydraulic.inner_diameter[i]),
                Do_m=float(hydraulic.outer_diameter[i]),
                Re=float(Re_arr[i]),
                k_pipe_W_mK=k_pipe_W_mK,
                burial_depth_m=burial_depth_m,
                n_parallel_pipes=npp,
                n_traces=int(n_traces[i]),
                pipe_spacing_m=pipe_spacing_m,
            )
        )
    return out


def compute_distribution_pipe_thermal_capacity(
    *,
    hydraulic: HydraulicResult,
    brine: HeatCarrier,
    soil: Soil,
    heat_pumps,
    times_heat_s: np.ndarray,
    times_cool_s: Optional[np.ndarray] = None,
    T_brine_min_heat: float = 0.0,
    T_brine_max_cool: Optional[float] = None,
) -> dict[str, ModeResult]:
    P_heat = np.asarray(heat_pumps.heating_ground_load_W, dtype=float)
    P_cool = getattr(heat_pumps, "cooling_ground_load_W", None)

    if P_cool is not None:
        P_heat, P_cool = apply_annual_balance(P_heat, np.asarray(P_cool, dtype=float))

    heating = ModeInput(
        times_s=np.asarray(times_heat_s, dtype=float),
        powers_W=P_heat,
        Ti_C=float(T_brine_min_heat),
        To_C=float(T_brine_min_heat - heat_pumps.deltaT_sys_heat),
    )

    pipe_groups = _build_pipe_groups_from_hydraulic(hydraulic, Re_arr=hydraulic.Re_heating)

    cooling = None

    if P_cool is not None:
        if times_cool_s is None:
            raise ValueError("times_cool_s must be provided when cooling_ground_load_W exists")
        if T_brine_max_cool is None:
            raise ValueError("T_brine_max_cool must be provided when cooling_ground_load_W exists")

        cooling = ModeInput(
            times_s=np.asarray(times_cool_s, dtype=float),
            powers_W=np.asarray(P_cool, dtype=float),
            Ti_C=float(T_brine_max_cool),
            To_C=float(T_brine_max_cool + heat_pumps.deltaT_sys_cool),
        )

    model = DistributionPipeModel(
        brine=brine,
        soil=soil,
        T0_C=float(soil.surfaceTemp),
        surface_amp_C=float(soil.surfaceTempAmp),
        pipe_groups=pipe_groups,
        heating=heating,
        cooling=cooling,
    )

    return compute_distribution_pipe_thermal_response(model)

def print_pipe_thermal_table(network, decimals: int = 2):
    """
    Print thermal load fractions for each pipe segment.
    Fractions are assumed to already represent the fraction
    of the total thermal load.
    """

    headers = ["ID", "L [m]", "Heat [%]", "Cool [%]", "Sum [%]"]

    rows = []
    sum_heat = 0.0
    sum_cool = 0.0

    for seg in network.pipe_segments:

        heat_pct = 100 * getattr(seg, "F_heat", 0.0)
        cool_pct = 100 * getattr(seg, "F_cool", 0.0)

        total_pct = heat_pct + cool_pct

        sum_heat += heat_pct
        sum_cool += cool_pct

        rows.append([
            str(seg.ID),
            f"{seg.length:.0f}",
            f"{heat_pct:.{decimals}f}",
            f"{cool_pct:.{decimals}f}",
            f"{total_pct:.{decimals}f}",
        ])

    total_row = [
        "TOTAL",
        "",
        f"{sum_heat:.{decimals}f}",
        f"{sum_cool:.{decimals}f}",
        f"{(sum_heat+sum_cool):.{decimals}f}",
    ]

    widths = [max(len(x) for x in col) for col in zip(headers, *rows, total_row)]

    def fmt(row):
        return "  ".join(row[i].rjust(widths[i]) if i else row[i].ljust(widths[i]) for i in range(len(row)))

    print("\nThermal pipe contribution")
    print(fmt(headers))
    print("  ".join("-"*w for w in widths))

    for r in rows:
        print(fmt(r))

    print("  ".join("-"*w for w in widths))
    print(fmt(total_row))