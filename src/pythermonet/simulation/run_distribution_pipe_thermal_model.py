from __future__ import annotations

from typing import Optional
import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.simulation.distribution_pipe_thermal_model import (
    PipeGroup,
    ModeInput,
    DistributionPipeModel,
    ModeResult,
    compute_distribution_pipe_thermal_response,
)


def _build_pipe_groups_from_network(*, network, Re_arr: np.ndarray) -> list[PipeGroup]:
    L_oneway = np.asarray(network.L_traces, dtype=float)
    npp = int(network.infrastructure.NParallelPipes)

    Do = np.asarray(
        [seg.outerDiameter for seg in network.infrastructure.traceSegments],
        dtype=float,
    )
    SDR = np.asarray(network.SDR, dtype=float)
    Re_arr = np.asarray(Re_arr, dtype=float)
    n_traces = np.asarray(network.N_traces, dtype=int)

    if SDR.ndim == 0:
        SDR = np.full_like(Do, float(SDR), dtype=float)

    if not (len(L_oneway) == len(Do) == len(SDR) == len(Re_arr) == len(n_traces)):
        raise ValueError(
            "Inconsistent lengths in network data: "
            f"L_traces={len(L_oneway)}, Do={len(Do)}, SDR={len(SDR)}, "
            f"Re_arr={len(Re_arr)}, N_traces={len(n_traces)}"
        )

    L_m = npp * L_oneway
    Di = Do * (1.0 - 2.0 / SDR)

    pipe_spacing_m = float(network.infrastructure.pipeDistance) if npp > 1 else None
    burial_depth_m = float(network.infrastructure.burialDepth)
    k_pipe_W_mK = float(network.globalMaterial.thermalCond)

    out: list[PipeGroup] = []
    for i in range(len(L_m)):
        out.append(
            PipeGroup(
                ID=i,
                L_m=float(L_m[i]),
                Di_m=float(Di[i]),
                Do_m=float(Do[i]),
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
    network,
    brine: HeatCarrier,
    soil: Soil,
    heat_pumps,
    times_heat_s: np.ndarray,
    times_cool_s: Optional[np.ndarray] = None,
    T_brine_min_heat: float = 0.0,
    T_brine_max_cool: Optional[float] = None,
) -> dict[str, ModeResult]:
    P_heat = np.asarray(heat_pumps.heating_ground_load_W, dtype=float)
    Re_heat = np.asarray(network.dimensionedPipeReynoldsNumberHeating, dtype=float)

    heating = ModeInput(
        times_s=np.asarray(times_heat_s, dtype=float),
        powers_W=P_heat,
        Ti_C=float(T_brine_min_heat),
        To_C=float(T_brine_min_heat - heat_pumps.deltaT_sys_heat),
    )

    pipe_groups = _build_pipe_groups_from_network(network=network, Re_arr=Re_heat)

    cooling = None
    P_cool = getattr(heat_pumps, "cooling_ground_load_W", None)

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