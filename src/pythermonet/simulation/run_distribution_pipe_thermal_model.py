from __future__ import annotations

from typing import Optional
import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.dimensioning.BHE.borehole_length import apply_annual_balance
from pythermonet.simulation.distribution_pipe_thermal_model import (
    PipeGroup,
    ModeInput,
    DistributionPipeModel,
    ThermonetPerformance,
    compute_distribution_pipe_thermal_response,
)


def _build_pipe_groups_from_hydraulic(hydraulic: HydraulicResult, Re_arr: np.ndarray) -> list[PipeGroup]:
    network = hydraulic.network
    L_oneway = np.asarray(network.lengths_trace, dtype=float)
    npp = int(network.pipe_infrastructure.n_pipes_parallel)
    n_traces = np.asarray(network.counts_trace, dtype=int)
    Re_arr = np.asarray(Re_arr, dtype=float)

    L_m = npp * L_oneway
    pipe_spacing = float(network.pipe_infrastructure.pipe_spacing) if npp > 1 else None
    burial_depth = float(network.pipe_infrastructure.burial_depth)
    k_pipe = float(network.material_pipe.thermal_conductivity)

    out: list[PipeGroup] = []
    for i in range(len(L_m)):
        out.append(
            PipeGroup(
                id_=i,
                length=float(L_m[i]),
                diameter_inner=float(hydraulic.diameters_inner[i]),
                diameter_outer=float(hydraulic.diameters_outer[i]),
                reynolds_number=float(Re_arr[i]),
                thermal_conductivity_pipe=k_pipe,
                burial_depth=burial_depth,
                n_parallel_pipes=npp,
                n_traces=int(n_traces[i]),
                pipe_spacing=pipe_spacing,
            )
        )
    return out


def compute_distribution_pipe_thermal_capacity(
    *,
    hydraulic: HydraulicResult,
    brine: HeatCarrier,
    soil: Soil,
    ground_loads,
    times_heat_s: np.ndarray,
    times_cool_s: Optional[np.ndarray] = None,
    T_brine_min_heat: float = 0.0,
    T_brine_max_cool: Optional[float] = None,
) -> dict[str, ThermonetPerformance]:
    P_heat = np.asarray(ground_loads.loads_ground_heating, dtype=float)
    P_cool = getattr(ground_loads, "loads_ground_cooling", None)

    if P_cool is not None:
        P_heat, P_cool = apply_annual_balance(P_heat, np.asarray(P_cool, dtype=float))

    heating = ModeInput(
        pulse_timescales=np.asarray(times_heat_s, dtype=float),
        pulse_loads=P_heat,
        temperature_heat_pump_inlet=float(T_brine_min_heat),
        temperature_heat_pump_outlet=float(T_brine_min_heat - ground_loads.temperature_delta_brine_flow_weighted_heating),
    )

    pipe_groups_heat = _build_pipe_groups_from_hydraulic(hydraulic, Re_arr=hydraulic.reynolds_numbers_heating)

    model_heat = DistributionPipeModel(
        brine=brine,
        soil=soil,
        temperature_ground_undisturbed=float(soil.temperature_surface_mean),
        temperature_surface_amplitude=float(soil.temperature_surface_amplitude),
        pipe_groups=pipe_groups_heat,
        heating=heating,
        cooling=None,
    )
    results = compute_distribution_pipe_thermal_response(model_heat)

    if P_cool is not None:
        if times_cool_s is None:
            raise ValueError("times_cool_s must be provided when ground_load_cooling exists")
        if T_brine_max_cool is None:
            raise ValueError("T_brine_max_cool must be provided when ground_load_cooling exists")

        cooling = ModeInput(
            pulse_timescales=np.asarray(times_cool_s, dtype=float),
            pulse_loads=np.asarray(P_cool, dtype=float),
            temperature_heat_pump_inlet=float(T_brine_max_cool),
            temperature_heat_pump_outlet=float(T_brine_max_cool + ground_loads.temperature_delta_brine_flow_weighted_cooling),
        )

        # Use the cooling Reynolds number so the pipe thermal resistance is
        # computed at the actual cooling flow rate (which differs from heating).
        Re_cool_arr = (
            hydraulic.reynolds_numbers_cooling
            if hydraulic.reynolds_numbers_cooling is not None
            else hydraulic.reynolds_numbers_heating
        )
        pipe_groups_cool = _build_pipe_groups_from_hydraulic(hydraulic, Re_arr=Re_cool_arr)

        model_cool = DistributionPipeModel(
            brine=brine,
            soil=soil,
            temperature_ground_undisturbed=float(soil.temperature_surface_mean),
            temperature_surface_amplitude=float(soil.temperature_surface_amplitude),
            pipe_groups=pipe_groups_cool,
            heating=heating,  # required field; only the cooling result is used
            cooling=cooling,
        )
        results["cooling"] = compute_distribution_pipe_thermal_response(model_cool)["cooling"]

    return results

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
            str(seg.id_),
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