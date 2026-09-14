from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pythermonet.components.ground_loads import GroundLoads
from pythermonet.components.vhe_field import VHEField
from pythermonet.components.heat_pump import BrineTemperatureLimits
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.BHE.borehole_length import (
    FieldSizingResult,
    apply_annual_balance,
    size_borehole_length_heating_cooling,
)
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.dimensioning.system_temperatures import (
    SystemBrineTemperatureResult,
    compute_system_brine_temperatures,
)
from pythermonet.physics.hydraulics import pressure_loss_per_length as _dp_per_m
from pythermonet.simulation.distribution_pipe_thermal_model import ThermonetPerformance
from pythermonet.simulation.run_distribution_pipe_thermal_model import (
    compute_distribution_pipe_thermal_capacity,
)


@dataclass(frozen=True)
class BHEWorkflowResult:
    """
    Output of the BHE sizing workflow.

    Load arrays are ordered [annual, winter, peak] in watts.
    ``loads_bhe_heating`` is the BHE portion after subtracting the
    thermonet distribution pipe contribution from ``loads_total_heating``.
    """

    sizing: FieldSizingResult
    temperatures_system: SystemBrineTemperatureResult
    loads_bhe_heating: np.ndarray                       # [W]
    loads_bhe_cooling: np.ndarray | None                # [W]
    loads_total_heating: np.ndarray                     # [W]
    loads_total_cooling: np.ndarray | None              # [W]
    performance_thermonet_heating: ThermonetPerformance
    performance_thermonet_cooling: ThermonetPerformance | None
    hydraulic: HydraulicResult
    brine: HeatCarrier
    pressure_loss_bhe_heating: float                    # [Pa]
    pressure_loss_bhe_cooling: float | None             # [Pa]


def run_bhe_sizing_workflow(
    *,
    ground_loads: GroundLoads,
    vhe_field: VHEField,
    hydraulic: HydraulicResult,
    brine: HeatCarrier,
    soil: Soil,
    sizing: SizingParameters,
    brine_temperature_limits: BrineTemperatureLimits,
) -> BHEWorkflowResult:
    """
    Run the complete BHE sizing workflow from ground loads to system temperatures.

    Covers: annual balance, distribution pipe thermal model, BHE load
    subtraction, borehole length sizing, and system brine temperature
    computation.
    """
    temperature_brine_min_heating = brine_temperature_limits.temperature_brine_min_heating
    temperature_brine_max_cooling = brine_temperature_limits.temperature_brine_max_cooling

    # Annual balance
    P_heat_full = np.asarray(ground_loads.loads_ground_heating, dtype=float)
    P_cool_full = (
        np.asarray(ground_loads.loads_ground_cooling, dtype=float)
        if ground_loads.has_cooling else None
    )
    if P_cool_full is not None:
        P_heat_full, P_cool_full = apply_annual_balance(P_heat_full.copy(), P_cool_full.copy())

    times_heat_s = sizing.times_s_peak_heating(ground_loads.hours_peak_heating)
    times_cool_s = (
        sizing.times_s_peak_cooling(ground_loads.hours_peak_cooling)
        if ground_loads.has_cooling else None
    )

    # Distribution pipe thermal model
    dist_thermal = compute_distribution_pipe_thermal_capacity(
        hydraulic=hydraulic,
        brine=brine,
        soil=soil,
        ground_loads=ground_loads,
        times_heat_s=np.flip(times_heat_s),
        times_cool_s=np.flip(times_cool_s) if times_cool_s is not None else None,
        T_brine_min_heat=temperature_brine_min_heating,
        T_brine_max_cool=temperature_brine_max_cooling if ground_loads.has_cooling else None,
    )

    # BHE loads = full balanced loads × (1 − distribution fraction)
    P_heating = (1 - dist_thermal["heating"].load_supply_fraction) * P_heat_full
    P_cooling = (
        (1 - dist_thermal["cooling"].load_supply_fraction) * P_cool_full
        if ground_loads.has_cooling else None
    )

    n_boreholes = vhe_field.n_boreholes

    # Size borehole length
    field_sizing = size_borehole_length_heating_cooling(
        T_fluid_min=temperature_brine_min_heating - 0.5 * ground_loads.temperature_delta_brine_flow_weighted_heating,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        T_fluid_max=temperature_brine_max_cooling + (
            0.5 * ground_loads.temperature_delta_brine_flow_weighted_cooling if ground_loads.has_cooling else 0.0
        ),
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        vhe_field=vhe_field,
        brine=brine,
        soil=soil,
        m_dot_per_borehole_heat_kg_s=ground_loads.mass_flow_peak_summed_heating / n_boreholes,
        m_dot_per_borehole_cool_kg_s=(
            ground_loads.mass_flow_peak_summed_cooling / n_boreholes
            if ground_loads.has_cooling else None
        ),
        pre_balanced=True,
    )

    # System brine temperatures
    temperatures_system = compute_system_brine_temperatures(
        sizing=field_sizing,
        vhe_field=vhe_field,
        hydraulic=hydraulic,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        soil=soil,
        performance_thermonet_heating=dist_thermal["heating"],
        performance_thermonet_cooling=dist_thermal["cooling"] if ground_loads.has_cooling else None,
        pre_balanced=True,
    )

    # BHE pressure drop at peak flow (2 legs of the U-pipe)
    Di_bhe = float(vhe_field.pipe.diameter_outer) * (1.0 - 2.0 / float(vhe_field.pipe.sdr))
    Q_per_bh_heat = ground_loads.mass_flow_peak_summed_heating / (n_boreholes * brine.density)
    pressure_loss_bhe_heating = (
        float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_bh_heat, Di_bhe))
        * 2.0 * field_sizing.length_element
    )

    pressure_loss_bhe_cooling: float | None = None
    if ground_loads.has_cooling:
        Q_per_bh_cool = ground_loads.mass_flow_peak_summed_cooling / (n_boreholes * brine.density)
        pressure_loss_bhe_cooling = (
            float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_bh_cool, Di_bhe))
            * 2.0 * field_sizing.length_element
        )

    return BHEWorkflowResult(
        sizing=field_sizing,
        temperatures_system=temperatures_system,
        loads_bhe_heating=P_heating,
        loads_bhe_cooling=P_cooling,
        loads_total_heating=P_heat_full,
        loads_total_cooling=P_cool_full,
        performance_thermonet_heating=dist_thermal["heating"],
        performance_thermonet_cooling=dist_thermal["cooling"] if ground_loads.has_cooling else None,
        hydraulic=hydraulic,
        brine=brine,
        pressure_loss_bhe_heating=pressure_loss_bhe_heating,
        pressure_loss_bhe_cooling=pressure_loss_bhe_cooling,
    )
