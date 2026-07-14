from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.system.diversity_factor import diversity_factor_from_n_heat_pumps

PeakSupplyMode = Literal["absolute", "incremental"]


@dataclass(frozen=True, slots=True)
class GroundLoads:
    """
    Aggregated ground-side thermal loads and flow parameters.

    Provides the unified interface consumed by all dimensioning workflows.
    Construct via :func:`ground_loads_from_heat_pumps` or
    :func:`ground_loads_from_district`.

    Arrays are ordered [annual, seasonal, peak] (the three-pulse convention).
    """

    loads_ground_heating: np.ndarray           # [W] (3,): annual, winter, peak
    loads_ground_cooling: Optional[np.ndarray] # [W] (3,) or None

    has_cooling: bool

    hours_peak_heating: float
    hours_peak_cooling: Optional[float]

    temperature_delta_brine_flow_weighted_heating: float      # [K] flow-weighted brine ΔT, heating
    temperature_delta_brine_flow_weighted_cooling: Optional[float]  # [K] or None

    mass_flow_peak_summed_heating: float       # [kg/s] summed peak mass flow, heating
    mass_flow_peak_summed_cooling: Optional[float]   # [kg/s] or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_peak_fraction(
    *,
    P_base: float,
    P_peak: float,
    mode: PeakSupplyMode,
    alpha: float,
) -> float:
    if mode == "absolute":
        return float(alpha * P_peak)
    return float(P_base + alpha * (P_peak - P_base))


# ---------------------------------------------------------------------------
# Factory: individual heat pumps
# ---------------------------------------------------------------------------

def ground_loads_from_heat_pumps(
    hp_list,
    brine: HeatCarrier,
    *,
    peak_hours_heating: float = 4.0,
    peak_hours_cooling: Optional[float] = 4.0,
    peak_fraction_heating_mode: PeakSupplyMode = "incremental",
    peak_fraction_heating: float = 1.0,
    peak_fraction_cooling_mode: PeakSupplyMode = "incremental",
    peak_fraction_cooling: float = 1.0,
) -> GroundLoads:
    """
    Aggregate a list of individual HeatPump objects into ground-side loads.

    Diversity factor is applied automatically from the HP count.
    """
    n = len(hp_list)
    if n <= 0:
        raise ValueError("hp_list must contain at least one HeatPump")
    if peak_hours_heating <= 0:
        raise ValueError("peak_hours_heating must be > 0")
    if peak_hours_cooling is not None and peak_hours_cooling <= 0:
        raise ValueError("peak_hours_cooling must be > 0 when provided")
    if not (0.0 <= peak_fraction_heating <= 1.0):
        raise ValueError("peak_fraction_heating must be in [0, 1]")
    if not (0.0 <= peak_fraction_cooling <= 1.0):
        raise ValueError("peak_fraction_cooling must be in [0, 1]")

    cp = float(brine.specific_heat)
    rho = float(brine.density)
    if cp <= 0 or rho <= 0:
        raise ValueError("brine.specific_heat and brine.density must be > 0")

    f = float(diversity_factor_from_n_heat_pumps(n))

    # --- Heating loads ---
    P_year = float(sum(hp.load_ground_annual_heating for hp in hp_list))
    P_wint = float(sum(hp.load_ground_winter_heating for hp in hp_list))
    P_peak_raw = float(sum(hp.load_ground_peak_heating for hp in hp_list))
    P_peak_div = f * P_peak_raw
    P_peak_eff = _apply_peak_fraction(
        P_base=P_wint, P_peak=P_peak_div,
        mode=peak_fraction_heating_mode, alpha=peak_fraction_heating,
    )
    ground_load_heating = np.asarray([P_year, P_wint, P_peak_eff], dtype=float)

    # --- Heating peak flow + system ΔT ---
    sum_mdot_raw = 0.0
    sum_mdot_dT = 0.0
    for hp in hp_list:
        dT = float(hp.temperature_delta_heating)
        if dT <= 0:
            raise ValueError(f"HeatPump ID={hp.id_}: temperature_delta_heating must be > 0")
        mdot = float(hp.load_ground_peak_heating) / (cp * dT)
        sum_mdot_raw += mdot
        sum_mdot_dT += mdot * dT

    if sum_mdot_raw <= 0.0:
        raise ValueError("Sum of peak heating mass flow must be > 0")

    flow_weighted_brine_delta_temperature_heating = float(sum_mdot_dT / sum_mdot_raw)
    scale_H = 0.0 if P_peak_div == 0.0 else (P_peak_eff / P_peak_div)
    summed_peak_mass_flow_heating = float(scale_H * f * sum_mdot_raw)

    # --- Cooling detection ---
    P_peak_c_raw = float(sum(abs(float(hp.load_ground_peak_cooling)) for hp in hp_list))
    if np.isclose(P_peak_c_raw, 0.0, atol=1.0):
        return GroundLoads(
            loads_ground_heating=ground_load_heating,
            loads_ground_cooling=None,
            has_cooling=False,
            hours_peak_heating=peak_hours_heating,
            hours_peak_cooling=peak_hours_cooling,
            temperature_delta_brine_flow_weighted_heating=flow_weighted_brine_delta_temperature_heating,
            temperature_delta_brine_flow_weighted_cooling=None,
            mass_flow_peak_summed_heating=summed_peak_mass_flow_heating,
            mass_flow_peak_summed_cooling=None,
        )

    # --- Cooling loads ---
    P_year_c = float(sum(abs(float(hp.load_ground_annual_cooling)) for hp in hp_list))
    P_summ_c = float(sum(abs(float(hp.load_ground_summer_cooling)) for hp in hp_list))
    P_peak_c_div = f * P_peak_c_raw
    if peak_hours_cooling is None:
        P_peak_c_eff = P_summ_c
    else:
        P_peak_c_eff = _apply_peak_fraction(
            P_base=P_summ_c, P_peak=P_peak_c_div,
            mode=peak_fraction_cooling_mode, alpha=peak_fraction_cooling,
        )
    ground_load_cooling = np.asarray([P_year_c, P_summ_c, P_peak_c_eff], dtype=float)

    # --- Cooling peak flow + system ΔT ---
    sum_mdot_c_raw = 0.0
    sum_mdot_dT_c = 0.0
    for hp in hp_list:
        Pmag = abs(float(hp.load_ground_peak_cooling))
        if Pmag <= 0.0:
            continue
        dT = float(hp.temperature_delta_cooling)
        if dT <= 0:
            raise ValueError(
                f"HeatPump ID={hp.id_}: temperature_delta_cooling must be > 0 when cooling is active"
            )
        mdot = Pmag / (cp * dT)
        sum_mdot_c_raw += mdot
        sum_mdot_dT_c += mdot * dT

    if sum_mdot_c_raw <= 0.0:
        # Loads exist but flows are inconsistent — return without cooling flows
        return GroundLoads(
            loads_ground_heating=ground_load_heating,
            loads_ground_cooling=ground_load_cooling,
            has_cooling=True,
            hours_peak_heating=peak_hours_heating,
            hours_peak_cooling=peak_hours_cooling,
            temperature_delta_brine_flow_weighted_heating=flow_weighted_brine_delta_temperature_heating,
            temperature_delta_brine_flow_weighted_cooling=None,
            mass_flow_peak_summed_heating=summed_peak_mass_flow_heating,
            mass_flow_peak_summed_cooling=None,
        )

    flow_weighted_brine_delta_temperature_cooling = float(sum_mdot_dT_c / sum_mdot_c_raw)
    scale_C = 0.0 if P_peak_c_div == 0.0 else (P_peak_c_eff / P_peak_c_div)
    summed_peak_mass_flow_cooling = float(scale_C * f * sum_mdot_c_raw)

    return GroundLoads(
        loads_ground_heating=ground_load_heating,
        loads_ground_cooling=ground_load_cooling,
        has_cooling=True,
        hours_peak_heating=peak_hours_heating,
        hours_peak_cooling=peak_hours_cooling,
        temperature_delta_brine_flow_weighted_heating=flow_weighted_brine_delta_temperature_heating,
        temperature_delta_brine_flow_weighted_cooling=flow_weighted_brine_delta_temperature_cooling,
        mass_flow_peak_summed_heating=summed_peak_mass_flow_heating,
        mass_flow_peak_summed_cooling=summed_peak_mass_flow_cooling,
    )


# ---------------------------------------------------------------------------
# Factory: aggregated district load
# ---------------------------------------------------------------------------

def ground_loads_from_district(
    load_input,
    brine: HeatCarrier,
    *,
    f_peak_heating: float = 1.0,
    f_peak_cooling: float = 1.0,
    peak_hours_heating: float = 4.0,
    peak_hours_cooling: Optional[float] = 4.0,
) -> GroundLoads:
    """
    Derive ground-side loads from aggregated district-level loads (AggregatedLoadInput).

    The peak is scaled by  f_peak × diversity_factor(n_consumers).
    """
    li = load_input
    cp = float(brine.specific_heat)

    if not (0.0 <= f_peak_heating <= 1.0):
        raise ValueError(f"f_peak_heating must be in [0, 1]. Got {f_peak_heating}")
    if not (0.0 <= f_peak_cooling <= 1.0):
        raise ValueError(f"f_peak_cooling must be in [0, 1]. Got {f_peak_cooling}")
    if peak_hours_heating <= 0:
        raise ValueError("peak_hours_heating must be > 0")
    if peak_hours_cooling is not None and peak_hours_cooling <= 0:
        raise ValueError("peak_hours_cooling must be > 0 when provided")
    if li.cop_annual_heating <= 0 or li.cop_winter_heating <= 0 or li.cop_peak_heating <= 0:
        raise ValueError("All COP values must be > 0")
    if li.temperature_delta_heating <= 0:
        raise ValueError("temperature_delta_heating must be > 0")

    P_ann_H = li.load_annual_heating  * (1.0 - 1.0 / li.cop_annual_heating)
    P_win_H = li.load_winter_heating  * (1.0 - 1.0 / li.cop_winter_heating)
    P_raw_H = li.load_peak_heating * (1.0 - 1.0 / li.cop_peak_heating)
    S_H = f_peak_heating * diversity_factor_from_n_heat_pumps(li.n_consumers_heating)
    P_peak_H = P_raw_H * S_H

    ground_load_heating = np.array([P_ann_H, P_win_H, P_peak_H], dtype=float)
    flow_weighted_brine_delta_temperature_heating = float(li.temperature_delta_heating)
    summed_peak_mass_flow_heating = float(P_peak_H / (cp * li.temperature_delta_heating))

    if not li.has_cooling:
        return GroundLoads(
            loads_ground_heating=ground_load_heating,
            loads_ground_cooling=None,
            has_cooling=False,
            hours_peak_heating=peak_hours_heating,
            hours_peak_cooling=peak_hours_cooling,
            temperature_delta_brine_flow_weighted_heating=flow_weighted_brine_delta_temperature_heating,
            temperature_delta_brine_flow_weighted_cooling=None,
            mass_flow_peak_summed_heating=summed_peak_mass_flow_heating,
            mass_flow_peak_summed_cooling=None,
        )

    if li.eer <= 0:
        raise ValueError("eer must be > 0 when cooling is active")
    if li.temperature_delta_cooling <= 0:
        raise ValueError("temperature_delta_cooling must be > 0 when cooling is active")

    P_ann_C = li.load_annual_cooling  * (1.0 + 1.0 / li.eer)
    P_sum_C = li.load_summer_cooling  * (1.0 + 1.0 / li.eer)
    P_raw_C = li.load_peak_cooling * (1.0 + 1.0 / li.eer)
    S_C = f_peak_cooling * diversity_factor_from_n_heat_pumps(li.n_consumers_cooling)
    P_peak_C = P_raw_C * S_C

    return GroundLoads(
        loads_ground_heating=ground_load_heating,
        loads_ground_cooling=np.array([P_ann_C, P_sum_C, P_peak_C], dtype=float),
        has_cooling=True,
        hours_peak_heating=peak_hours_heating,
        hours_peak_cooling=peak_hours_cooling,
        temperature_delta_brine_flow_weighted_heating=flow_weighted_brine_delta_temperature_heating,
        temperature_delta_brine_flow_weighted_cooling=float(li.temperature_delta_cooling),
        mass_flow_peak_summed_heating=summed_peak_mass_flow_heating,
        mass_flow_peak_summed_cooling=float(P_peak_C / (cp * li.temperature_delta_cooling)),
    )
