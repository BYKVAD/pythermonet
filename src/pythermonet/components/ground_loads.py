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

    heating_ground_load_W: np.ndarray           # [W] (3,): annual, winter, peak
    cooling_ground_load_W: Optional[np.ndarray] # [W] (3,) or None

    has_cooling: bool

    peak_heating_h: float
    peak_cooling_h: Optional[float]

    deltaT_sys_heat: float                      # [K] flow-weighted brine ΔT, heating
    deltaT_sys_cool: Optional[float]            # [K] or None

    aggregated_mdot_peak_heat_kg_s: float       # [kg/s] total peak mass flow, heating
    aggregated_mdot_peak_cool_kg_s: Optional[float]  # [kg/s] or None


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
    peak_heating_h: float = 4.0,
    peak_cooling_h: Optional[float] = 4.0,
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
    if peak_heating_h <= 0:
        raise ValueError("peak_heating_h must be > 0")
    if peak_cooling_h is not None and peak_cooling_h <= 0:
        raise ValueError("peak_cooling_h must be > 0 when provided")
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
    P_year = float(sum(hp.annualHeating_ground_load for hp in hp_list))
    P_wint = float(sum(hp.winterHeating_ground_load for hp in hp_list))
    P_peak_raw = float(sum(hp.peakHeating_ground_load for hp in hp_list))
    P_peak_div = f * P_peak_raw
    P_peak_eff = _apply_peak_fraction(
        P_base=P_wint, P_peak=P_peak_div,
        mode=peak_fraction_heating_mode, alpha=peak_fraction_heating,
    )
    heating_ground_load_W = np.asarray([P_year, P_wint, P_peak_eff], dtype=float)

    # --- Heating peak flow + system ΔT ---
    sum_mdot_raw = 0.0
    sum_mdot_dT = 0.0
    for hp in hp_list:
        dT = float(hp.deltaTHeating)
        if dT <= 0:
            raise ValueError(f"HeatPump ID={hp.ID}: deltaTHeating must be > 0")
        mdot = float(hp.peakHeating_ground_load) / (cp * dT)
        sum_mdot_raw += mdot
        sum_mdot_dT += mdot * dT

    if sum_mdot_raw <= 0.0:
        raise ValueError("Sum of peak heating mass flow must be > 0")

    deltaT_sys_heat = float(sum_mdot_dT / sum_mdot_raw)
    scale_H = 0.0 if P_peak_div == 0.0 else (P_peak_eff / P_peak_div)
    aggregated_mdot_peak_heat_kg_s = float(scale_H * f * sum_mdot_raw)

    # --- Cooling detection ---
    P_peak_c_raw = float(sum(abs(float(hp.peakCooling_ground_load)) for hp in hp_list))
    if np.isclose(P_peak_c_raw, 0.0, atol=1.0):
        return GroundLoads(
            heating_ground_load_W=heating_ground_load_W,
            cooling_ground_load_W=None,
            has_cooling=False,
            peak_heating_h=peak_heating_h,
            peak_cooling_h=peak_cooling_h,
            deltaT_sys_heat=deltaT_sys_heat,
            deltaT_sys_cool=None,
            aggregated_mdot_peak_heat_kg_s=aggregated_mdot_peak_heat_kg_s,
            aggregated_mdot_peak_cool_kg_s=None,
        )

    # --- Cooling loads ---
    P_year_c = float(sum(abs(float(hp.annualCooling_ground_load)) for hp in hp_list))
    P_summ_c = float(sum(abs(float(hp.summerCooling_ground_load)) for hp in hp_list))
    P_peak_c_div = f * P_peak_c_raw
    if peak_cooling_h is None:
        P_peak_c_eff = P_summ_c
    else:
        P_peak_c_eff = _apply_peak_fraction(
            P_base=P_summ_c, P_peak=P_peak_c_div,
            mode=peak_fraction_cooling_mode, alpha=peak_fraction_cooling,
        )
    cooling_ground_load_W = np.asarray([P_year_c, P_summ_c, P_peak_c_eff], dtype=float)

    # --- Cooling peak flow + system ΔT ---
    sum_mdot_c_raw = 0.0
    sum_mdot_dT_c = 0.0
    for hp in hp_list:
        Pmag = abs(float(hp.peakCooling_ground_load))
        if Pmag <= 0.0:
            continue
        dT = float(hp.deltaTCooling)
        if dT <= 0:
            raise ValueError(
                f"HeatPump ID={hp.ID}: deltaTCooling must be > 0 when cooling is active"
            )
        mdot = Pmag / (cp * dT)
        sum_mdot_c_raw += mdot
        sum_mdot_dT_c += mdot * dT

    if sum_mdot_c_raw <= 0.0:
        # Loads exist but flows are inconsistent — return without cooling flows
        return GroundLoads(
            heating_ground_load_W=heating_ground_load_W,
            cooling_ground_load_W=cooling_ground_load_W,
            has_cooling=True,
            peak_heating_h=peak_heating_h,
            peak_cooling_h=peak_cooling_h,
            deltaT_sys_heat=deltaT_sys_heat,
            deltaT_sys_cool=None,
            aggregated_mdot_peak_heat_kg_s=aggregated_mdot_peak_heat_kg_s,
            aggregated_mdot_peak_cool_kg_s=None,
        )

    deltaT_sys_cool = float(sum_mdot_dT_c / sum_mdot_c_raw)
    scale_C = 0.0 if P_peak_c_div == 0.0 else (P_peak_c_eff / P_peak_c_div)
    aggregated_mdot_peak_cool_kg_s = float(scale_C * f * sum_mdot_c_raw)

    return GroundLoads(
        heating_ground_load_W=heating_ground_load_W,
        cooling_ground_load_W=cooling_ground_load_W,
        has_cooling=True,
        peak_heating_h=peak_heating_h,
        peak_cooling_h=peak_cooling_h,
        deltaT_sys_heat=deltaT_sys_heat,
        deltaT_sys_cool=deltaT_sys_cool,
        aggregated_mdot_peak_heat_kg_s=aggregated_mdot_peak_heat_kg_s,
        aggregated_mdot_peak_cool_kg_s=aggregated_mdot_peak_cool_kg_s,
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
    peak_heating_h: float = 4.0,
    peak_cooling_h: Optional[float] = 4.0,
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
    if peak_heating_h <= 0:
        raise ValueError("peak_heating_h must be > 0")
    if peak_cooling_h is not None and peak_cooling_h <= 0:
        raise ValueError("peak_cooling_h must be > 0 when provided")
    if li.cop_yearly_heating <= 0 or li.cop_winter_heating <= 0 or li.cop_peak_heating <= 0:
        raise ValueError("All COP values must be > 0")
    if li.deltaT_heating <= 0:
        raise ValueError("deltaT_heating must be > 0")

    P_ann_H = li.load_yearly_heating  * (1.0 - 1.0 / li.cop_yearly_heating)
    P_win_H = li.load_winter_heating  * (1.0 - 1.0 / li.cop_winter_heating)
    P_raw_H = li.load_daily_peak_heating * (1.0 - 1.0 / li.cop_peak_heating)
    S_H = f_peak_heating * diversity_factor_from_n_heat_pumps(li.n_consumers_heating)
    P_peak_H = P_raw_H * S_H

    heating_ground_load_W = np.array([P_ann_H, P_win_H, P_peak_H], dtype=float)
    deltaT_sys_heat = float(li.deltaT_heating)
    aggregated_mdot_peak_heat_kg_s = float(P_peak_H / (cp * li.deltaT_heating))

    if not li.has_cooling:
        return GroundLoads(
            heating_ground_load_W=heating_ground_load_W,
            cooling_ground_load_W=None,
            has_cooling=False,
            peak_heating_h=peak_heating_h,
            peak_cooling_h=peak_cooling_h,
            deltaT_sys_heat=deltaT_sys_heat,
            deltaT_sys_cool=None,
            aggregated_mdot_peak_heat_kg_s=aggregated_mdot_peak_heat_kg_s,
            aggregated_mdot_peak_cool_kg_s=None,
        )

    if li.eer_cooling <= 0:
        raise ValueError("eer_cooling must be > 0 when cooling is active")
    if li.deltaT_cooling <= 0:
        raise ValueError("deltaT_cooling must be > 0 when cooling is active")

    P_ann_C = li.load_yearly_cooling  * (1.0 + 1.0 / li.eer_cooling)
    P_sum_C = li.load_summer_cooling  * (1.0 + 1.0 / li.eer_cooling)
    P_raw_C = li.load_daily_peak_cooling * (1.0 + 1.0 / li.eer_cooling)
    S_C = f_peak_cooling * diversity_factor_from_n_heat_pumps(li.n_consumers_cooling)
    P_peak_C = P_raw_C * S_C

    return GroundLoads(
        heating_ground_load_W=heating_ground_load_W,
        cooling_ground_load_W=np.array([P_ann_C, P_sum_C, P_peak_C], dtype=float),
        has_cooling=True,
        peak_heating_h=peak_heating_h,
        peak_cooling_h=peak_cooling_h,
        deltaT_sys_heat=deltaT_sys_heat,
        deltaT_sys_cool=float(li.deltaT_cooling),
        aggregated_mdot_peak_heat_kg_s=aggregated_mdot_peak_heat_kg_s,
        aggregated_mdot_peak_cool_kg_s=float(P_peak_C / (cp * li.deltaT_cooling)),
    )
