from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pythermonet.components.heat_pumps import HeatPumps
from pythermonet.components.aggregated_heat_pump import AggregatedHeatPump
from pythermonet.core.heat_carrier import HeatCarrier


def _safe_divide(num: float, den: float, fallback: float = 0.0) -> float:
    return float(num / den) if den > 0 else float(fallback)


def _count_active(values: np.ndarray) -> int:
    """Active = finite and non-zero."""
    values = np.asarray(values, dtype=float)
    return int(np.sum(np.isfinite(values) & (values != 0.0)))


def aggregated_load_from_heatpumps(
    heat_pumps: HeatPumps,
    heat_carrier: HeatCarrier,
) -> AggregatedHeatPump:
    """
    Aggregér HeatPumps -> AggregatedHeatPump.

    - Qdim beregnes ud fra peak load og deltaT pr. HP
    - Outlet temperatur er flow-vægtet
    - Diversity: S = fractionPeak * (0.62 + 0.38/n_active)
      (bruger HeatPumps.fractionPeakHeating/Cooling hvis sat, ellers 1.0)
    """
    if heat_carrier.rho <= 0 or heat_carrier.c <= 0:
        raise ValueError("HeatCarrier properties must be positive: rho > 0 and c > 0.")

    hplist = heat_pumps.heatPumpList
    if not hplist:
        raise ValueError("HeatPumps.heatPumpList is empty.")

    # -------- Heating arrays --------
    Ti_H = float(heat_pumps.TInletHeating[0]) if heat_pumps.TInletHeating else 0.0  # fallback
    peak_H = np.array([hp.peakHeatingLoad for hp in hplist], dtype=float)
    dT_H = np.array([hp.deltaTHeating for hp in hplist], dtype=float)

    if np.any(dT_H <= 0):
        raise ValueError("All deltaTHeating must be > 0.")

    # peak volumetric flow per HP (m3/s)
    q_H = peak_H / dT_H / heat_carrier.rho / heat_carrier.c
    q_H = np.where(np.isfinite(q_H), q_H, 0.0)

    n_active_H = _count_active(peak_H)
    if n_active_H <= 0:
        raise ValueError("No active heating consumers found (peakHeatingLoad all zero/NaN).")

    f_peak_H = float(heat_pumps.fractionPeakHeating[0]) if heat_pumps.fractionPeakHeating else 1.0
    S_H = f_peak_H * (0.62 + 0.38 / n_active_H)

    sum_q_H = float(np.sum(q_H))
    # outlet (heating): colder than inlet
    To_H = Ti_H - _safe_divide(float(np.sum(q_H * dT_H)), sum_q_H, 0.0)

    # -------- Cooling arrays (optional) --------
    peak_C = np.array([hp.peakCoolingLoad for hp in hplist], dtype=float)
    dT_C = np.array([hp.deltaTCooling for hp in hplist], dtype=float)

    has_cooling = bool(np.any(np.isfinite(peak_C) & (peak_C != 0.0))) and bool(np.any(dT_C > 0))

    Ti_C = float(heat_pumps.TInletCooling[0]) if heat_pumps.TInletCooling else 0.0  # fallback
    To_C = Ti_C
    sum_q_C = 0.0
    S_C = 0.0

    if has_cooling:
        if np.any((peak_C != 0.0) & (dT_C <= 0)):
            raise ValueError("deltaTCooling must be > 0 for heat pumps with non-zero peakCoolingLoad.")

        q_C = np.where(
            (np.isfinite(peak_C) & (peak_C != 0.0)),
            peak_C / np.maximum(dT_C, 1e-12) / heat_carrier.rho / heat_carrier.c,
            0.0,
        )
        q_C = np.where(np.isfinite(q_C), q_C, 0.0)

        n_active_C = _count_active(peak_C)
        f_peak_C = float(heat_pumps.fractionPeakCooling[0]) if heat_pumps.fractionPeakCooling else 1.0
        S_C = f_peak_C * (0.62 + 0.38 / n_active_C) if n_active_C > 0 else 0.0

        sum_q_C = float(np.sum(q_C))
        # outlet (cooling): warmer than inlet
        To_C = Ti_C + _safe_divide(float(np.sum(q_C * dT_C)), sum_q_C, 0.0)

    # Byg AggregatedHeatPump (tilpas feltnavne til din class)
    agg = AggregatedHeatPump(
        Ti_H=Ti_H,
        To_H=To_H,
        Qdim_H=sum_q_H * S_H,
        S_H=S_H,
        has_cooling=has_cooling,
    )

    if has_cooling:
        agg.Ti_C = Ti_C
        agg.To_C = To_C
        agg.Qdim_C = sum_q_C * S_C
        agg.S_C = S_C

    return agg