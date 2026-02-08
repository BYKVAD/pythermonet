from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pythermonet.components.heat_pump import HeatPump
from pythermonet.components.heat_pumps import HeatPumps
from pythermonet.core.heat_carrier import HeatCarrier


_REQUIRED_COLS = [
    "Heat_pump_ID",
    "Yearly_heating_load_(W)",
    "Winter_heating_load_(W)",
    "Daily_heating_load_(W)",
    "Year_COP",
    "Winter_COP",
    "Hour_COP",
    "dT_HP_Heating",
]

_OPTIONAL_COOLING_COLS = [
    "Yearly_cooling_load_(W)",
    "Summer_cooling_load_(W)",
    "Daily_cooling_load_(W)",
    "EER",
    "dT_HP_Cooling",
]


def _has_meaningful_cooling(df: pd.DataFrame) -> bool:
    if not all(c in df.columns for c in _OPTIONAL_COOLING_COLS):
        return False
    dT = pd.to_numeric(df["dT_HP_Cooling"], errors="coerce")
    q = pd.to_numeric(df["Daily_cooling_load_(W)"], errors="coerce")
    eer = pd.to_numeric(df["EER"], errors="coerce")
    return bool((dT.notna() & (dT != 0)).any() and (q.notna() & (q != 0)).any() and (eer.notna() & (eer > 0)).any())


def read_heat_pumps_tsv(
    path: str | Path,
    *,
    source_heat_carrier: HeatCarrier,
) -> HeatPumps:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Heat pump TSV not found: {p}")

    df = pd.read_csv(p, sep=r"\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Heat pump TSV missing required columns: {missing}")

    has_cooling = _has_meaningful_cooling(df)

    heat_pumps: list[HeatPump] = []

    for r in df.to_dict(orient="records"):
        # --- raw loads (building side) ---
        Qh_y = float(r["Yearly_heating_load_(W)"])
        Qh_w = float(r["Winter_heating_load_(W)"])
        Qh_p = float(r["Daily_heating_load_(W)"])

        COP_y = float(r["Year_COP"])
        COP_w = float(r["Winter_COP"])
        COP_p = float(r["Hour_COP"])

        dT_h = float(r["dT_HP_Heating"])

        # --- ground loads (heating): + extracted ---
        # P_ground = Q_load * (1 - 1/COP)
        Gh_y = Qh_y * (1.0 - 1.0 / COP_y)
        Gh_w = Qh_w * (1.0 - 1.0 / COP_w)
        Gh_p = Qh_p * (1.0 - 1.0 / COP_p)

        # --- optional cooling ---
        Qc_y = float(r["Yearly_cooling_load_(W)"]) if has_cooling else 0.0
        Qc_s = float(r["Summer_cooling_load_(W)"]) if has_cooling else 0.0
        Qc_p = float(r["Daily_cooling_load_(W)"]) if has_cooling else 0.0

        EER = float(r["EER"]) if has_cooling else 0.0
        dT_c = float(r["dT_HP_Cooling"]) if has_cooling else 0.0

        # ground loads (cooling): - injected
        # P_rejected = Q_cool * (1 + 1/EER)  then apply minus sign
        if has_cooling:
            Gc_y = -Qc_y * (1.0 + 1.0 / EER)
            Gc_s = -Qc_s * (1.0 + 1.0 / EER)
            Gc_p = -Qc_p * (1.0 + 1.0 / EER)
        else:
            Gc_y = Gc_s = Gc_p = 0.0

        hp = HeatPump(
            ID=int(r["Heat_pump_ID"]),

            annualHeatingLoad=Qh_y,
            winterHeatingLoad=Qh_w,
            peakHeatingLoad=Qh_p,

            annualSCOP=COP_y,
            winterSCOP=COP_w,
            peakCOP=COP_p,

            deltaTHeating=dT_h,

            annualCoolingLoad=Qc_y,
            summerCoolingLoad=Qc_s,
            peakCoolingLoad=Qc_p,

            EER=EER,
            deltaTCooling=dT_c,

            sourceHeatCarrier=source_heat_carrier,

            annualHeating_ground_load=Gh_y,
            winterHeating_ground_load=Gh_w,
            peakHeating_ground_load=Gh_p,

            annualCooling_ground_load=Gc_y,
            summerCooling_ground_load=Gc_s,
            peakCooling_ground_load=Gc_p,
        )
        heat_pumps.append(hp)

    return HeatPumps(heatPumpList=heat_pumps)