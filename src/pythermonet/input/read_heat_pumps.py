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


# from __future__ import annotations

# from pathlib import Path

# import pandas as pd

# from pythermonet.core.heat_carrier import HeatCarrier
# from pythermonet.components.heat_pumps import HeatPumps
# from pythermonet.components.heat_pump import HeatPump

# Forudsætter at disse findes i din fil
# _REQUIRED_COLS = [...]
# def _has_meaningful_cooling(df: pd.DataFrame) -> bool: ...


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
        hp_id = int(r["Heat_pump_ID"])

        # --- raw loads (building side) ---
        Qh_y = float(r["Yearly_heating_load_(W)"])
        Qh_w = float(r["Winter_heating_load_(W)"])
        Qh_p = float(r["Daily_heating_load_(W)"])

        COP_y = float(r["Year_COP"])
        COP_w = float(r["Winter_COP"])
        COP_p = float(r["Hour_COP"])

        dT_h = float(r["dT_HP_Heating"])

        # --- fail-fast validation (heating) ---
        if COP_y <= 0 or COP_w <= 0 or COP_p <= 0:
            raise ValueError(
                f"Invalid COP (<=0) for Heat_pump_ID={hp_id}: "
                f"Year_COP={COP_y}, Winter_COP={COP_w}, Hour_COP={COP_p}"
            )
        if dT_h <= 0:
            raise ValueError(f"Invalid dT_HP_Heating (<=0) for Heat_pump_ID={hp_id}: {dT_h}")

        # --- optional cooling ---
        if has_cooling:
            Qc_y = float(r["Yearly_cooling_load_(W)"])
            Qc_s = float(r["Summer_cooling_load_(W)"])
            Qc_p = float(r["Daily_cooling_load_(W)"])

            EER = float(r["EER"])
            dT_c = float(r["dT_HP_Cooling"])

            # Hvis der er en kølekolonne, men denne HP reelt ikke køler,
            # accepter 0 og lad HeatPump.__post_init__ håndtere det.
            if Qc_p > 0:
                if EER <= 0:
                    raise ValueError(f"Invalid EER (<=0) for Heat_pump_ID={hp_id} with cooling load > 0: {EER}")
                if dT_c <= 0:
                    raise ValueError(f"Invalid dT_HP_Cooling (<=0) for Heat_pump_ID={hp_id} with cooling load > 0: {dT_c}")
        else:
            Qc_y = 0.0
            Qc_s = 0.0
            Qc_p = 0.0
            EER = 0.0
            dT_c = 0.0

        hp = HeatPump(
            ID=hp_id,

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
        )

        heat_pumps.append(hp)

    return heat_pumps
