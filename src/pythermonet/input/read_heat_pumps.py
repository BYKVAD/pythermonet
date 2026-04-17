from __future__ import annotations

from pathlib import Path
#import numpy as np
import pandas as pd

from pythermonet.components.heat_pump import HeatPump
#from pythermonet.components.heat_pumps import HeatPumps
#from pythermonet.core.heat_carrier import HeatCarrier


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

# antager: HeatPump, _REQUIRED_COLS, _has_meaningful_cooling er defineret i samme modul

# antager: HeatPump, _REQUIRED_COLS, _has_meaningful_cooling er defineret i samme modul


def read_heat_pumps_tsv(df: pd.DataFrame) -> list[HeatPump]:
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Heat pump TSV missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    has_cooling = _has_meaningful_cooling(df)

    def _get_float(r: dict, col: str, hp_id: int) -> float:
        if col not in r:
            raise ValueError(
                f"Missing column '{col}' for Heat_pump_ID={hp_id}. "
                f"Available columns: {list(r.keys())}"
            )
        try:
            return float(r[col])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value in column '{col}' for Heat_pump_ID={hp_id}: {r[col]!r}") from None

    heat_pumps: list[HeatPump] = []

    for r in df.to_dict(orient="records"):
        hp_id = int(r["Heat_pump_ID"])

        # --- heating (building side) ---
        Qh_y = _get_float(r, "Yearly_heating_load_(W)", hp_id)
        Qh_w = _get_float(r, "Winter_heating_load_(W)", hp_id)
        Qh_p = _get_float(r, "Daily_heating_load_(W)", hp_id)

        COP_y = _get_float(r, "Year_COP", hp_id)
        COP_w = _get_float(r, "Winter_COP", hp_id)
        COP_p = _get_float(r, "Hour_COP", hp_id)

        dT_h = _get_float(r, "dT_HP_Heating", hp_id)

        if COP_y <= 0 or COP_w <= 0 or COP_p <= 0:
            raise ValueError(
                f"Invalid COP (<=0) for Heat_pump_ID={hp_id}: "
                f"Year_COP={COP_y}, Winter_COP={COP_w}, Hour_COP={COP_p}"
            )
        if dT_h <= 0:
            raise ValueError(f"Invalid dT_HP_Heating (<=0) for Heat_pump_ID={hp_id}: {dT_h}")

        # --- optional cooling ---
        if has_cooling:
            Qc_y = _get_float(r, "Yearly_cooling_load_(W)", hp_id)
            Qc_s = _get_float(r, "Summer_cooling_load_(W)", hp_id)
            Qc_p = _get_float(r, "Daily_cooling_load_(W)", hp_id)

            EER = _get_float(r, "EER", hp_id)
            dT_c = _get_float(r, "dT_HP_Cooling", hp_id)

            if Qc_p > 0:
                if EER <= 0:
                    raise ValueError(f"Invalid EER (<=0) for Heat_pump_ID={hp_id} with cooling load > 0: {EER}")
                if dT_c <= 0:
                    raise ValueError(
                        f"Invalid dT_HP_Cooling (<=0) for Heat_pump_ID={hp_id} with cooling load > 0: {dT_c}"
                    )
            else:
                # Normalisér “ingen cooling” for denne HP
                Qc_y = max(0.0, Qc_y)
                Qc_s = max(0.0, Qc_s)
                Qc_p = 0.0
                if EER <= 0:
                    EER = 0.0
                if dT_c <= 0:
                    dT_c = 0.0
        else:
            Qc_y = 0.0
            Qc_s = 0.0
            Qc_p = 0.0
            EER = 0.0
            dT_c = 0.0

        heat_pumps.append(
            HeatPump(
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
            )
        )

    return heat_pumps