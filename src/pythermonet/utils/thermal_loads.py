from __future__ import annotations

import numpy as np
import pandas as pd


def source_loads_all_timescales(
    row: pd.Series,
    heating: bool,
) -> np.ndarray:
    """
    Compute source-side thermal loads for one heat pump on three timescales:
    yearly, seasonal, and peak.

    Returns
    -------
    np.ndarray
        Shape (1, 3):
        [ yearly, seasonal, peak ]  [W]
        Positive = heat extracted from ground
    """

    if heating:
        # Loads on building side
        Q_year = float(row["Yearly_heating_load_(W)"])
        Q_season = float(row["Winter_heating_load_(W)"])
        Q_peak = float(row["Daily_heating_load_(W)"])

        COP_year = float(row["Year_COP"])
        COP_season = float(row["Winter_COP"])
        COP_peak = float(row["Hour_COP"])

        P_source = np.array([
            Q_year * (1.0 / COP_year - 1.0),
            Q_season * (1.0 / COP_season - 1.0),
            Q_peak * (1.0 / COP_peak - 1.0),
        ])

    else:
        # Cooling
        Q_year = float(row["Yearly_cooling_load_(W)"])
        Q_season = float(row["Summer_cooling_load_(W)"])
        Q_peak = float(row["Daily_cooling_load_(W)"])

        EER = float(row["EER"])

        P_source = np.array([
            Q_year * (1.0 + 1.0 / EER),
            Q_season * (1.0 + 1.0 / EER),
            Q_peak * (1.0 + 1.0 / EER),
        ])

        # Cooling injects heat into ground → negative by convention
        P_source *= -1.0

    # legacy shape: (n_HP, 3)
    return P_source.reshape(1, 3)