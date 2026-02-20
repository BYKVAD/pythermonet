from __future__ import annotations
import numpy as np

SECONDS_IN_HOUR = 3600.0
SECONDS_IN_MONTH = 24.0 * (365.25 / 12.0) * SECONDS_IN_HOUR
SECONDS_IN_YEAR = 12.0 * SECONDS_IN_MONTH

def three_pulse_times_s(t_peak_h: float, *, years: float = 30.0, months: float = 3.0) -> np.ndarray:
    """
    Standard 3-puls tider: [years + months + peak, months + peak, peak] i sekunder.
    """
    if t_peak_h <= 0:
        raise ValueError("t_peak_h must be > 0")

    t_peak_s = float(t_peak_h) * SECONDS_IN_HOUR
    return np.asarray(
        [
            years * SECONDS_IN_YEAR + months * SECONDS_IN_MONTH + t_peak_s,
            months * SECONDS_IN_MONTH + t_peak_s,
            t_peak_s,
        ],
        dtype=float,
    )
