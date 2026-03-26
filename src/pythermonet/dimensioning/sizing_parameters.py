from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SizingParameters:
    """
    Controls the 3-pulse ground thermal sizing algorithm.

    Separates simulation/design configuration from the physical description
    of the heat pump population (HeatPumps / AggregatedHeatPumps).

    Parameters
    ----------
    time_horizon_years : float
        Design lifetime — the annual pulse extends this far into the future.
    """

    time_horizon_years: float = 30.0

    # Seasonal pulse duration: one quarter-year (≈ 91.3 days = 3-month heating/cooling season).
    _SEASONAL_S: float = field(default=365.25 / 4.0 * 24.0 * 3600.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.time_horizon_years <= 0:
            raise ValueError(f"time_horizon_years must be > 0. Got {self.time_horizon_years}")

    def times_s_peak_heating(self, peak_heating_h: float) -> list[float]:
        """
        3-pulse time vector for peak-heating sizing [s]: [t_peak, t_seasonal, t_annual].

        Parameters
        ----------
        peak_heating_h : float
            Peak load pulse duration from HeatPumps [h].
        """
        t_peak = peak_heating_h * 3600.0
        t_seasonal = t_peak + self._SEASONAL_S
        t_annual = t_seasonal + self.time_horizon_years * 365.25 * 24.0 * 3600.0
        return [t_peak, t_seasonal, t_annual]

    def times_s_peak_cooling(self, peak_cooling_h: float | None) -> list[float] | None:
        """
        3-pulse time vector for peak-cooling sizing [s]: [t_peak, t_seasonal, t_annual].
        Returns None when peak_cooling_h is None (cooling peak pulse disabled).

        Parameters
        ----------
        peak_cooling_h : float or None
            Peak load pulse duration from HeatPumps [h], or None.
        """
        if peak_cooling_h is None:
            return None
        t_peak = peak_cooling_h * 3600.0
        t_seasonal = t_peak + self._SEASONAL_S
        t_annual = t_seasonal + self.time_horizon_years * 365.25 * 24.0 * 3600.0
        return [t_peak, t_seasonal, t_annual]
