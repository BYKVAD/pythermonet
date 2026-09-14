from __future__ import annotations

from dataclasses import dataclass

# Seasonal pulse duration: one quarter-year (~91.3-day heating/cooling season).
_SECONDS_IN_SEASON: float = 365.25 / 4.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class SizingParameters:
    """
    Controls the 3-pulse ground thermal sizing algorithm.

    Separates simulation/design configuration from the physical description
    of the heat pump population (HeatPumps / AggregatedHeatPumps).

    Parameters
    ----------
    thermal_dimensioning_lifetime : float
        Design lifetime — the annual pulse extends this far into the future.
    """

    thermal_dimensioning_lifetime: float  # [years]

    def __post_init__(self) -> None:
        if self.thermal_dimensioning_lifetime <= 0:
            raise ValueError(f"thermal_dimensioning_lifetime must be > 0. Got {self.thermal_dimensioning_lifetime}")

    def times_s_peak_heating(self, peak_hours_heating: float) -> list[float]:
        """
        3-pulse time vector for peak-heating sizing [s]: [t_peak, t_seasonal, t_annual].

        Parameters
        ----------
        peak_hours_heating : float
            Peak load pulse duration from HeatPumps [h].
        """
        t_peak = peak_hours_heating * 3600.0
        t_seasonal = t_peak + _SECONDS_IN_SEASON
        t_annual = t_seasonal + self.thermal_dimensioning_lifetime * 365.25 * 24.0 * 3600.0
        return [t_peak, t_seasonal, t_annual]

    def times_s_peak_cooling(self, peak_hours_cooling: float | None) -> list[float] | None:
        """
        3-pulse time vector for peak-cooling sizing [s]: [t_peak, t_seasonal, t_annual].
        Returns None when peak_hours_cooling is None (cooling peak pulse disabled).

        Parameters
        ----------
        peak_hours_cooling : float or None
            Peak load pulse duration from HeatPumps [h], or None.
        """
        if peak_hours_cooling is None:
            return None
        t_peak = peak_hours_cooling * 3600.0
        t_seasonal = t_peak + _SECONDS_IN_SEASON
        t_annual = t_seasonal + self.thermal_dimensioning_lifetime * 365.25 * 24.0 * 3600.0
        return [t_peak, t_seasonal, t_annual]
