from __future__ import annotations

from pythermonet.components.heat_pump import HeatPump


FAKE_HP_COP = 2
FAKE_HP_CAPEX_DKK_PER_W = 5.0
FAKE_HP_ANNUAL_CAPACITY_FACTOR = 0.4
# Within-peak-season average / nameplate. Maps to winter for a building
# heating HP and to summer for an active injector.
FAKE_HP_PEAK_SEASON_CAPACITY_FACTOR = 0.6

_HEATING_DUMMY_COP = 3.5
_HEATING_DUMMY_DELTA_T = 1.0
_COOLING_DELTA_T = 3.0


def _resolve_cf(capacity_factor: float | None) -> float:
    return FAKE_HP_ANNUAL_CAPACITY_FACTOR if capacity_factor is None else float(capacity_factor)


def _resolve_peak_season_cf(peak_season_capacity_factor: float | None) -> float:
    return (FAKE_HP_PEAK_SEASON_CAPACITY_FACTOR
            if peak_season_capacity_factor is None
            else float(peak_season_capacity_factor))


def create_heat_pump(
    rated_power_W: float,
    hp_id: int = 999,
    capacity_factor: float | None = None,
    peak_season_capacity_factor: float | None = None,
    is_injector: bool = False,
) -> HeatPump:
    """
    Fake heat pump for the optimizer.

    is_injector=False (default): regular building heat pump that extracts
    heat from the brine. Populates the heating channel; the heating load
    drives borehole sizing via apply_annual_balance.

    is_injector=True: supplementary source that rejects heat into the
    brine via the cooling channel (active-cooling convention). Netted
    against heating extraction to shorten the boreholes.

    capacity_factor              : annual energy / nameplate energy.
    peak_season_capacity_factor  : within-peak-season average / nameplate.
                                   Winter for heating, summer for injection.
    """
    if rated_power_W < 0:
        raise ValueError("rated_power_W must be >= 0")

    cf = _resolve_cf(capacity_factor)
    season_cf = _resolve_peak_season_cf(peak_season_capacity_factor)
    annual_W = rated_power_W * cf
    season_W = rated_power_W * season_cf

    if is_injector:
        return HeatPump(
            ID=hp_id,
            annualHeatingLoad=0.0,
            winterHeatingLoad=0.0,
            peakHeatingLoad=0.0,
            annualSCOP=_HEATING_DUMMY_COP,
            winterSCOP=_HEATING_DUMMY_COP,
            peakCOP=_HEATING_DUMMY_COP,
            deltaTHeating=_HEATING_DUMMY_DELTA_T,
            annualCoolingLoad=annual_W,
            summerCoolingLoad=season_W,
            peakCoolingLoad=rated_power_W,
            EER=FAKE_HP_COP,
            deltaTCooling=_COOLING_DELTA_T,
        )

    return HeatPump(
        ID=hp_id,
        annualHeatingLoad=annual_W,
        winterHeatingLoad=season_W,
        peakHeatingLoad=rated_power_W,
        annualSCOP=_HEATING_DUMMY_COP,
        winterSCOP=_HEATING_DUMMY_COP,
        peakCOP=_HEATING_DUMMY_COP,
        deltaTHeating=_HEATING_DUMMY_DELTA_T,
        annualCoolingLoad=0.0,
        summerCoolingLoad=0.0,
        peakCoolingLoad=0.0,
        EER=0.0,
        deltaTCooling=0.0,
    )


def fake_hp_capex_DKK(rated_power_W: float) -> float:
    return rated_power_W * FAKE_HP_CAPEX_DKK_PER_W


def fake_hp_annual_elec_MWh(
    rated_power_W: float,
    capacity_factor: float | None = None,
) -> float:
    cf = _resolve_cf(capacity_factor)
    annual_W = rated_power_W * cf
    annual_MWh_heat = annual_W * 8760.0 / 1_000_000.0
    return annual_MWh_heat / FAKE_HP_COP
