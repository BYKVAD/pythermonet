from __future__ import annotations

from pythermonet.components.heat_pump import HeatPump


FAKE_HP_COP = 2
FAKE_HP_CAPEX_DKK_PER_W = 5.0
FAKE_HP_ANNUAL_CAPACITY_FACTOR = 0.4
# Default 0 = uniform year-round operation (no seasonal concentration).
# Set > 0 to model a HP that runs only in summer/shoulder months: same
# annual energy but concentrated into part of the year, which drives the
# summer brine-temperature constraint in the BHE sizer.
FAKE_HP_SUMMER_CAPACITY_FACTOR = 0.6

_HEATING_DUMMY_COP = 3.5
_HEATING_DUMMY_DELTA_T = 1.0
_COOLING_DELTA_T = 3.0


def _resolve_cf(capacity_factor: float | None) -> float:
    return FAKE_HP_ANNUAL_CAPACITY_FACTOR if capacity_factor is None else float(capacity_factor)


def _resolve_summer_cf(summer_capacity_factor: float | None) -> float:
    return (FAKE_HP_SUMMER_CAPACITY_FACTOR
            if summer_capacity_factor is None
            else float(summer_capacity_factor))


def build_fake_heat_pump(
    rated_power_W: float,
    hp_id: int = 999,
    capacity_factor: float | None = None,
    summer_capacity_factor: float | None = None,
) -> HeatPump:
    """
    Supplementary heat source injecting heat into the brine network.

    Modeled via the HeatPump cooling channel: active-cooling convention
    rejects heat (= heat injected into the brine), which propagates through
    ground_loads_from_heat_pumps and is netted against the annual heating
    extraction by apply_annual_balance, shortening the boreholes.

    capacity_factor          : annual energy / nameplate energy. Drives
                               annualCoolingLoad → regeneration and the
                               electricity bill.
    summer_capacity_factor   : within-summer average / nameplate. Drives
                               summerCoolingLoad → seasonal-peak BHE
                               constraint (T_brine_max_cool). Set > 0 to
                               model summer/shoulder-only operation.
    """
    if rated_power_W < 0:
        raise ValueError("rated_power_W must be >= 0")

    cf = _resolve_cf(capacity_factor)
    summer_cf = _resolve_summer_cf(summer_capacity_factor)
    annual_W = rated_power_W * cf
    summer_W = rated_power_W * summer_cf

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
        summerCoolingLoad=summer_W,
        peakCoolingLoad=rated_power_W,
        EER=FAKE_HP_COP,
        deltaTCooling=_COOLING_DELTA_T,
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
