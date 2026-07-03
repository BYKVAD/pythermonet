from typing import List

from pythermonet.optimizer.lcoe_model import LCOE
from pythermonet.optimizer.fake_heat_pump import (
    fake_hp_capex_DKK,
    fake_hp_annual_elec_MWh,
)
from lcc5gdhc.cost_items import (
    CapexScheduleTS,
    OpexFixedTS,
    OpexVariableTS,
    EnergyPricePathTS,
    LoadPathTS,
    DebtScheduleTS
)


def calc_borehole_cost_pr_m(dims: dict[str, float | int]) -> int | float:
    """
    Dims params:
        {
            "source_length": float,
            "num_boreholes": int ,
            "elec_consump": float
        }
    """
    borehole_length = dims["source_length"]
    num_boreholes = dims["num_boreholes"]

    tot_length = borehole_length*num_boreholes

    price_pr_m = 670

    tot_price = tot_length*price_pr_m
    return tot_price


def _set_loads_and_rebuild_debt(lcoe_model: LCOE, elec: float) -> None:
    T = lcoe_model.years * lcoe_model.steps_per_year
    lcoe_model.loads_ts = LoadPathTS(
        lcoe_model.years,
        lcoe_model.steps_per_year,
        elec_heat_MWh_ts=[elec] * T,
    )

    total_capex = sum(c.cashflow_t0 for c in lcoe_model.capex_ts)
    lcoe_model.debt_ts = DebtScheduleTS(
        name="Fixed-rate loan",
        principal=total_capex,
        nominal_rate_ann=0.026,
        years=lcoe_model.years,
        steps_per_year=lcoe_model.steps_per_year,
        inflation_ann=0.0,
        grace_steps=0,
        fees_t0=0,
    )


def manipulate_lcoe_input_gshp(
    lcoe_model: LCOE,
    dims: dict[str, float | int],
    elec: float
) -> LCOE:
    """
    GSHP case: adds borehole CAPEX and the fake supplementary HP on top
    of the brine-net baseline.

    Dims params:
        {
            "source_length": float,
            "num_boreholes": int,
            "elec_consump": float   # building HPs' annual electricity [MWh]
        }
    rated_power_W: fake heat-injection HP capacity [W] (decision variable).
    """

    boreholes = CapexScheduleTS(name="boreholes", cashflow_t0=calc_borehole_cost_pr_m(dims), cashflow_ts=[], residual_at_end=0.0)
    lcoe_model.capex_ts.append(boreholes)

    # fake_hp_capex = CapexScheduleTS(name="Fake HP", cashflow_t0=fake_hp_capex_DKK(rated_power_W), cashflow_ts=[], residual_at_end=0.0)
    # lcoe_model.capex_ts.append(fake_hp_capex)

    _set_loads_and_rebuild_debt(lcoe_model, elec)
    return lcoe_model


def manipulate_lcoe_input_ashp(lcoe_model: LCOE, elec: float) -> LCOE:
    """
    ASHP case: baseline already excludes brine-net earthworks/pipes and the
    various investeringsbidrag; only the load path and debt need populating.
    """
    _set_loads_and_rebuild_debt(lcoe_model, elec)
    return lcoe_model


# Cost of the 15 building HPs in the GSHP baseline. The ASHP baseline scales
# this up by _ASHP_HP_COST_MULTIPLIER to reflect the ~30% higher unit cost of
# an air-source unit.
_GSHP_HP_CAPEX_DKK = 883005.0
_ASHP_HP_COST_MULTIPLIER = 1.15


def _shared_opex_and_price() -> tuple[list[OpexFixedTS], list[OpexVariableTS], EnergyPricePathTS]:
    opex_f_ts = [
        OpexFixedTS(name="O&M contract", series=[37320.0] * 20)
    ]
    opex_v_ts = [
        OpexVariableTS(name="Water make-up", unit="MWh_heat", rates_ts=[0.0] * 20)
    ]
    energy_price = EnergyPricePathTS(
        name="electricity",
        values_ts=[1004.46, 1004.46, 936.46, 856.46] + [844.46] * 16,
    )
    return opex_f_ts, opex_v_ts, energy_price


def _scale_capex(items: List[CapexScheduleTS], factor: float) -> List[CapexScheduleTS]:
    return [
        CapexScheduleTS(
            name=item.name,
            cashflow_t0=item.cashflow_t0 * factor,
            cashflow_ts=list(item.cashflow_ts),
            residual_at_end=item.residual_at_end,
        )
        for item in items
    ]


def _build_lcoe_from_capex(capex_ts: List[CapexScheduleTS]) -> LCOE:
    opex_f_ts, opex_v_ts, energy_price = _shared_opex_and_price()
    total_debt = sum(c.cashflow_t0 for c in capex_ts)
    debt_ts = DebtScheduleTS(
        name="Fixed-rate loan",
        principal=total_debt,
        nominal_rate_ann=0.026,
        years=LCOE.years,
        steps_per_year=LCOE.steps_per_year,
        inflation_ann=0.00,
        grace_steps=0,
        fees_t0=0,
    )
    return LCOE(
        capex_ts=capex_ts,
        opex_fixed_ts=opex_f_ts,
        opex_variable_ts=opex_v_ts,
        price_path_ts=energy_price,
        debt_ts=debt_ts,
    )


_MAX_N_HPS = 15


def set_init_lcoe_gshp(n_hps: int = _MAX_N_HPS) -> LCOE:
    capex_ts: List[CapexScheduleTS] = [
        CapexScheduleTS(name="Jordarbejde - hovedledningsnet",             cashflow_t0=136693.0,           cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Ledningsnet - hovedledninger og stik",       cashflow_t0=147188.0,           cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="El-forsyning til VP",                        cashflow_t0=123300.0,           cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Energimålere (15 á 3,120 kr/stk)",           cashflow_t0=46800.0,            cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Gravearbejde stikledninger (15 á 6,600)",    cashflow_t0=99000.0,            cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Investeringsbidrag (15 á -7,380)",           cashflow_t0=-110700.0,          cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Forhøjet investeringsbidrag (15 á -29,000)", cashflow_t0=-435000.0,          cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Stikledningsbidrag (15 á -18,800)",          cashflow_t0=-282000.0,          cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Byggemodningsbidrag (15 á -10,590.77)",      cashflow_t0=-275360.0,          cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Varmepumper (15 á 58,867)",                  cashflow_t0=_GSHP_HP_CAPEX_DKK, cashflow_ts=[], residual_at_end=0.0),
    ]
    return _build_lcoe_from_capex(_scale_capex(capex_ts, n_hps / _MAX_N_HPS))


def set_init_lcoe_ashp(n_hps: int = _MAX_N_HPS) -> LCOE:
    # No brine-net earthworks/pipes/service-line excavation, and none of the
    # bidrag subsidies (those are tied to the brine-net connection). HPs are
    # scaled up to reflect the higher unit cost of an air-source unit.
    ashp_hp_capex = _GSHP_HP_CAPEX_DKK * _ASHP_HP_COST_MULTIPLIER
    markup_pct = int(round((_ASHP_HP_COST_MULTIPLIER - 1.0) * 100))
    capex_ts: List[CapexScheduleTS] = [
        CapexScheduleTS(name="El-forsyning til VP",                        cashflow_t0=123300.0,      cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Energimålere (15 á 3,120 kr/stk)",           cashflow_t0=46800.0,       cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name=f"Varmepumper ASHP (15 á +{markup_pct}%)",    cashflow_t0=ashp_hp_capex, cashflow_ts=[], residual_at_end=0.0),
    ]
    return _build_lcoe_from_capex(_scale_capex(capex_ts, n_hps / _MAX_N_HPS))


