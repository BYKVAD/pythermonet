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


def manipulate_lcoe_input(
    lcoe_model: LCOE,
    dims: dict[str, float | int],
    rated_power_W: float,
    capacity_factor: float | None = None,
) -> LCOE:
    """
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

    fake_hp_capex = CapexScheduleTS(name="Fake HP", cashflow_t0=fake_hp_capex_DKK(rated_power_W), cashflow_ts=[], residual_at_end=0.0)
    lcoe_model.capex_ts.append(fake_hp_capex)

    total_elec_MWh = float(dims["elec_consump"]) + fake_hp_annual_elec_MWh(rated_power_W, capacity_factor=capacity_factor)
    annual_heat_MWh = float(dims["heat_consump"])
    T = lcoe_model.years * lcoe_model.steps_per_year
    lcoe_model.loads_ts = LoadPathTS(
        lcoe_model.years,
        lcoe_model.steps_per_year,
        heat_MWh_ts=[annual_heat_MWh] * T,
        elec_heat_MWh_ts=[total_elec_MWh] * T,
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

    return lcoe_model


def set_init_lcoe() -> LCOE:
    capex_ts: List[CapexScheduleTS] = [
        CapexScheduleTS(name="Jordarbejde - hovedledningsnet",             cashflow_t0=136693.0,   cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Ledningsnet - hovedledninger og stik",       cashflow_t0=147188.0,   cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="El-forsyning til VP",                        cashflow_t0=123300.0,   cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Energimålere (15 á 3,120 kr/stk)",           cashflow_t0=46800.0,    cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Gravearbejde stikledninger (15 á 6,600)",    cashflow_t0=99000.0,    cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Investeringsbidrag (15 á -7,380)",           cashflow_t0=-110700.0,  cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Forhøjet investeringsbidrag (15 á -29,000)", cashflow_t0=-435000.0,  cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Stikledningsbidrag (15 á -18,800)",          cashflow_t0=-282000.0,  cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Byggemodningsbidrag (15 á -10,590.77)",      cashflow_t0=-275360.0,  cashflow_ts=[], residual_at_end=0.0),
        CapexScheduleTS(name="Varmepumper (15 á 58,867)",                  cashflow_t0=883005.0,   cashflow_ts=[], residual_at_end=0.0),
        # CapexScheduleTS(name="Boringer (6 ká 79,746)",                      cashflow_t0=478476.0,   cashflow_ts=[], residual_at_end=0.0),
    ]

    opex_f_ts: List[OpexFixedTS] = [
        OpexFixedTS(name="O&M contract", series=[37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0, 37320.0])
    ]

    opex_v_ts: List[OpexVariableTS] = [
        OpexVariableTS(name="Water make-up", unit="MWh_heat", rates_ts=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ]

    energy_price = EnergyPricePathTS(name="electricity", values_ts=[1004.46, 1004.46, 936.46, 856.46, 844.46, 844.46, 844.46, 844.46, 844.46, 844.46,
        844.46, 844.46, 844.46, 844.46, 844.46, 844.46, 844.46, 844.46, 844.46, 844.46])
    
    
    total_debt = 0
    for capex in capex_ts:
        total_debt += capex.cashflow_t0

    debt_ts = DebtScheduleTS(name="Fixed-rate loan", principal=total_debt, nominal_rate_ann=0.026,
                             years=LCOE.years, steps_per_year=LCOE.steps_per_year, inflation_ann=0.00, grace_steps=0, fees_t0=0)
    
    return LCOE(capex_ts=capex_ts, opex_fixed_ts=opex_f_ts, opex_variable_ts=opex_v_ts, 
                price_path_ts=energy_price, debt_ts=debt_ts)


