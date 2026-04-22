from typing import List

from pythermonet.optimizer.lcoe_model import LCOE
from lcc5gdhc.cost_items import (
    CapexScheduleTS,
    OpexFixedTS,
    OpexVariableTS,
    EnergyPricePathTS,
    LoadPathTS,
    DebtScheduleTS
)

def manipulate_lcoe_input():
    # calc LoadPathTS her
    return


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
        CapexScheduleTS(name="Boringer (6 á 79,746)",                      cashflow_t0=478476.0,   cashflow_ts=[], residual_at_end=0.0),
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


