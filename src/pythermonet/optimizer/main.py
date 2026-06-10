from copy import deepcopy

from pythermonet.optimizer.configure_thermonet import execute_dimensioning
from lcc5gdhc.system_model import evaluate_project_ts
from pythermonet.optimizer.configure_lcoe import (
    manipulate_lcoe_input,
    set_init_lcoe,
    calc_borehole_cost_pr_m,
)
from pythermonet.optimizer.fake_heat_pump import (
    fake_hp_capex_DKK,
    fake_hp_annual_elec_MWh,
    FAKE_HP_COP,
)

from scipy.optimize import linprog, minimize_scalar  # noqa: F401


def print_lcoe_breakdown(res: dict, currency: str):
    """Pretty LCOE breakdown table."""
    pv_s = res.get("PV_service_MWh", float("nan"))
    items = [
        ("CAPEX",               res.get("NPV_capex", 0.0)),
        ("OPEX fixed",          res.get("NPV_opex_fixed", 0.0)),
        ("Electricity (total)", res.get("NPV_elec_total", 0.0)),
        ("Other variable OPEX", res.get("NPV_opex_variable_other", 0.0)),
        ("Financing",           res.get("NPV_financing", 0.0)),
    ]
    subtotal = sum(v for _, v in items)
    total = res.get("NPV_total_cost", subtotal)

    def per_mwh(x):
        return (x / pv_s) if (isinstance(pv_s, (int, float)) and pv_s > 0) else float("nan")

    print("\n=== LCOE breakdown (NPV basis) ===")
    print(f"{'Component':<24} {'NPV ['+currency+']':>18} {'Unit cost ['+currency+'/MWh]':>24} {'Share':>10}")
    print("-" * 80)
    for name, val in items:
        ucost = per_mwh(val)
        share = (val / total) if total != 0 else float("nan")
        print(f"{name:<24} {val:>18,.2f} {ucost:>24,.2f} {share:>9.1%}")
    print("-" * 80)
    lcoe = res.get("LCOx_total_(currency_per_MWh_service)", float("nan"))
    print(f"{'TOTAL (LCOE)':<24} {total:>18,.2f} {lcoe:>24,.2f} {1.0:>9.1%}")


_BASELINE_LCOE = set_init_lcoe()


def evaluate_cost(
    rated_power_W: float,
    capacity_factor: float | None = None,
    summer_capacity_factor: float | None = None,
    n_boreholes: int = 6,
    verbose: bool = True,
) -> dict:
    """
    Run dimensioning + LCOE for a given fake-HP rated power [W] and
    borefield size.

    Returns a dict with the LCOE (DKK/MWh_service) and the dimensioning
    outputs that drove it, so callers (e.g. sweeps/plots) can inspect both.
    """
    dims = execute_dimensioning(
        rated_power_W,
        capacity_factor=capacity_factor,
        summer_capacity_factor=summer_capacity_factor,
        n_boreholes=n_boreholes,
    )

    lcoe_model = manipulate_lcoe_input(
        deepcopy(_BASELINE_LCOE), dims, rated_power_W, capacity_factor=capacity_factor
    )

    res = evaluate_project_ts(
        years=lcoe_model.years,
        steps_per_year=lcoe_model.steps_per_year,
        real_discount_rate=lcoe_model.r,
        capex_ts=lcoe_model.capex_ts,
        opex_fixed_ts=lcoe_model.opex_fixed_ts,
        opex_variable_ts=lcoe_model.opex_variable_ts,
        price_paths_ts={"electricity": lcoe_model.price_path_ts},
        loads_ts=lcoe_model.loads_ts,
        debt_ts=[lcoe_model.debt_ts],
    )

    if verbose:
        print_lcoe_breakdown(res, "DKK")

    fake_capex = fake_hp_capex_DKK(rated_power_W)
    borehole_capex = calc_borehole_cost_pr_m(dims)

    fake_hp_energy_MWh = fake_hp_annual_elec_MWh(rated_power_W, capacity_factor=capacity_factor) * FAKE_HP_COP
    borehole_energy_MWh = max(0.0, dims["building_ground_extraction_MWh"] - fake_hp_energy_MWh)

    return {
        "rated_power_W": rated_power_W,
        "capacity_factor": capacity_factor,
        "summer_capacity_factor": summer_capacity_factor,
        "lcoe_DKK_per_MWh": res["LCOx_total_(currency_per_MWh_service)"],
        "source_length_m": dims["source_length"],
        "num_boreholes": dims["num_boreholes"],
        "fake_hp_capex_DKK": fake_capex,
        "borehole_capex_DKK": borehole_capex,
        "fake_hp_energy_MWh": fake_hp_energy_MWh,
        "borehole_energy_MWh": borehole_energy_MWh,
    }


if __name__ == "__main__":
    # --- linprog scaffold (kept for reference) ---
    # LCOE is nonlinear in rated_power_W (borehole length is a nonlinear
    # function of the residual ground load), so linprog cannot find the true
    # minimum — it only picks a feasible corner. Left commented for reference.
    #
    # c = [1]
    # A_ub = [[-1]]
    # b_ub = [[-1000]]
    # bounds = [(1000, None)]
    # result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    # print("Optimal x:", result.x)
    # print("Optimal cost:", result.fun)
    # evaluate_cost(float(result.x[0]))

    # --- 2D optimization: integer n_boreholes (outer) + continuous
    #     rated_power_W (inner). For each candidate n, find the LCOE-minimizing
    #     fake-HP capacity with bounded Brent, and check the lower bound
    #     explicitly so monotone landscapes still report the true minimum.
    SEARCH_BOUNDS_W = (0.0, 100_000.0)
    N_BOREHOLES_GRID = range(2, 13)  # 2..12 boreholes

    def lcoe_for(rated_W: float, n: int) -> float:
        try:
            return evaluate_cost(rated_W, n_boreholes=n, verbose=False)["lcoe_DKK_per_MWh"]
        except ValueError:
            return float("inf")

    print(f"\n=== Optimizing over n_boreholes in {list(N_BOREHOLES_GRID)} ===")
    best = {"lcoe": float("inf"), "n": None, "rated_W": None}

    for n in N_BOREHOLES_GRID:
        lcoe_at_lower = lcoe_for(SEARCH_BOUNDS_W[0], n)
        interior = minimize_scalar(
            lambda x, n=n: lcoe_for(x, n),
            bounds=SEARCH_BOUNDS_W,
            method="bounded",
            options={"xatol": 100.0},
        )
        if interior.fun < lcoe_at_lower:
            x_opt_n, lcoe_n = float(interior.x), float(interior.fun)
        else:
            x_opt_n, lcoe_n = SEARCH_BOUNDS_W[0], lcoe_at_lower

        marker = ""
        if lcoe_n < best["lcoe"]:
            best = {"lcoe": lcoe_n, "n": n, "rated_W": x_opt_n}
            marker = "  <-- new best"
        print(f"  n={n:>2}  rated={x_opt_n/1000:>6.2f} kW  LCOE={lcoe_n:>8.2f} DKK/MWh{marker}")

    print(f"\n=== Optimum ===")
    print(f"n_boreholes      : {best['n']}")
    print(f"rated_power_W    : {best['rated_W']:,.0f}  ({best['rated_W']/1000:.2f} kW)")
    print(f"LCOE             : {best['lcoe']:.2f} DKK/MWh")

    print("\n--- LCOE breakdown at optimum ---")
    evaluate_cost(best["rated_W"], n_boreholes=best["n"])
