from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pythermonet.components.heat_pump import HeatPump
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.optimizer.heat_pump import create_heat_pump
from pythermonet.optimizer.configure_thermonet import execute_dimensioning
from pythermonet.optimizer.ashp_vs_gshp import monthly_elec_loads
from pythermonet.input.df_from_sources import df_from_csv
from lcc5gdhc.system_model import evaluate_project_ts
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.optimizer.configure_lcoe import (
    manipulate_lcoe_input_gshp,
    manipulate_lcoe_input_ashp,
    set_init_lcoe_gshp,
    set_init_lcoe_ashp,
    calc_borehole_cost_pr_m,
)
from scipy.optimize import minimize_scalar  # noqa: F401


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


_BASELINE_LCOE_GSHP = set_init_lcoe_gshp()
_BASELINE_LCOE_ASHP = set_init_lcoe_ashp()



def _setup_heatpump(rated_power_W: float, num_hp=15):
    list_of_num_hp = range(0,num_hp,1)
    hp_list = [
        create_heat_pump(rated_power_W=rated_power_W/15, hp_id=i + 1)
        for i in list_of_num_hp
    ]

    sizing = SizingParameters(time_horizon_years=30.0)

    return hp_list, sizing


def _setup_heatpump_from_file(
    n_hps: int,
    hp_file: Path | None = None,
    default_hp_load_W: float | None = None,
) -> tuple[list[HeatPump], SizingParameters]:
    """Build the per-building HP list for the N-buildings scenario.

    If `hp_file` is provided, read it (silkeborg-shape TSV) and keep the first
    `n_hps` rows in file order (IDs 1..n_hps). Otherwise fall back to
    identical fake heating HPs sized at `default_hp_load_W` each.
    """
    if n_hps < 1:
        raise ValueError("n_hps must be >= 1")

    if hp_file is not None:
        all_hps = read_heat_pumps_tsv(df_from_csv(hp_file, sep=r"\t+"))
        if n_hps > len(all_hps):
            raise ValueError(
                f"n_hps={n_hps} exceeds available HPs in file ({len(all_hps)})"
            )
        hp_list = all_hps[:n_hps]
    else:
        if default_hp_load_W is None:
            raise ValueError("default_hp_load_W is required when hp_file is None")
        hp_list = [
            create_heat_pump(rated_power_W=default_hp_load_W, hp_id=i + 1)
            for i in range(n_hps)
        ]

    sizing = SizingParameters(time_horizon_years=30.0)
    return hp_list, sizing

    


@dataclass(frozen=True)
class CostEvaluation:
    rated_power_W: float
    lcoe: int
    source_length_m: float
    num_boreholes: int
    borehole_capex_DKK: float
    elec: float

def evaluate_cost(
    rated_power_W: float,
    n_boreholes: int = 6,
    verbose: bool = True,
) -> dict:
    """
    Run dimensioning + LCOE for a given fake-HP rated power [W] and
    borefield size.

    Returns a dict with the LCOE (DKK/MWh_service) and the dimensioning
    outputs that drove it, so callers (e.g. sweeps/plots) can inspect both.
    """

    hp_list, sizing = _setup_heatpump(rated_power_W)
    hp_list_for_loads = list(hp_list)


    

    gshp_elec_loads = monthly_elec_loads(hp_list_for_loads, [3.2, 3.2, 3.4, 3.4, 3.4, 3.4, 3.5, 3.6, 3.6, 3.6, 3.5, 3.4])
    ashp_elec_loads = monthly_elec_loads(hp_list_for_loads, [2.3, 2.4, 2.8, 3.4, 4.0, 4.3, 4.5, 4.4, 4.0, 3.4, 2.9, 2.5])

    elec_demand = {"tot_gshp_load": sum(gshp_elec_loads), "tot_ashp_load": sum(ashp_elec_loads)}



    # Iterate over dims of electricity consumption
    # Collect results in datastructure
    # Return both structures

    som_struc = {}

    # for consumption in dims["elec_consump"]:
    #     single_dim = deepcopy(dims)
    #     single_dim["elec_consump"] = dims["elec_consump"][consumption]
    for demand in elec_demand:
        elec = elec_demand[demand]
        is_ashp = demand == "tot_ashp_load"

        if is_ashp:
            lcoe_model = manipulate_lcoe_input_ashp(deepcopy(_BASELINE_LCOE_ASHP), elec)
        else:
            dims = execute_dimensioning(
                hp_list=hp_list_for_loads,
                n_boreholes=n_boreholes,
                sizing=sizing
            )
            lcoe_model = manipulate_lcoe_input_gshp(
                deepcopy(_BASELINE_LCOE_GSHP), dims, rated_power_W, elec
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

        borehole_capex = 0.0 if is_ashp else calc_borehole_cost_pr_m(dims)

        cost = CostEvaluation(
            rated_power_W=rated_power_W,
            lcoe=res["NPV_total_cost"],
            source_length_m=0.0 if is_ashp else dims["source_length"],
            num_boreholes=0 if is_ashp else dims["num_boreholes"],
            borehole_capex_DKK=borehole_capex,
            elec=elec
        )

        som_struc[demand] = cost

    # borehole_energy_MWh = max(0.0, single_dim["building_ground_extraction_MWh"] - fake_hp_energy_MWh)

    return som_struc


def evaluate_cost_by_n_hps(
    n_hps: int,
    hp_file: Path | None = None,
    default_hp_load_W: float | None = None,
    n_boreholes: int = 6,
    verbose: bool = True,
) -> dict:
    """
    Evaluate GSHP + ASHP LCOE for a project of N buildings, each with its own
    heating demand from the HP file (or an identical fallback load). Baseline
    CAPEX scales by n_hps/15 in both variants; no fake supplementary HP.
    """
    hp_list, sizing = _setup_heatpump_from_file(n_hps, hp_file, default_hp_load_W)

    gshp_elec_loads = monthly_elec_loads(hp_list, [3.2, 3.2, 3.4, 3.4, 3.4, 3.4, 3.5, 3.6, 3.6, 3.6, 3.5, 3.4])
    ashp_elec_loads = monthly_elec_loads(hp_list, [2.3, 2.4, 2.8, 3.4, 4.0, 4.3, 4.5, 4.4, 4.0, 3.4, 2.9, 2.5])
    elec_demand = {"tot_gshp_load": sum(gshp_elec_loads), "tot_ashp_load": sum(ashp_elec_loads)}

    active_hp_ids = set(range(1, n_hps + 1))
    som_struc: dict = {}

    for demand, elec in elec_demand.items():
        is_ashp = demand == "tot_ashp_load"

        if is_ashp:
            lcoe_model = manipulate_lcoe_input_ashp(
                deepcopy(set_init_lcoe_ashp(n_hps=n_hps)), elec
            )
            dims = None
        else:
            dims = execute_dimensioning(
                hp_list=hp_list,
                n_boreholes=n_boreholes,
                sizing=sizing,
                active_hp_ids=active_hp_ids,
            )
            lcoe_model = manipulate_lcoe_input_gshp(
                deepcopy(set_init_lcoe_gshp(n_hps=n_hps)), dims, elec
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

        som_struc[demand] = CostEvaluation(
            rated_power_W=float(n_hps),
            lcoe=res["NPV_total_cost"],
            source_length_m=0.0 if is_ashp else dims["source_length"],
            num_boreholes=0 if is_ashp else dims["num_boreholes"],
            borehole_capex_DKK=0.0 if is_ashp else calc_borehole_cost_pr_m(dims),
            elec=elec
        )

    return som_struc


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
