"""
For each fake-HP rated power, plot the initial price of each heat source
against the annual energy it delivers to the brine network.

- Fake HP: CAPEX scales linearly with rated power; energy delivered scales
  linearly with rated power × annual capacity factor.
- Boreholes: CAPEX = drilled length × num_boreholes × DKK/m; energy
  delivered is the residual building ground-extraction load that the
  fake HP did not cover.

Reading the plot: the slope (DKK per MWh annual delivery) is the price
per unit of capacity for each source. A straight, low-slope line is a
cheap source; a steep line is an expensive source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from pythermonet.optimizer.main import evaluate_cost, evaluate_cost_by_n_hps
from pythermonet.optimizer.methods import find_root, fit_curves


RATED_POWER_W = [0.0, 10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0, 60_000.0, 75_000.0]

SILKEBORG_HP_FILE = Path(
    r"c:\software\pythermonetII\examples\silkeborg_bhe_full_dimensioning_heat\data\silkeborg_heat_pump_heat_only.dat"
)
N_HPS_GRID = range(1, 16)


def sweep() -> list[dict]:
    results = []
    for x in RATED_POWER_W:
        print(f"  rated={x/1000:>5.1f} kW ...", flush=True)
        try:
            res = evaluate_cost(x, verbose=False)
        except ValueError as e:
            print(f"    -> INFEASIBLE: {e}")
            continue
        results.append(res)
    return results


def plot_rated_vs_lcoe(rows: list[dict]) -> None:
    """Plot rated power [kW] on the x-axis vs LCOE on the y-axis, one line
    per demand case (GSHP, ASHP)."""
    rated_kW = [r["tot_gshp_load"].rated_power_W / 1000.0 for r in rows]
    gshp_lcoe = [r["tot_gshp_load"].lcoe for r in rows]
    ashp_lcoe = [r["tot_ashp_load"].lcoe for r in rows]

    _, ax = plt.subplots()
    ax.plot(rated_kW, gshp_lcoe, marker="o", label="GSHP")
    ax.plot(rated_kW, ashp_lcoe, marker="s", label="ASHP")
    ax.set_xlabel("Fake HP rated power [kW]")
    ax.set_ylabel("LCOE")
    ax.legend()
    ax.grid(True)
    plt.show()


def sweep_by_n_hps(
    n_hps_grid=N_HPS_GRID,
    hp_file: Path | None = SILKEBORG_HP_FILE,
    default_hp_load_W: float | None = None,
) -> list[dict]:
    results = []
    for n in n_hps_grid:
        print(f"  n_hps={n:>2} ...", flush=True)
        try:
            res = evaluate_cost_by_n_hps(
                n_hps=n,
                hp_file=hp_file,
                default_hp_load_W=default_hp_load_W,
                verbose=False,
            )
        except ValueError as e:
            print(f"    -> INFEASIBLE: {e}")
            continue
        results.append(res)
    return results


def extract_list_from_models(rows: list[dict]):
    n_hps = [int(r["tot_gshp_load"].rated_power_W) for r in rows]
    gshp_lcoe = [r["tot_gshp_load"].lcoe for r in rows]
    ashp_lcoe = [r["tot_ashp_load"].lcoe for r in rows]
    ashp_elec = [r["tot_ashp_load"].elec for r in rows]
    gshp_elec = [r["tot_gshp_load"].elec for r in rows]

    return ashp_elec, gshp_elec, n_hps, gshp_lcoe, ashp_lcoe


def plot_n_hps_vs_lcoe(n_hps, gshp_lcoe, ashp_lcoe) -> None:
    """Plot number of buildings on the x-axis vs project NPV cost on the
    y-axis, one line per demand case (GSHP, ASHP). CostEvaluation.rated_power_W
    carries n_hps in this mode."""

    _, ax = plt.subplots()
    ax.plot(n_hps, gshp_lcoe, marker="o", label="GSHP")
    ax.plot(n_hps, ashp_lcoe, marker="s", label="ASHP")
    ax.set_xlabel("Number of buildings")
    ax.set_ylabel("LCOE")
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_Wh_vs_lcoe(gshp_elec, ashp_elec, gshp_lcoe, ashp_lcoe) -> None:
    """Plot number of buildings on the x-axis vs project NPV cost on the
    y-axis, one line per demand case (GSHP, ASHP). CostEvaluation.rated_power_W
    carries n_hps in this mode."""

    _, ax = plt.subplots()
    ax.plot(gshp_elec, gshp_lcoe, marker="o", label="GSHP")
    ax.plot(ashp_elec, ashp_lcoe, marker="s", label="ASHP")
    ax.set_xlabel("watt*h")
    ax.set_ylabel("LCOE")
    ax.legend()
    ax.grid(True)
    plt.show()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # rows = sweep()
    # plot_rated_vs_lcoe(rows)

    n_rows = sweep_by_n_hps()
    ashp_elec, gshp_elec, n_hps, gshp_lcoe, ashp_lcoe = extract_list_from_models(n_rows)
    # gshp_model, ashp_model = fit_curves(n_hps, gshp_lcoe, ashp_lcoe)

    # find_root(gshp_model, ashp_model, min(n_hps), max(n_hps))
    plot_Wh_vs_lcoe(gshp_elec, ashp_elec, gshp_lcoe, ashp_lcoe)
    # plot_n_hps_vs_lcoe(n_hps, gshp_lcoe, ashp_lcoe)
    




if __name__ == "__main__":
    main()
