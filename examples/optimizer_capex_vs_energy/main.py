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

import matplotlib.pyplot as plt

from pythermonet.optimizer.main import evaluate_cost
from pythermonet.optimizer.configure_lcoe import set_init_lcoe
from pythermonet.optimizer.fake_heat_pump import FAKE_HP_COP


RATED_POWER_W = [0.0, 10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0, 60_000.0, 75_000.0]


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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    rows = sweep()

    rated_kW = [r["rated_power_W"] / 1000.0 for r in rows]
    fake_capex = [r["fake_hp_capex_DKK"] for r in rows]
    bore_capex = [r["borehole_capex_DKK"] for r in rows]
    fake_E = [r["fake_hp_energy_MWh"] for r in rows]
    bore_E = [r["borehole_energy_MWh"] for r in rows]

    # Unit cost of capacity: DKK of initial CAPEX per MWh/year of delivery.
    # Skip rows where energy delivered is ~0 (division blows up and the
    # point isn't physically meaningful).
    EPS_MWh = 0.1

    def unit_costs(capex, energy):
        out = []
        for C, E in zip(capex, energy):
            out.append(C / E if E > EPS_MWh else float("nan"))
        return out

    fake_unit = unit_costs(fake_capex, fake_E)
    bore_unit = unit_costs(bore_capex, bore_E)

    # Lifetime electricity NPV per MWh/yr of source delivery, using the
    # real 20-year electricity price path and discount rate from the LCOE
    # baseline. Convention matches lcc5gdhc: cashflow at end of each step.
    base = set_init_lcoe()
    prices = base.price_path_ts.values_ts
    r = base.r
    npv_factor_per_MWh_yr = sum(p / (1 + r) ** (t + 1) for t, p in enumerate(prices))

    # Per MWh of brine heat delivered:
    #   - Boreholes: no own electricity (passive ground source) → 0 OPEX.
    #   - Fake HP:   own electricity = 1 / FAKE_HP_COP MWh per MWh brine.
    fake_opex_npv = [(E / FAKE_HP_COP) * npv_factor_per_MWh_yr for E in fake_E]
    bore_opex_npv = [0.0 for _ in bore_E]

    fake_lifetime = [c + o for c, o in zip(fake_capex, fake_opex_npv)]
    bore_lifetime = [c + o for c, o in zip(bore_capex, bore_opex_npv)]

    fig, (ax_xy, ax_x, ax_life) = plt.subplots(3, 1, figsize=(10, 12))

    ax_xy.plot(fake_unit, fake_E, marker="o", color="tab:orange",
               label="Fake HP")
    ax_xy.plot(bore_unit, bore_E, marker="s", color="tab:blue",
               label="Boreholes")
    for u, E, kW in zip(fake_unit, fake_E, rated_kW):
        if u == u:  # skip NaN
            ax_xy.annotate(f"{kW:.0f} kW", (u, E), fontsize=8,
                           textcoords="offset points", xytext=(4, 4),
                           color="tab:orange")
    for u, E, kW in zip(bore_unit, bore_E, rated_kW):
        if u == u:
            ax_xy.annotate(f"{kW:.0f} kW", (u, E), fontsize=8,
                           textcoords="offset points", xytext=(4, -10),
                           color="tab:blue")
    ax_xy.set_xlabel("Unit cost of capacity [DKK per MWh/year delivered]")
    ax_xy.set_ylabel("Annual heat delivered to brine [MWh/year]")
    ax_xy.set_title("Unit cost vs annual energy delivered (Silkeborg)")
    ax_xy.set_xscale("log")
    ax_xy.grid(True, which="both", alpha=0.3)
    ax_xy.legend()

    ax_x.plot(rated_kW, fake_capex, marker="o", color="tab:orange",
              label="Fake HP CAPEX")
    ax_x.plot(rated_kW, bore_capex, marker="s", color="tab:blue",
              label="Boreholes CAPEX")
    ax_x.plot(rated_kW, [a + b for a, b in zip(fake_capex, bore_capex)],
              marker="^", color="black", linestyle="--",
              label="Combined source CAPEX")
    ax_x.set_xlabel("Fake-HP rated power [kW]")
    ax_x.set_ylabel("Initial CAPEX [DKK]")
    ax_x.set_title("How the CAPEX split moves as we shift load to the fake HP")
    ax_x.grid(True, alpha=0.3)
    ax_x.legend()

    # CAPEX-only baseline (dashed) + CAPEX + 20-year electricity NPV (solid).
    # The gap between dashed and solid is the "expensive-to-run" tax.
    ax_life.plot(fake_E, fake_capex, marker="o", color="tab:orange",
                 linestyle=":", alpha=0.5, label="Fake HP — CAPEX only")
    ax_life.plot(fake_E, fake_lifetime, marker="o", color="tab:orange",
                 label="Fake HP — CAPEX + 20-yr elec NPV")
    ax_life.plot(bore_E, bore_capex, marker="s", color="tab:blue",
                 linestyle=":", alpha=0.5, label="Boreholes — CAPEX only")
    ax_life.plot(bore_E, bore_lifetime, marker="s", color="tab:blue",
                 label="Boreholes — CAPEX + 20-yr elec NPV")
    for E, C, kW in zip(fake_E, fake_lifetime, rated_kW):
        ax_life.annotate(f"{kW:.0f} kW", (E, C), fontsize=8,
                         textcoords="offset points", xytext=(4, 4),
                         color="tab:orange")
    for E, C, kW in zip(bore_E, bore_lifetime, rated_kW):
        ax_life.annotate(f"{kW:.0f} kW", (E, C), fontsize=8,
                         textcoords="offset points", xytext=(4, -10),
                         color="tab:blue")
    ax_life.set_xlabel("Annual heat delivered to brine [MWh/year]")
    ax_life.set_ylabel("NPV cost over 20 years [DKK]")
    ax_life.set_title("Lifetime cost vs annual energy delivered")
    ax_life.grid(True, alpha=0.3)
    ax_life.legend(fontsize=8)

    fig.tight_layout()
    out = "capex_vs_energy.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot to {out}")

    plt.show()


if __name__ == "__main__":
    main()
