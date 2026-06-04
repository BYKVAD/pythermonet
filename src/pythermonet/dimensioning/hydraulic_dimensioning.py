from __future__ import annotations

import numpy as np

from pythermonet.physics.hydraulics import pressure_loss_per_length as dp
from pythermonet.physics.hydraulics import reynolds_number as Re
from pythermonet.logging_config import get_logger
from pythermonet.system.diversity_factor import diversity_factor_from_n_heat_pumps
from pythermonet.dimensioning.hydraulic_result import HydraulicResult

logger = get_logger(__name__)


def run_pipedimensioning(
    pipe_catalogue,   # Iterable of pipe catalogue entries (outer_diameter, sdr, ...)
    brine,            # HeatCarrier (density, specific_heat, dynamic_viscosity)
    network,          # DistributionNetwork
    heat_pump_list,   # list[HeatPump]
) -> HydraulicResult:
    """
    Dimensionerer distributionsrør pr. trace ud fra tryktabskriterium i både
    heating og cooling mode og vælger den største nødvendige diameter pr. trace.

    Returnerer et HydraulicResult med alle dimensioneringsresultater.
    Netværksobjektet muteres ikke.
    """

    # Sortér unikke outer diameters fra kataloget
    pipe_catalogue_sorted = np.asarray(
        sorted({a.outer_diameter for a in pipe_catalogue}),
        dtype=float,
    )

    hp_by_id = {hp.ID: hp for hp in heat_pump_list}
    N_trace = len(network.hp_id_trace)

    doCooling = any(
        np.isfinite(getattr(hp, "peakCooling_ground_load", 0.0))
        and hp.peakCooling_ground_load != 0.0
        for hp in heat_pump_list
    )

    m3_s_per_trace_heating = np.zeros(N_trace, dtype=float)
    m3_s_per_trace_cooling = np.zeros(N_trace, dtype=float) if doCooling else None

    sdr_arr = np.asarray(network.sdr, dtype=float)
    L_trace_arr = np.asarray(network.L_traces, dtype=float)
    N_traces_arr = np.asarray(network.N_traces, dtype=float)
    max_dp_arr = np.asarray(network.max_pressure_loss_trace, dtype=float)

    n_parallel_pipes = float(network.infrastructure.NParallelPipes)

    # ------------------------------------------------------------------
    # 1) Design flow per trace
    # ------------------------------------------------------------------
    for i in range(N_trace):
        ids = network.hp_id_trace[i]
        N_traces_i = N_traces_arr[i]

        N_HP_per_trace = len(ids) / N_traces_i if N_traces_i > 0 else len(ids)

        S_H = diversity_factor_from_n_heat_pumps(N_HP_per_trace)

        m3_s_peak_heating_sum = 0.0
        for hpid in ids:
            hp = hp_by_id[int(hpid)]
            Qv_hp = (
                hp.peakHeating_ground_load
                / hp.deltaTHeating
                / brine.density
                / brine.specific_heat
            )
            m3_s_peak_heating_sum += float(Qv_hp)

        m3_s_per_trace_heating[i] = S_H * m3_s_peak_heating_sum / N_traces_i

        if doCooling:
            S_C = diversity_factor_from_n_heat_pumps(N_HP_per_trace)

            m3_s_peak_cooling_sum = 0.0
            for hpid in ids:
                hp = hp_by_id[int(hpid)]
                Qv_hp = (
                    hp.peakCooling_ground_load
                    / hp.deltaTCooling
                    / brine.density
                    / brine.specific_heat
                )
                m3_s_peak_cooling_sum += float(Qv_hp)

            m3_s_per_trace_cooling[i] = S_C * m3_s_peak_cooling_sum / N_traces_i

    # ------------------------------------------------------------------
    # 2) Find required diameter in heating og cooling pr. trace
    #    og vælg installeret diameter + governing mode
    # ------------------------------------------------------------------
    selected_outer_diameter = np.zeros(N_trace, dtype=float)
    governing_mode = np.empty(N_trace, dtype=object)

    for i in range(N_trace):
        sdr_i = float(sdr_arr[i])

        candidate_outer_diameters = pipe_catalogue_sorted
        candidate_inner_diameters = candidate_outer_diameters * (1.0 - 2.0 / sdr_i)

        L_tot_i = n_parallel_pipes * float(L_trace_arr[i])
        dp_limit_i = float(max_dp_arr[i])

        # Heating candidate
        ok_heat = np.array(
            [
                L_tot_i * dp(
                    brine.density,
                    brine.dynamic_viscosity,
                    float(m3_s_per_trace_heating[i]),
                    float(di),
                ) < dp_limit_i
                for di in candidate_inner_diameters
            ],
            dtype=bool,
        )

        if not np.any(ok_heat):
            raise ValueError(
                f"No pipe diameter satisfies heating pressure-loss criterion for trace {i}."
            )

        idx_heat = int(np.argmax(ok_heat))
        d_heat = float(candidate_outer_diameters[idx_heat])

        # Cooling candidate
        if doCooling:
            ok_cool = np.array(
                [
                    L_tot_i * dp(
                        brine.density,
                        brine.dynamic_viscosity,
                        float(m3_s_per_trace_cooling[i]),
                        float(di),
                    ) < dp_limit_i
                    for di in candidate_inner_diameters
                ],
                dtype=bool,
            )

            if not np.any(ok_cool):
                raise ValueError(
                    f"No pipe diameter satisfies cooling pressure-loss criterion for trace {i}."
                )

            idx_cool = int(np.argmax(ok_cool))
            d_cool = float(candidate_outer_diameters[idx_cool])

            if d_heat > d_cool:
                selected_outer_diameter[i] = d_heat
                governing_mode[i] = "heating"
            elif d_cool > d_heat:
                selected_outer_diameter[i] = d_cool
                governing_mode[i] = "cooling"
            else:
                selected_outer_diameter[i] = d_heat
                governing_mode[i] = "equal"
        else:
            selected_outer_diameter[i] = d_heat
            governing_mode[i] = "heating"

    # ------------------------------------------------------------------
    # 3) Beregn Reynolds-tal og tryktab for det installerede netværk
    #    i begge modes
    # ------------------------------------------------------------------
    installed_inner_diameter = selected_outer_diameter * (1.0 - 2.0 / sdr_arr)
    L_tot_arr = n_parallel_pipes * L_trace_arr

    v_H = m3_s_per_trace_heating / (np.pi * installed_inner_diameter**2 / 4.0)
    Re_heating = np.array(
        [
            Re(brine.density, brine.dynamic_viscosity, abs(float(v)), float(d))
            for v, d in zip(v_H, installed_inner_diameter)
        ],
        dtype=float,
    )
    dp_heating = np.array(
        [
            L_tot_arr[i] * dp(
                brine.density,
                brine.dynamic_viscosity,
                float(m3_s_per_trace_heating[i]),
                float(installed_inner_diameter[i]),
            )
            for i in range(N_trace)
        ],
        dtype=float,
    )

    Re_cooling = None
    dp_cooling = None
    if doCooling:
        v_C = m3_s_per_trace_cooling / (np.pi * installed_inner_diameter**2 / 4.0)
        Re_cooling = np.array(
            [
                Re(brine.density, brine.dynamic_viscosity, abs(float(v)), float(d))
                for v, d in zip(v_C, installed_inner_diameter)
            ],
            dtype=float,
        )
        dp_cooling = np.array(
            [
                L_tot_arr[i] * dp(
                    brine.density,
                    brine.dynamic_viscosity,
                    float(m3_s_per_trace_cooling[i]),
                    float(installed_inner_diameter[i]),
                )
                for i in range(N_trace)
            ],
            dtype=float,
        )

    return HydraulicResult(
        network=network,
        outer_diameter=selected_outer_diameter,
        inner_diameter=installed_inner_diameter,
        governing_mode=np.asarray(governing_mode, dtype=object),
        m3_s_heating=m3_s_per_trace_heating,
        m3_s_cooling=m3_s_per_trace_cooling,
        Re_heating=Re_heating,
        Re_cooling=Re_cooling,
        dp_heating=dp_heating,
        dp_cooling=dp_cooling,
    )

def print_pipe_dimensioning_table(hydraulic: HydraulicResult):
    """
    Printer en tabel for det installerede trace-netværk.
    """

    N = len(hydraulic.outer_diameter)
    doCooling = hydraulic.Re_cooling is not None

    header_parts = [
        f"{'Trace':>6}",
        f"{'Mode':>10}",
        f"{'Do [mm]':>10}",
        f"{'Di [mm]':>10}",
        f"{'Re_heat':>12}",
        f"{'dp_heat [kPa]':>14}",
    ]

    if doCooling:
        header_parts += [
            f"{'Re_cool':>12}",
            f"{'dp_cool [kPa]':>14}",
        ]

    header = " ".join(header_parts)

    print("\nDistribution pipe dimensioning")
    print(header)
    print("-" * len(header))

    for i in range(N):
        row_parts = [
            f"{i:6d}",
            f"{str(hydraulic.governing_mode[i]):>10}",
            f"{hydraulic.outer_diameter[i] * 1000:10.1f}",
            f"{hydraulic.inner_diameter[i] * 1000:10.1f}",
            f"{hydraulic.Re_heating[i]:12.0f}",
            f"{hydraulic.dp_heating[i] / 1000:14.2f}",
        ]

        if doCooling:
            row_parts += [
                f"{hydraulic.Re_cooling[i]:12.0f}",
                f"{hydraulic.dp_cooling[i] / 1000:14.2f}",
            ]

        print(" ".join(row_parts))

    print()