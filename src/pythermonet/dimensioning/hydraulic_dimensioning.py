from __future__ import annotations

import numpy as np

from pythermonet.physics.hydraulics import pressure_loss_per_length as dp
from pythermonet.physics.hydraulics import reynolds_number as Re
from pythermonet.logging_config import get_logger
from pythermonet.system.diversity_factor import diversity_factor_from_n_heat_pumps

logger = get_logger(__name__)


def run_pipedimensioning(
    pipe_catalogue,   # Iterable of pipe catalogue entries (outerDiameter, SDR, ...)
    brine,            # HeatCarrier (rho, c, dynamicViscosity)
    network,          # DistributionNetwork (geometry, grouping, pressure limits)
    heat_pumps,       # HeatPumps (list of HeatPump with *_ground_load fields)
):
    """
    Dimension distribution pipes based on a pressure-loss criterion.

    IMPORTANT CONVENTION
    --------------------
    This function operates **only on ground-side thermal loads**.
    Electrical contributions from heat pumps have already been removed
    upstream (via *_ground_load fields on HeatPump).

    Sign convention:
      - Heating ground loads: positive (heat extracted from ground)
      - Cooling ground loads: negative (heat injected into ground)

    EXPECTS
    -------
    network.hp_id_trace : list[np.ndarray]
        Heat pump IDs connected to each trace group
    network.N_traces : np.ndarray
        Number of parallel traces per group
    network.L_traces : np.ndarray
        Trace length [m] per group (one-way)
    network.max_pressure_loss_trace : np.ndarray
        Maximum allowed pressure loss per trace (forward + return) [Pa]
    network.SDR : np.ndarray
        SDR per trace group
    network.infrastructure.traceSegments : list[PipeSegment]
        One representative segment per trace group (updated in-place)

    heat_pumps.heatPumpList : list[HeatPump]
        Each HeatPump must provide:
          - peakHeating_ground_load [W]
          - peakCooling_ground_load [W] (negative if cooling)
          - deltaTHeating [K]
          - deltaTCooling [K]

    RETURNS
    -------
    network
        Updated in-place with selected pipe diameters and Reynolds numbers
    """

    # Unique outer diameters from catalogue (sorted ascending)
    pipe_catalogue_sorted = np.asarray(
        sorted({a.outerDiameter for a in pipe_catalogue}),
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 0) Index heat pumps by ID for fast lookup
    # ------------------------------------------------------------------
    hp_by_id = {hp.ID: hp for hp in heat_pumps.heatPumpList}

    N_trace = len(network.hp_id_trace)

    # Cooling is considered present if at least one HP has a non-zero peak cooling load
    doCooling = any(
        np.isfinite(getattr(hp, "peakCooling_ground_load", 0.0))
        and hp.peakCooling_ground_load != 0.0
        for hp in heat_pumps.heatPumpList
    )

    # Volumetric flow per trace [m3/s]
    m3_s_per_trace_heating = np.zeros(N_trace, dtype=float)
    m3_s_per_trace_cooling = np.zeros(N_trace, dtype=float) if doCooling else None

    # ------------------------------------------------------------------
    # 1) Compute design volumetric flow per trace
    # ------------------------------------------------------------------
    for i in range(N_trace):
        ids = network.hp_id_trace[i]
        N_traces = float(network.N_traces[i])

        # Number of heat pumps per trace (legacy aggregation logic)
        N_HP_per_trace = len(ids) / N_traces if N_traces > 0 else len(ids)

        # Diversity factor (legacy formulation)
        # Peak fractions are handled elsewhere → unity scaling here
        S_H = diversity_factor_from_n_heat_pumps(N_HP_per_trace)

        # ---- Heating: sum ground-side peak loads ----
        m3_s_peak_heating_sum = 0.0
        for hpid in ids:
            hp = hp_by_id[int(hpid)]

            # Convert peak ground load [W] → volumetric flow [m3/s]
            Qv_hp = (
                hp.peakHeating_ground_load
                / hp.deltaTHeating
                / brine.rho
                / brine.c
            )
            m3_s_peak_heating_sum += float(Qv_hp)

        # Flow per trace (accounting for parallel traces)
        m3_s_per_trace_heating[i] = S_H * m3_s_peak_heating_sum / N_traces

        # ---- Cooling (optional) ----
        if doCooling:
            S_C = diversity_factor_from_n_heat_pumps(N_HP_per_trace)

            m3_s_peak_cooling_sum = 0.0
            for hpid in ids:
                hp = hp_by_id[int(hpid)]

                # Cooling ground load is negative → take minus to get magnitude
                Qv_hp = (
                    -hp.peakCooling_ground_load
                    / hp.deltaTCooling
                    / brine.rho
                    / brine.c
                )
                m3_s_peak_cooling_sum += float(Qv_hp)

            m3_s_per_trace_cooling[i] = S_C * m3_s_peak_cooling_sum / N_traces

    # ------------------------------------------------------------------
    # 2) Select smallest pipe diameter fulfilling pressure-loss criterion
    # ------------------------------------------------------------------
    dimensioned_pipe_diameter_heating = np.zeros(N_trace, dtype=float)
    dimensioned_pipe_diameter_cooling = np.zeros(N_trace, dtype=float) if doCooling else None

    for i in range(N_trace):
        SDR_i = float(network.SDR[i])

        # Convert catalogue outer diameters → inner diameters via SDR
        candidate_inner_diameters = pipe_catalogue_sorted * (1.0 - 2.0 / SDR_i)

        # Total hydraulic length = forward + return
        L_tot = network.infrastructure.NParallelPipes * float(network.L_traces[i])

        # Heating
        ok = (
            L_tot
            * dp(
                brine.rho,
                brine.dynamicViscosity,
                m3_s_per_trace_heating[i],
                candidate_inner_diameters,
            )
            < float(network.max_pressure_loss_trace[i])
        )
        idx = int(np.argmax(ok))  # first diameter satisfying criterion
        dimensioned_pipe_diameter_heating[i] = float(pipe_catalogue_sorted[idx])

        # Cooling (optional)
        if doCooling:
            okc = (
                L_tot
                * dp(
                    brine.rho,
                    brine.dynamicViscosity,
                    m3_s_per_trace_cooling[i],
                    candidate_inner_diameters,
                )
                < float(network.max_pressure_loss_trace[i])
            )
            idxc = int(np.argmax(okc))
            dimensioned_pipe_diameter_cooling[i] = float(pipe_catalogue_sorted[idxc])

    # ------------------------------------------------------------------
    # 3) Update representative trace pipe segments
    # ------------------------------------------------------------------
    # Assumption: one PipeSegment per trace group, same ordering
    for i, seg in enumerate(network.infrastructure.traceSegments):
        seg.outerDiameter = dimensioned_pipe_diameter_heating[i]
        # Heating diameter is treated as governing;
        # cooling diameter can be stored separately on network if needed

    # ------------------------------------------------------------------
    # 4) Compute Reynolds numbers for reporting / validation
    # ------------------------------------------------------------------
    di_H = dimensioned_pipe_diameter_heating * (1.0 - 2.0 / network.SDR.astype(float))
    v_H = m3_s_per_trace_heating / (np.pi * di_H**2 / 4.0)

    network.dimensionedPipeReynoldsNumberHeating = np.array(
        [Re(brine.rho, brine.dynamicViscosity, v, d) for v, d in zip(v_H, di_H)],
        dtype=float,
    )
    network.m3_s_per_trace_heating = m3_s_per_trace_heating

    if doCooling:
        di_C = dimensioned_pipe_diameter_cooling * (1.0 - 2.0 / network.SDR.astype(float))
        v_C = m3_s_per_trace_cooling / (np.pi * di_C**2 / 4.0)

        network.dimensionedPipeReynoldsNumberCooling = np.array(
            [Re(brine.rho, brine.dynamicViscosity, v, d) for v, d in zip(v_C, di_C)],
            dtype=float,
        )
        network.m3_s_per_trace_cooling = m3_s_per_trace_cooling

    logger.info(
        "Hydraulic pipe dimensioning complete. doCooling=%s, N_traces=%s",
        doCooling,
        N_trace,
    )

    return network