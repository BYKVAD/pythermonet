from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.components.distribution_network import DistributionNetwork
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.physics.hydraulics import reynolds_number


_REQUIRED_COLS = [
    "Section",
    "do_(mm)",
    "SDR",
    "Trace_(m)",
    "Number_of_traces",
    "Peak_flow_heating_m3/s",
]


def read_dimensioned_topology_tsv_to_hydraulic(
    path: str | Path,
    *,
    pipe_material: Material,
    brine: HeatCarrier,
    burial_depth: float,
    pipe_distance: Optional[float],
    n_parallel_pipes: int,
) -> tuple[DistributionNetwork, HydraulicResult]:
    """
    Read a pre-sized distribution network topology and return a
    ``DistributionNetwork`` together with a ``HydraulicResult`` built
    directly from the given flow rates — no hydraulic dimensioning is
    performed.

    File format (tab-separated)
    ---------------------------
    Required columns:
        Section, do_(mm), SDR, Trace_(m), Number_of_traces,
        Peak_flow_heating_m3/s

    Optional column:
        Peak_flow_cooling_m3/s   (required when cooling is present)

    Parameters
    ----------
    pipe_material : Material
        Thermal properties of the pipe wall material.
    brine : HeatCarrier
        Brine properties (used to compute Reynolds numbers).
    burial_depth : float
        Pipe burial depth [m].
    pipe_distance : float or None
        Centre-to-centre distance between supply and return pipes [m].
        Required when n_parallel_pipes == 2.
    n_parallel_pipes : int
        Number of parallel pipes per trace (1 or 2).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dimensioned topology TSV not found: {p}")

    df = pd.read_csv(p, sep=r"\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dimensioned topology TSV missing required columns: {missing}")

    has_cooling = "Peak_flow_cooling_m3/s" in df.columns and (
        pd.to_numeric(df["Peak_flow_cooling_m3/s"], errors="coerce").fillna(0.0) > 0
    ).any()

    n = len(df)
    trace_names = df["Section"].astype(str).str.strip().to_list()
    do_m        = df["do_(mm)"].astype(float).to_numpy() / 1000.0
    sdr         = df["SDR"].astype(float).to_numpy()
    trace_lengths    = df["Trace_(m)"].astype(float).to_numpy()
    trace_counts    = df["Number_of_traces"].astype(int).to_numpy()
    q_heat      = df["Peak_flow_heating_m3/s"].astype(float).to_numpy()

    di_m = do_m * (1.0 - 2.0 / sdr)

    # --- DistributionNetwork ---
    trace_segments: list[PipeSegment] = []
    for i in range(n):
        seg = PipeSegment(
            diameter_outer=float(do_m[i]),
            sdr=float(sdr[i]),
            material=pipe_material,
            roughness=0.0,   # not needed — pipes already sized
            id_=int(i),
            length=float(trace_lengths[i] * n_parallel_pipes),
        )
        trace_segments.append(seg)

    infrastructure = PipeInfrastructure(
        n_pipes_parallel=int(n_parallel_pipes),
        segments_trace=trace_segments,
        pipe_spacing=pipe_distance,
        burial_depth=float(burial_depth),
    )

    network = DistributionNetwork(
        pipe_infrastructure=infrastructure,
        names_trace=trace_names,
        heat_pump_ids_trace=[np.array([], dtype=int)] * n,
        pressure_losses_max_trace=np.full(n, np.nan),
        sdr=sdr,
        lengths_trace=trace_lengths,
        counts_trace=trace_counts,
    )

    # --- Reynolds numbers ---
    v_heat = q_heat / (np.pi * di_m**2 / 4.0)
    Re_heat = np.array(
        [reynolds_number(brine.density, brine.dynamic_viscosity, float(v), float(d))
         for v, d in zip(v_heat, di_m)],
        dtype=float,
    )

    if has_cooling:
        q_cool = df["Peak_flow_cooling_m3/s"].astype(float).to_numpy()
        v_cool = q_cool / (np.pi * di_m**2 / 4.0)
        Re_cool = np.array(
            [reynolds_number(brine.density, brine.dynamic_viscosity, float(v), float(d))
             for v, d in zip(v_cool, di_m)],
            dtype=float,
        )
    else:
        q_cool = None
        Re_cool = None

    hydraulic = HydraulicResult(
        network=network,
        diameters_outer=do_m,
        diameters_inner=di_m,
        governing_mode=np.full(n, "heating", dtype=object),
        volume_flow_rates_peak_heating=q_heat,
        volume_flow_rates_peak_cooling=q_cool,
        reynolds_numbers_heating=Re_heat,
        reynolds_numbers_cooling=Re_cool,
        pressure_losses_heating=np.full(n, np.nan),
        pressure_losses_cooling=np.full(n, np.nan) if has_cooling else None,
    )

    return network, hydraulic
