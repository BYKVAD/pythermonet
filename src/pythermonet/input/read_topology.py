from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pythermonet.components.distribution_network import UndimensionedTopologyInput

_REQUIRED_COLS = [
    "Section",
    "SDR",
    "Trace_(m)",
    "Number_of_traces",
    "Max_pressure_loss_(Pa)",
    "HP_ID_vector",
]


def _parse_hp_id_vector(cell: str) -> np.ndarray:
    """
    TSV-feltet ser typisk ud som "1,2,3" eller "1, 2, 3".
    Returnerer np.array(dtype=int).
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.array([], dtype=int)

    s = str(cell).strip()
    if not s:
        return np.array([], dtype=int)

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return np.array([int(p) for p in parts], dtype=int)


def read_undimensioned_topology_tsv(path: str | Path) -> UndimensionedTopologyInput:
    """Read an undimensioned topology TSV file into raw per-trace arrays.

    Does no domain-object assembly — this only parses the file. Combine the
    result with a pipe material and `DistributionNetworkParameters` via
    `pythermonet.components.distribution_network.build_distribution_network`.

    Parameters
    ----------
    path : str or Path
        Path to the undimensioned topology TSV file. Required columns:
        Section, SDR, Trace_(m), Number_of_traces, Max_pressure_loss_(Pa),
        HP_ID_vector.

    Returns
    -------
    UndimensionedTopologyInput
        Raw per-trace arrays parsed from the file.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If any required column is missing.

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Topology TSV not found: {p}")

    df = pd.read_csv(p, sep=r"\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Topology TSV missing required columns: {missing}")

    trace_names = df["Section"].astype(str).str.strip().to_list()
    sdr = df["SDR"].astype(float).to_numpy()
    trace_lengths = df["Trace_(m)"].astype(float).to_numpy()
    trace_counts = df["Number_of_traces"].astype(int).to_numpy()
    trace_max_pressure_loss = df["Max_pressure_loss_(Pa)"].astype(float).to_numpy()
    heat_pump_ids_trace = [_parse_hp_id_vector(x) for x in df["HP_ID_vector"].to_list()]

    return UndimensionedTopologyInput(
        trace_names=trace_names,
        sdr=sdr,
        trace_lengths=trace_lengths,
        trace_counts=trace_counts,
        trace_max_pressure_loss=trace_max_pressure_loss,
        heat_pump_ids_trace=heat_pump_ids_trace,
    )
