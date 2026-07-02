from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.components.distribution_network import DistributionNetwork


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


def read_undimensioned_topology_tsv_to_network(
    path: str | Path,
    *,
    pipe_material: Material,
    roughness: float,
    burial_depth: float,
    pipe_distance: float | None,
    n_parallel_pipes: int | None,
) -> DistributionNetwork:
    """
    Læser undimensioneret topologi og bygger et samlet DistributionNetwork.

    - PipeInfrastructure.trace_segments oprettes med outer_diameter=np.nan
      (diameter sættes senere af pipe-dimensionering).
    - Topologi/dimensioneringsfelter lagres på DistributionNetwork.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Topology TSV not found: {p}")

    df = pd.read_csv(p, sep=r"\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Topology TSV missing required columns: {missing}")

    # Parse arrays
    trace_names = df["Section"].astype(str).str.strip().to_list()
    sdr = df["SDR"].astype(float).to_numpy()
    trace_lengths = df["Trace_(m)"].astype(float).to_numpy()
    trace_counts = df["Number_of_traces"].astype(int).to_numpy()
    trace_max_pressure_loss = df["Max_pressure_loss_(Pa)"].astype(float).to_numpy()

    # HP grupper
    heat_pump_id_trace = [_parse_hp_id_vector(x) for x in df["HP_ID_vector"].to_list()]

    # Byg fysisk infrastruktur (segments 1:1 med pipe groups)
    trace_segments: list[PipeSegment] = []
    for i in range(len(df)):
        seg = PipeSegment(
            outer_diameter=np.nan,          # udfyldes efter dimensionering
            sdr=float(sdr[i]),
            material=pipe_material,
            roughness=float(roughness),
            id_=int(i),
            length=float(trace_lengths[i] * n_parallel_pipes),  # The number of parallel pipes multiplied by the number of traces 
        )
        trace_segments.append(seg)

    infrastructure = PipeInfrastructure(
        n_parallel_pipes=int(n_parallel_pipes),
        trace_segments=trace_segments,
        pipe_distance=pipe_distance,
        burial_depth=float(burial_depth),
    )

    return DistributionNetwork(
        infrastructure=infrastructure,
        trace_names=trace_names,
        heat_pump_id_trace=heat_pump_id_trace,
        trace_max_pressure_loss=trace_max_pressure_loss,
        sdr=sdr,
        trace_lengths=trace_lengths,
        trace_counts=trace_counts,
    )
