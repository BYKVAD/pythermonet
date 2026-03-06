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
    roughness_height: float,
    burial_depth: float,
    pipe_distance: float | None,
    n_parallel_pipes: int | None,
) -> DistributionNetwork:
    """
    Læser undimensioneret topologi og bygger et samlet DistributionNetwork.

    - PipeInfrastructure.traceSegments oprettes med outerDiameter=np.nan
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
    SDR = df["SDR"].astype(float).to_numpy()
    L_traces = df["Trace_(m)"].astype(float).to_numpy()
    N_traces = df["Number_of_traces"].astype(int).to_numpy()
    max_pressure_loss_trace = df["Max_pressure_loss_(Pa)"].astype(float).to_numpy()

    # HP grupper
    hp_id_trace = [_parse_hp_id_vector(x) for x in df["HP_ID_vector"].to_list()]

    # Byg fysisk infrastruktur (segments 1:1 med pipe groups)
    trace_segments: list[PipeSegment] = []
    for i in range(len(df)):
        seg = PipeSegment(
            outerDiameter=np.nan,          # udfyldes efter dimensionering
            SDR=float(SDR[i]),
            material=pipe_material,
            roughnessHeight=float(roughness_height),
            ID=int(i),
            length=float(L_traces[i] * n_parallel_pipes),  # The number of parallel pipes multiplied by the number of traces 
        )
        trace_segments.append(seg)

    infrastructure = PipeInfrastructure(
        NParallelPipes=int(n_parallel_pipes),
        traceSegments=trace_segments,
        pipeDistance=pipe_distance,
        burialDepth=float(burial_depth),
    )

    return DistributionNetwork(
        infrastructure=infrastructure,
        trace_names=trace_names,
        hp_id_trace=hp_id_trace,
        max_pressure_loss_trace=max_pressure_loss_trace,
        SDR=SDR,
        L_traces=L_traces,
        N_traces=N_traces,
    )
