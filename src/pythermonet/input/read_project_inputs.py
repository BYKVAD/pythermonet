from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pythermonet.core.material import Material
from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.components.heat_pump import HeatPump

from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_infrastructure
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv


@dataclass(frozen=True)
class ProjectInputs:
    pipe_catalogue: list[Annulus]         # geometric catalogue entries
    d_pipes_m: np.ndarray                 # unique outer diameters [m]
    infrastructure: object                # PipeInfrastructure (type if available)
    pipe_group_names: list[str]
    hp_id_groups: list[np.ndarray]
    dp_PG: np.ndarray
    heat_pump_list: list[HeatPump]


def load_project_inputs(
    *,
    pipe_catalogue_file: str | Path,
    topology_file: str | Path,
    heat_pump_file: str | Path,
    source_heat_carrier: HeatCarrier,
    pipe_material: Material,
    roughness_height: float,
    burial_depth: float,
    pipe_distance: float | None,
    n_parallel_pipes: int = 2,
) -> ProjectInputs:
    # 1) Pipe catalogue (Annulus only)
    pipe_catalogue = read_pipe_catalogue(pipe_catalogue_file)
    d_pipes_m = np.asarray(
        sorted({a.outerDiameter for a in pipe_catalogue}),
        dtype=float,
    )

    # 2) Topology -> infrastructure
    infrastructure, pipe_group_names, hp_id_groups, dp_PG = (
        read_undimensioned_topology_tsv_to_infrastructure(
            str(topology_file),
            pipe_material=pipe_material,
            roughness_height=roughness_height,
            burial_depth=burial_depth,
            pipe_distance=pipe_distance,
            n_parallel_pipes=n_parallel_pipes,
        )
    )

    # 3) Heat pumps
    heat_pump_list = read_heat_pumps_tsv(path=str(heat_pump_file))

    return ProjectInputs(
        pipe_catalogue=pipe_catalogue,
        d_pipes_m=d_pipes_m,
        infrastructure=infrastructure,
        pipe_group_names=pipe_group_names,
        hp_id_groups=hp_id_groups,
        dp_PG=dp_PG,
        heat_pump_list=heat_pump_list,
    )
