from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pythermonet.core.material import Material
from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.components.heat_pump import HeatPump
from pythermonet.components.distribution_network import DistributionNetwork

from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv


@dataclass(frozen=True)
class ProjectInputs:
    pipe_catalogue: list[Annulus]
    catalogue_outer_diameters: np.ndarray
    network: DistributionNetwork
    heat_pump_list: list[HeatPump]


def load_project_inputs(
    *,
    pipe_catalogue_file: str | Path,
    topology_file: str | Path,
    heat_pump_file: str | Path,
    source_heat_carrier: HeatCarrier,
    pipe_material: Material,
    roughness: float,
    burial_depth: float,
    pipe_distance: float | None,
    n_parallel_pipes: int = 2,
) -> ProjectInputs:
    pipe_catalogue = read_pipe_catalogue(pipe_catalogue_file)
    catalogue_outer_diameters = np.asarray(
        sorted({a.outer_diameter for a in pipe_catalogue}),
        dtype=float,
    )

    network = read_undimensioned_topology_tsv_to_network(
        str(topology_file),
        pipe_material=pipe_material,
        roughness=roughness,
        burial_depth=burial_depth,
        pipe_distance=pipe_distance,
        n_parallel_pipes=n_parallel_pipes,
    )

    heat_pump_list = read_heat_pumps_tsv(path=str(heat_pump_file))

    return ProjectInputs(
        pipe_catalogue=pipe_catalogue,
        catalogue_outer_diameters=catalogue_outer_diameters,
        network=network,
        heat_pump_list=heat_pump_list,
    )
