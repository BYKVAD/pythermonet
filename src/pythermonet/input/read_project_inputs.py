from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pythermonet.core.material import Material
from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.components.heat_pump import HeatPump
from pythermonet.components.distribution_network import (
    DistributionNetwork,
    DistributionNetworkParameters,
    build_distribution_network,
)

from pythermonet.input.read_pipe_catalog import read_pipe_catalog
from pythermonet.input.read_topology import read_undimensioned_topology_tsv
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv


@dataclass(frozen=True)
class ProjectInputs:
    pipe_catalog: list[Annulus]
    diameters_outer_catalog: np.ndarray
    network: DistributionNetwork
    heat_pumps: list[HeatPump]


def load_project_inputs(
    *,
    pipe_catalog_file: str | Path,
    topology_file: str | Path,
    heat_pump_file: str | Path,
    source_heat_carrier: HeatCarrier,
    pipe_material: Material,
    roughness: float,
    burial_depth: float,
    pipe_distance: float | None,
    n_parallel_pipes: int = 2,
) -> ProjectInputs:
    pipe_catalog = read_pipe_catalog(pipe_catalog_file)
    catalog_outer_diameters = np.asarray(
        sorted({a.diameter_outer for a in pipe_catalog}),
        dtype=float,
    )

    topology = read_undimensioned_topology_tsv(str(topology_file))
    network = build_distribution_network(
        topology,
        pipe_material=pipe_material,
        network_parameters=DistributionNetworkParameters(
            roughness=roughness,
            burial_depth=burial_depth,
            pipe_spacing=pipe_distance,
            n_pipes_parallel=n_parallel_pipes,
        ),
    )

    heat_pumps = read_heat_pumps_tsv(path=str(heat_pump_file))

    return ProjectInputs(
        pipe_catalog=pipe_catalog,
        diameters_outer_catalog=catalog_outer_diameters,
        network=network,
        heat_pumps=heat_pumps,
    )
