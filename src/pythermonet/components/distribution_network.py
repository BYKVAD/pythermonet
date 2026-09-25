# src/pythermonet/components/distribution_network.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment


@dataclass
class DistributionNetworkParameters:
    """Construction parameters for an undimensioned distribution network.

    Parameters
    ----------
    roughness : float
        Pipe wall roughness [m].
    burial_depth : float
        Burial depth of the distribution pipes [m].
    pipe_spacing : float or None
        Wall-to-wall spacing between parallel pipes [m]. Only meaningful
        when `n_pipes_parallel` > 1; may be None otherwise.
    n_pipes_parallel : int
        Number of parallel pipes per trace [-].

    """

    roughness: float             # m
    burial_depth: float          # m
    pipe_spacing: float | None   # m
    n_pipes_parallel: int


@dataclass
class UndimensionedTopologyInput:
    """Raw per-trace arrays parsed from an undimensioned topology TSV file.

    Holds only what was read from the file — no pipe material, sizing, or
    installation parameters have been applied yet. See
    `pythermonet.input.read_topology.read_undimensioned_topology_tsv`, and
    `build_distribution_network` for turning this into a `DistributionNetwork`.

    Parameters
    ----------
    trace_names : list of str
        Name of each trace.
    sdr : numpy.ndarray
        Standard dimension ratio per trace [-].
    trace_lengths : numpy.ndarray
        One-way trace length [m].
    trace_counts : numpy.ndarray
        Number of traces per section [-].
    trace_max_pressure_loss : numpy.ndarray
        Maximum allowed pressure loss per trace [Pa].
    heat_pump_ids_trace : list of numpy.ndarray
        Heat pump IDs served by each trace.

    """

    trace_names: list[str]
    sdr: np.ndarray
    trace_lengths: np.ndarray             # m
    trace_counts: np.ndarray
    trace_max_pressure_loss: np.ndarray   # Pa
    heat_pump_ids_trace: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class DistributionNetwork:
    pipe_infrastructure: PipeInfrastructure
    names_trace: list[str]
    heat_pump_ids_trace: list[np.ndarray]
    pressure_losses_max_trace: np.ndarray
    sdr: np.ndarray
    lengths_trace: np.ndarray
    counts_trace: np.ndarray

    material_pipe: Material | None = None  # shared pipe material — must be identical across all trace segments

    def __post_init__(self) -> None:
        segments = self.pipe_infrastructure.segments_trace

        if len(segments) == 0:
            raise ValueError("No segments_trace in infrastructure.")

        first = segments[0].material
        k0 = first.thermal_conductivity
        rho0 = first.density  # eller hvad din anden property hedder
        c0 = first.specific_heat

        for seg in segments[1:]:
            mat = seg.material
            if mat.thermal_conductivity != k0 or mat.density != rho0 or mat.specific_heat != c0:
                raise ValueError(
                    "All pipe materials must have identical thermal properties."
                )

        object.__setattr__(self, "material_pipe", first)


def _validate_heat_pumps_per_trace(topology: UndimensionedTopologyInput) -> None:
    """Reject traces serving fewer heat pumps than parallel traces.

    Mirrors how pipe dimensioning computes heat pumps per trace
    (`len(ids) / count`); anything below 1 has no valid diversity factor.
    """
    problems = []
    for name, ids, count in zip(
        topology.trace_names, topology.heat_pump_ids_trace, topology.trace_counts
    ):
        n_heat_pumps = len(ids)
        if n_heat_pumps == 0:
            problems.append(f"'{name}' has no heat pumps connected")
        elif count > 0 and n_heat_pumps < count:
            problems.append(
                f"'{name}' has {n_heat_pumps} heat pump(s) spread over "
                f"{int(count)} parallel traces (fewer than 1 per trace)"
            )

    if problems:
        raise ValueError(
            "Topology contains traces that cannot be dimensioned, since each "
            "trace must serve at least one heat pump: "
            + "; ".join(problems)
            + ". Remove these traces from the topology or connect heat pumps "
            "to them."
        )


def build_distribution_network(
    topology: UndimensionedTopologyInput,
    *,
    pipe_material: Material,
    network_parameters: DistributionNetworkParameters,
) -> DistributionNetwork:
    """Assemble a `DistributionNetwork` from parsed topology and construction inputs.

    Does no file I/O — `topology` is expected to already be parsed, e.g. via
    `pythermonet.input.read_topology.read_undimensioned_topology_tsv`.

    Parameters
    ----------
    topology : UndimensionedTopologyInput
        Raw per-trace arrays parsed from an undimensioned topology file.
    pipe_material : Material
        Thermal properties of the distribution pipe wall material.
    network_parameters : DistributionNetworkParameters
        Roughness, burial depth, and parallel-pipe layout for the network.

    Returns
    -------
    DistributionNetwork
        Segments have `diameter_outer = nan` — sizing is applied later by
        pipe dimensioning.

    Raises
    ------
    ValueError
        If any trace serves no heat pumps, or fewer heat pumps than its
        number of parallel traces — such a trace carries less than one heat
        pump's flow and cannot be dimensioned.

    """
    _validate_heat_pumps_per_trace(topology)

    trace_segments: list[PipeSegment] = []
    for i in range(len(topology.trace_names)):
        # trace length x number of parallel pipes
        length = topology.trace_lengths[i] * network_parameters.n_pipes_parallel
        seg = PipeSegment(
            diameter_outer=np.nan,  # udfyldes efter dimensionering
            sdr=float(topology.sdr[i]),
            material=pipe_material,
            roughness=float(network_parameters.roughness),
            id_=int(i),
            length=float(length),
        )
        trace_segments.append(seg)

    infrastructure = PipeInfrastructure(
        n_pipes_parallel=int(network_parameters.n_pipes_parallel),
        segments_trace=trace_segments,
        pipe_spacing=network_parameters.pipe_spacing,
        burial_depth=float(network_parameters.burial_depth),
    )

    return DistributionNetwork(
        pipe_infrastructure=infrastructure,
        names_trace=topology.trace_names,
        heat_pump_ids_trace=topology.heat_pump_ids_trace,
        pressure_losses_max_trace=topology.trace_max_pressure_loss,
        sdr=topology.sdr,
        lengths_trace=topology.trace_lengths,
        counts_trace=topology.trace_counts,
    )
