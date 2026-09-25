"""
test_topology_validation.py
---------------------------
Tests for the input checks that turn malformed topologies into clear,
trace-naming errors instead of failures deep inside pipe dimensioning
(e.g. "n must be >= 1" from the diversity factor, or a bare KeyError).

Run with:
    python -m pytest tests/test_topology_validation.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pythermonet.components.distribution_network import (
    DistributionNetworkParameters,
    UndimensionedTopologyInput,
    build_distribution_network,
)
from pythermonet.core.material import Material
from pythermonet.dimensioning import run_pipedimensioning

_PIPE_MATERIAL = Material(density=940.0, specific_heat=2000.0, thermal_conductivity=0.4)
_NETWORK_PARAMETERS = DistributionNetworkParameters(
    roughness=1e-5, burial_depth=1.0, pipe_spacing=None, n_pipes_parallel=1
)


def _topology(
    heat_pump_ids: list[list[int]], counts: list[int] | None = None
) -> UndimensionedTopologyInput:
    n = len(heat_pump_ids)
    return UndimensionedTopologyInput(
        trace_names=[f"Pipe_branch_{i + 1}" for i in range(n)],
        sdr=np.full(n, 17.0),
        trace_lengths=np.full(n, 10.0),
        trace_counts=np.asarray(counts if counts is not None else [1] * n),
        trace_max_pressure_loss=np.full(n, 1800.0),
        heat_pump_ids_trace=[np.asarray(ids, dtype=int) for ids in heat_pump_ids],
    )


def _build(topology: UndimensionedTopologyInput):
    return build_distribution_network(
        topology, pipe_material=_PIPE_MATERIAL, network_parameters=_NETWORK_PARAMETERS
    )


def test_valid_topology_builds() -> None:
    """Every trace serving at least one heat pump per parallel trace passes."""
    network = _build(_topology([[1, 2], [1, 2, 3]], counts=[2, 1]))

    assert network.names_trace == ["Pipe_branch_1", "Pipe_branch_2"]


def test_trace_without_heat_pumps_raises() -> None:
    """A trace with an empty HP_ID_vector is rejected, naming the trace."""
    with pytest.raises(ValueError, match="'Pipe_branch_2' has no heat pumps connected"):
        _build(_topology([[1, 2], [], [1, 2, 3]]))


def test_all_offending_traces_are_reported() -> None:
    """Every bad trace is listed in one error, not just the first."""
    with pytest.raises(ValueError) as excinfo:
        _build(_topology([[], [1], []]))

    message = str(excinfo.value)
    assert "'Pipe_branch_1'" in message
    assert "'Pipe_branch_3'" in message
    assert "'Pipe_branch_2'" not in message


def test_fewer_heat_pumps_than_parallel_traces_raises() -> None:
    """1 heat pump spread over 2 parallel traces is < 1 per trace -> rejected."""
    with pytest.raises(ValueError, match="'Pipe_branch_1' has 1 heat pump"):
        _build(_topology([[1]], counts=[2]))


def test_unknown_heat_pump_id_raises() -> None:
    """A topology ID missing from the heat pump list is reported by trace and ID."""
    network = _build(_topology([[1, 2], [1, 2, 99]]))
    # Only `id_` is read before the check fires, so stand-ins suffice here.
    heat_pumps = [SimpleNamespace(id_=1), SimpleNamespace(id_=2)]

    with pytest.raises(ValueError, match=r"'Pipe_branch_2' references \[99\]"):
        run_pipedimensioning(
            pipe_catalog=[], brine=None, network=network, heat_pumps=heat_pumps
        )
