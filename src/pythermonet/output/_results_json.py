"""Building blocks shared by the BHE and HHE results JSON files.

Each exported quantity becomes an entry ``{"value", "unit", "description"}``,
with unit and description read from the result classes' field metadata. The
BHE/HHE to-dict functions place entries in sections explicitly; anything not
placed (or placed twice) fails loudly, so no result is silently dropped.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from pythermonet.components.distribution_network import DistributionNetwork
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.simulation.distribution_pipe_thermal_model import ThermonetPerformance

_ENTRY_KEYS = frozenset({"value", "unit", "description"})

# Pulse suffixes of the six per-scope temperatures, in result-field order.
_PULSES_TEMPERATURE = (
    "annual_heating",
    "winter_heating",
    "peak_heating",
    "annual_cooling",
    "summer_cooling",
    "peak_cooling",
)

# Topology inputs on the network that are not exported (input echoes).
_NETWORK_FIELDS_NOT_EXPORTED = frozenset(
    {
        "pipe_infrastructure",
        "heat_pump_ids_trace",
        "pressure_losses_max_trace",
        "sdr",
        "material_pipe",
    }
)


def _to_json_value(value: object) -> object:
    """Convert a value to a plain Python type that json can write.

    Numpy arrays become lists; numpy scalars become Python floats or ints.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_quantities(
    cls: type,
    obj: object | None,
    *,
    excluded: frozenset[str] = frozenset(),
    mode: str | None = None,
) -> dict[str, dict[str, object]]:
    """All quantity fields of `cls` as JSON entries, keyed by field name.

    `obj` may be None (e.g. no cooling): every entry then gets value None so the
    file keeps the same keys. With `mode`, ``_<mode>`` is appended to each key
    and the mode to each description (for per-mode objects).
    """
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = excluded - names
    if unknown:
        raise ValueError(
            f"Excluded names are not fields of {cls.__name__}: {sorted(unknown)}"
        )

    entries: dict[str, dict[str, object]] = {}
    for f in dataclasses.fields(cls):
        if f.name in excluded:
            continue
        if "unit" not in f.metadata or "description" not in f.metadata:
            raise ValueError(
                f"{cls.__name__}.{f.name} has no unit/description metadata; "
                "add field(metadata=...) or exclude it explicitly."
            )
        key = f.name if mode is None else f"{f.name}_{mode}"
        description = f.metadata["description"]
        if mode is not None:
            description = f"{description}, {mode}"
        value = None if obj is None else _to_json_value(getattr(obj, f.name))
        entries[key] = {
            "value": value,
            "unit": f.metadata["unit"],
            "description": description,
        }
    return entries


def _take(
    entries: dict[str, dict[str, object]], keys: list[str]
) -> dict[str, dict[str, object]]:
    """Move `keys` out of `entries` into a new dict, in the given order.

    Fails loudly on a key that isn't there, or was already placed elsewhere.
    """
    missing = [key for key in keys if key not in entries]
    if missing:
        raise KeyError(
            f"Not available or already placed in another section: {missing}"
        )
    return {key: entries.pop(key) for key in keys}


def _check_all_placed(**pools: dict[str, dict[str, object]]) -> None:
    """Fail loudly if any collected quantity was not placed in a section."""
    leftover = {name: sorted(pool) for name, pool in pools.items() if pool}
    if leftover:
        raise ValueError(
            f"Result quantities not placed in any JSON section: {leftover}"
        )


def _metadata_section(
    ground_field_type: str, timestamp_run: datetime
) -> dict[str, object]:
    """Build the metadata section that identifies the results.

    Holds the ground field type, pythermonet version and run time (ISO 8601).
    """
    if timestamp_run.tzinfo is None:
        raise ValueError("timestamp_run must be timezone-aware to write its offset.")
    return {
        "ground_field_type": ground_field_type,
        "pythermonet_version": version("pythermonet"),
        "timestamp_run": timestamp_run.isoformat(timespec="seconds"),
    }


def _distribution_network_section(
    hydraulic: HydraulicResult,
    performance_heating: ThermonetPerformance,
    performance_cooling: ThermonetPerformance | None,
    extra_performance: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build the distribution network section, identical for BHE and HHE.

    ``performance`` holds whole-network values, ``traces`` the per-section ones.
    """
    performance = {
        **_collect_quantities(
            ThermonetPerformance, performance_heating, mode="heating"
        ),
        **_collect_quantities(
            ThermonetPerformance, performance_cooling, mode="cooling"
        ),
    }
    network = _collect_quantities(
        DistributionNetwork, hydraulic.network, excluded=_NETWORK_FIELDS_NOT_EXPORTED
    )
    per_trace = _collect_quantities(
        HydraulicResult, hydraulic, excluded=frozenset({"network"})
    )

    section: dict[str, object] = {
        "performance": {
            **_take(
                performance,
                [
                    "load_supply_fraction_heating",
                    "load_supply_fraction_cooling",
                    "temperatures_mean_heating",
                    "temperatures_mean_cooling",
                ],
            ),
            **extra_performance,
        },
        "traces": {
            **_take(network, list(network)),
            **_take(per_trace, list(per_trace)),
            # A pipe group is one section, so the per-group split is per trace row.
            **_take(
                performance,
                [
                    "load_supply_fractions_per_group_heating",
                    "load_supply_fractions_per_group_cooling",
                ],
            ),
        },
    }
    _check_all_placed(performance=performance, network=network, per_trace=per_trace)
    return section


def _format_json(node: object, level: int = 0) -> str:
    """Indented JSON with each ``{value, unit, description}`` entry on one line."""
    if isinstance(node, dict) and node and set(node) != _ENTRY_KEYS:
        pad = "  " * (level + 1)
        items = [
            f"{pad}{json.dumps(key)}: {_format_json(value, level + 1)}"
            for key, value in node.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    return json.dumps(node, allow_nan=False)


def _save_results_json(results: dict[str, object], path: str | os.PathLike) -> None:
    """Write results as UTF-8 JSON with LF line endings; NaN anywhere fails loudly."""
    Path(path).write_bytes((_format_json(results) + "\n").encode("utf-8"))
