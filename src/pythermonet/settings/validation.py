"""Validating a settings block's fields against its resolved dataclass type."""

from __future__ import annotations

import dataclasses

from pythermonet.settings.presets import PRESETS_BY_TYPE, PRESETS_SOURCE_NOTE
from pythermonet.settings.units import UNITS_BY_TYPE


def field_names(cls: type) -> frozenset[str]:
    """Return the constructor-settable field names declared on a dataclass.

    Fields declared with ``init=False`` (computed, not settable) are excluded
    — they can never legally appear in a settings block, since the
    dataclass's own constructor doesn't accept them as arguments.

    Parameters
    ----------
    cls : type
        A dataclass type.

    Returns
    -------
    frozenset of str
        Every constructor-settable field name declared on `cls` or an
        ancestor dataclass.

    """
    return frozenset(f.name for f in dataclasses.fields(cls) if f.init)


def _format_presets(cls: type) -> str:
    presets = PRESETS_BY_TYPE.get(cls, {})
    if not presets:
        return "    (no reference values available for this type)"
    units = UNITS_BY_TYPE[cls]
    rows = []
    for name, instance in presets.items():
        field_values = ", ".join(
            f"{f.name}={getattr(instance, f.name)!r} [{units[f.name]}]"
            for f in dataclasses.fields(cls)
            if f.init
        )
        rows.append(f"    {name}: {field_values}")
    return "\n".join(rows)


def describe_block_problem(
    role: str,
    cls: type,
    unrecognized: frozenset[str],
    missing: frozenset[str],
) -> str:
    """Build a human-readable description of one invalid settings block.

    Parameters
    ----------
    role : str
        The settings file's role key for this block (e.g. ``"soil"``).
    cls : type
        The resolved domain class for this block.
    unrecognized : frozenset of str
        Field names present in the block that `cls` does not have.
    missing : frozenset of str
        Field names `cls` requires that are absent from the block.

    Returns
    -------
    str
        A multi-line description, including a reference-values table for
        `cls` when any field is missing.

    """
    lines = [f"- Block '{role}' (type: {cls.__name__}):"]
    if unrecognized:
        lines.append(f"    unrecognized field(s): {', '.join(sorted(unrecognized))}")
    if missing:
        lines.append(f"    missing field(s): {', '.join(sorted(missing))}")
        lines.append(f"    Reference values ({PRESETS_SOURCE_NOTE}):")
        lines.append(_format_presets(cls))
    return "\n".join(lines)


def validate_field_values(
    role: str, cls: type, values: dict[str, object]
) -> tuple[list[str], dict[str, object]]:
    """Validate each field's ``{value, unit}`` entry and unwrap it to a plain value.

    Every entry must be an object with exactly the keys ``value`` and
    ``unit``, and its declared unit must match `cls`'s registered unit for
    that field. Both malformed entries (wrong shape) and unit mismatches are
    collected as problems rather than raised immediately, so a caller can
    report every problem in a block at once.

    Parameters
    ----------
    role : str
        The settings file's role key for this block (e.g. ``"soil"``).
    cls : type
        The resolved domain class for this block.
    values : dict of str to object
        The block's raw ``"values"`` mapping, already confirmed to have
        exactly `cls`'s field names as keys.

    Returns
    -------
    tuple of (list of str, dict of str to object)
        Problem-description lines (empty if none) and the unwrapped
        ``{field_name: value}`` mapping, ready to pass to ``cls(**...)``.

    """
    units = UNITS_BY_TYPE[cls]
    problems: list[str] = []
    unwrapped: dict[str, object] = {}

    for field_name, entry in values.items():
        if not isinstance(entry, dict) or set(entry) != {"value", "unit"}:
            problems.append(
                f"- Block '{role}' (type: {cls.__name__}), field '{field_name}': "
                f"expected an object with exactly 'value' and 'unit' keys, "
                f"got {entry!r}"
            )
            continue

        expected_unit = units[field_name]
        if entry["unit"] != expected_unit:
            problems.append(
                f"- Block '{role}' (type: {cls.__name__}), field '{field_name}': "
                f"unit '{expected_unit}' expected, got '{entry['unit']}'"
            )
            continue

        unwrapped[field_name] = entry["value"]

    return problems, unwrapped
