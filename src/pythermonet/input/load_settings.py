"""Reading and validating a project's settings file into pythermonet domain objects."""

from __future__ import annotations

import json
from pathlib import Path

from pythermonet.settings import (
    class_for_type_name,
    describe_block_problem,
    field_names,
    validate_field_values,
)

_RESERVED_KEY_PREFIX = "qthermonet_"


def load_settings(path: str | Path) -> dict[str, object]:
    """Load and validate a project settings file into pythermonet domain objects.

    Loading is pure and all-or-nothing: it never mutates the file, and never
    fills in a missing field from a default. Either every block in the file
    matches its declared type's current schema exactly — including each
    field's declared unit — or nothing is constructed and a `ValueError`
    describing every problem block is raised.

    Top-level keys whose name starts with ``"qthermonet_"`` (case-insensitive)
    are reserved for external consumers (currently QThermonet) to stash their
    own arbitrary JSON alongside pythermonet's own blocks. Such keys are
    skipped entirely — never type-checked, validated, or included in the
    returned mapping.

    Parameters
    ----------
    path : str or Path
        Path to the settings JSON file.

    Returns
    -------
    dict of str to object
        Mapping from role name (the settings file's top-level key) to the
        constructed domain object for that role.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If any block in the file has an unrecognized `"type"` tag; fields
        that do not match its declared type's current schema (unrecognized
        and/or missing fields); a field entry that isn't a well-formed
        `{"value": ..., "unit": ...}` object; or a field whose declared unit
        doesn't match the type's registered unit for that field. The message
        lists every problem block found in the file, not just the first one.

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Settings file not found: {p}")

    with p.open(encoding="utf-8") as f:
        raw: dict[str, dict[str, object]] = json.load(f)

    problems: list[str] = []
    objects: dict[str, object] = {}

    for role, block in raw.items():
        if role.lower().startswith(_RESERVED_KEY_PREFIX):
            continue
        type_name = block["type"]
        values = dict(block["values"])

        try:
            cls = class_for_type_name(type_name)
        except ValueError as exc:
            problems.append(f"- Block '{role}': {exc}")
            continue

        required = field_names(cls)
        provided = frozenset(values)
        unrecognized = provided - required
        missing = required - provided

        if unrecognized or missing:
            problems.append(describe_block_problem(role, cls, unrecognized, missing))
            continue

        field_problems, unwrapped = validate_field_values(role, cls, values)
        if field_problems:
            problems.extend(field_problems)
        else:
            objects[role] = cls(**unwrapped)

    if problems:
        message = f"Problems found in settings file '{p}':\n" + "\n".join(problems)
        raise ValueError(message)

    return objects
