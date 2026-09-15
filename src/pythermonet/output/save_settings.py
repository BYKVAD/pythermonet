"""Writing a project's domain objects out to a settings file."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from pythermonet.settings import UNITS_BY_TYPE, type_name_for


def save_settings(path: str | Path, objects: dict[str, object]) -> None:
    """Write the full current state of a project's domain objects to a settings file.

    Always writes a complete snapshot of every role in `objects` — never a
    partial patch against whatever is already on disk. Called explicitly by
    the caller; never triggered automatically as a side effect of loading.
    Each field is written as a `{"value": ..., "unit": ...}` object, so the
    file stays self-describing for consumers that can't `import pythermonet`.

    Parameters
    ----------
    path : str or Path
        Path to the settings JSON file to write.
    objects : dict of str to object
        Mapping from role name to the domain object currently in effect for
        that role (e.g. as returned by `pythermonet.input.load_settings`).

    Raises
    ------
    ValueError
        If any object's type is not registered in
        `pythermonet.settings.KNOWN_TYPES`, or if `UNITS_BY_TYPE` has no
        registered unit for one of its fields.

    """
    raw = {}
    for role, obj in objects.items():
        cls = type(obj)
        type_name = type_name_for(cls)
        units = UNITS_BY_TYPE[cls]

        field_values = dataclasses.asdict(obj)
        missing_units = [name for name in field_values if name not in units]
        if missing_units:
            raise ValueError(
                f"UNITS_BY_TYPE[{cls.__name__}] has no registered unit for "
                f"field(s): {missing_units}"
            )

        raw[role] = {
            "type": type_name,
            "values": {
                name: {"value": value, "unit": units[name]}
                for name, value in field_values.items()
            },
        }

    content = json.dumps(raw, indent=2) + "\n"
    Path(path).write_bytes(content.encode("utf-8"))
