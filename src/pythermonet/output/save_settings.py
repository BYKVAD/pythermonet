"""Writing a project's domain objects out to a settings file."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from pythermonet.settings import type_name_for


def save_settings(path: str | Path, objects: dict[str, object]) -> None:
    """Write the full current state of a project's domain objects to a settings file.

    Always writes a complete snapshot of every role in `objects` — never a
    partial patch against whatever is already on disk. Called explicitly by
    the caller; never triggered automatically as a side effect of loading.

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
        `pythermonet.settings.KNOWN_TYPES`.

    """
    raw = {
        role: {
            "type": type_name_for(type(obj)),
            "values": dataclasses.asdict(obj),
        }
        for role, obj in objects.items()
    }

    content = json.dumps(raw, indent=2) + "\n"
    Path(path).write_bytes(content.encode("utf-8"))
