from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shapely.wkt

from pythermonet.components.vhe_field import BorefieldCoordinatesInput

_REQUIRED_COLS = ["ID", "WKT"]


def _parse_point_wkt(text: str) -> tuple[float, float, float | None, str | None]:
    """Parse a single WKT or EWKT ``POINT`` string into (x, y, z, srid).

    Parameters
    ----------
    text : str
        A WKT ``POINT [Z] (x y [z])`` string, optionally prefixed with an
        EWKT ``SRID=<code>;`` tag (as produced by PostGIS ``ST_AsEWKT``).

    Returns
    -------
    tuple of (float, float, float or None, str)
        (x, y, z, srid) — `z` is None for a 2D point.

    Raises
    ------
    ValueError
        If `text` is not a valid WKT/EWKT POINT, or carries no ``SRID=``
        prefix — the source CRS must be known, otherwise there is no way
        to tell projected meters from geographic degrees.

    """
    raw = text.strip()
    if not raw.upper().startswith("SRID="):
        raise ValueError(
            f"WKT value has no 'SRID=' prefix: {text!r}. The source CRS must "
            "be known (e.g. via PostGIS ST_AsEWKT) — plain WKT without an "
            "SRID cannot be safely localized."
        )
    prefix, _, raw = raw.partition(";")
    srid = f"EPSG:{prefix.split('=', 1)[1].strip()}"

    point = shapely.wkt.loads(raw)
    if point.geom_type != "Point":
        raise ValueError(f"Expected a WKT POINT, got '{point.geom_type}': {text!r}")

    z = float(point.z) if point.has_z else None
    return float(point.x), float(point.y), z, srid


def read_borefield_coordinates_tsv(path: str | Path) -> BorefieldCoordinatesInput:
    """Read borehole coordinates from a WKT/EWKT-based TSV file.

    Each row holds one borehole's identifier and its position as a WKT or
    EWKT ``POINT [Z] (...)`` string — matching what PostGIS ``ST_AsText``/
    ``ST_AsEWKT`` and QGIS both produce natively, so no bespoke coordinate
    columns are needed. Does no re-referencing to a local origin — see
    `pythermonet.components.vhe_field.localize_borefield_coordinates` for
    turning this into `VHEField.coordinates`.

    Parameters
    ----------
    path : str or Path
        Path to the borefield coordinates TSV file. Required columns: ID,
        WKT.

    Returns
    -------
    BorefieldCoordinatesInput
        Raw per-borehole coordinates in their original (source-CRS) frame.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If any required column is missing, a WKT value isn't a valid POINT
        or carries no SRID, or rows disagree on their SRID.

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Borefield coordinates TSV not found: {p}")

    df = pd.read_csv(p, sep=r"\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Borefield coordinates TSV missing required columns: {missing}")

    ids = df["ID"].astype(str).str.strip().to_list()
    parsed = [_parse_point_wkt(str(w)) for w in df["WKT"].to_list()]

    x = np.array([p[0] for p in parsed], dtype=float)
    y = np.array([p[1] for p in parsed], dtype=float)
    z_values = [p[2] for p in parsed]
    srids = {p[3] for p in parsed}

    if len(srids) > 1:
        raise ValueError(
            f"Borefield coordinates TSV mixes multiple SRIDs: {sorted(srids)} — "
            "every row must share the same source CRS."
        )
    crs = srids.pop()

    if all(v is None for v in z_values):
        z = None
    elif any(v is None for v in z_values):
        raise ValueError(
            "Borefield coordinates TSV mixes 2D and 3D points — "
            "either every row must carry a Z value, or none may."
        )
    else:
        z = np.array(z_values, dtype=float)

    return BorefieldCoordinatesInput(ids=ids, x=x, y=y, z=z, crs=crs)
