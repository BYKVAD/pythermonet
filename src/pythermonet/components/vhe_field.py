from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any

import numpy as np
import pygfunction as gt
from pyproj import CRS, Transformer

from ..core.pipe_segment import PipeSegment, PipeSegmentParameters
from ..core.annulus import Annulus
from ..core.material import Material


@dataclass
class BorefieldCoordinatesInput:
    """Raw per-borehole coordinates parsed from a WKT/EWKT geometry source.

    Holds coordinates in their original (source-CRS) frame — no
    re-referencing to a local origin has been applied yet. See
    `localize_borefield_coordinates` for turning this into
    `VHEField.coordinates`.

    Parameters
    ----------
    ids : list of str
        Borehole identifier, one per row.
    x : numpy.ndarray
        Easting/X coordinate (or longitude, if `crs` is geographic) in the
        source CRS.
    y : numpy.ndarray
        Northing/Y coordinate (or latitude, if `crs` is geographic) in the
        source CRS.
    z : numpy.ndarray or None
        Z coordinate [m], or None if no row carried one.
    crs : str
        SRID shared by every row (e.g. ``"EPSG:25832"``) — required, since
        there is no way to safely localize coordinates without knowing
        whether they are projected meters or geographic degrees.

    """

    ids: list[str]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None      # m
    crs: str


def localize_borefield_coordinates(
    borefield_input: BorefieldCoordinatesInput,
    origin: tuple[float, float] | None = None,
) -> list[list[float]]:
    """Re-reference raw borehole coordinates to a local origin, in meters.

    `VHEField`/`pygfunction` only need relative distances between
    boreholes, in meters. Two cases:

    - `crs` is already a projected (linear-unit) CRS: a plain subtraction
      of the origin is exact and cheap — no reprojection needed.
    - `crs` is geographic (angular-unit, e.g. WGS84 lon/lat): coordinates
      are reprojected through a local azimuthal-equidistant (AEQD)
      projection centered at `origin`, which gives correct local meters
      with negligible distortion at borefield scale (tens-hundreds of m).

    Parameters
    ----------
    borefield_input : BorefieldCoordinatesInput
        Raw per-borehole coordinates, e.g. from
        `pythermonet.input.read_borefield_coordinates_tsv`.
    origin : tuple of (float, float) or None
        Point to center the local frame on, in the same CRS as
        `borefield_input` (i.e. (x, y) if projected, (lon, lat) if
        geographic). Defaults to the first borehole's position.

    Returns
    -------
    list of list of float
        Local coordinates in meters, suitable for `VHEField.coordinates` —
        each row is `[x, y]` or `[x, y, z]`, matching whether
        `borefield_input.z` is set.

    """
    crs = CRS.from_user_input(borefield_input.crs)
    origin_a, origin_b = (
        origin
        if origin is not None
        else (float(borefield_input.x[0]), float(borefield_input.y[0]))
    )

    if crs.is_geographic:
        ellipsoid = crs.ellipsoid
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={origin_b} +lon_0={origin_a} "
            f"+a={ellipsoid.semi_major_metre} +rf={ellipsoid.inverse_flattening} "
            "+units=m +no_defs"
        )
        transformer = Transformer.from_crs(crs, local_crs, always_xy=True)
        local_x, local_y = transformer.transform(borefield_input.x, borefield_input.y)
    else:
        axis_unit = crs.axis_info[0].unit_name
        if axis_unit not in ("metre", "meter"):
            raise ValueError(
                f"{borefield_input.crs} is a projected CRS with unit "
                f"'{axis_unit}', not meters — cannot localize."
            )
        local_x = borefield_input.x - origin_a
        local_y = borefield_input.y - origin_b

    if borefield_input.z is None:
        return [[float(x), float(y)] for x, y in zip(local_x, local_y)]

    return [
        [float(x), float(y), float(z)]
        for x, y, z in zip(local_x, local_y, borefield_input.z)
    ]


@dataclass(frozen=True, slots=True)
class VHEField:
    id_: int
    pipe: PipeSegment
    borehole: Annulus
    grout: Material
    coordinates: Sequence[Sequence[float]]
    shank_spacing: float   # m

    length_borehole: float  # m
    burial_depth: float     # m
    use_z_as_depth: bool = False
    tilt_rad: float = 0.0
    orientation_rad: float = 0.0
    heat_exchanger_type: str = "1U"  # e.g. "1U", "2U", "CX" — stored for future use, not yet active in calculations

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)

        if coords.ndim != 2:
            raise ValueError("coordinates must be a 2D array-like structure.")
        if coords.shape[0] == 0:
            raise ValueError("VHEField must contain at least one borehole.")
        if coords.shape[1] not in (2, 3):
            raise ValueError("coordinates must have shape (n, 2) or (n, 3).")

        if self.pipe is None:
            raise ValueError("pipe must not be None.")
        if self.borehole is None:
            raise ValueError("borehole must not be None.")
        if self.grout is None:
            raise ValueError("grout must not be None.")
        if self.shank_spacing <= 0.0:
            raise ValueError("shank_spacing must be > 0.")

        if self.length_borehole <= 0.0:
            raise ValueError("borehole_length must be > 0.")
        if self.burial_depth < 0.0:
            raise ValueError("burial_depth must be >= 0.")

        object.__setattr__(
            self,
            "coordinates",
            tuple(tuple(float(v) for v in row) for row in coords),
        )
        object.__setattr__(self, "shank_spacing", float(self.shank_spacing))
        object.__setattr__(self, "length_borehole", float(self.length_borehole))
        object.__setattr__(self, "burial_depth", float(self.burial_depth))
        object.__setattr__(self, "use_z_as_depth", bool(self.use_z_as_depth))
        object.__setattr__(self, "tilt_rad", float(self.tilt_rad))
        object.__setattr__(self, "orientation_rad", float(self.orientation_rad))

    @property
    def n_dims(self) -> int:
        return len(self.coordinates[0])

    @property
    def n_boreholes(self) -> int:
        return len(self.coordinates)

    @property
    def x_coords(self) -> tuple[float, ...]:
        """X-coordinate of each borehole [m]."""
        return tuple(row[0] for row in self.coordinates)

    @property
    def y_coords(self) -> tuple[float, ...]:
        """Y-coordinate of each borehole [m]."""
        return tuple(row[1] for row in self.coordinates)

    @property
    def z_coords(self) -> tuple[float, ...] | None:
        """Z-coordinate of each borehole [m], or None if coordinates are 2D."""
        if self.n_dims == 3:
            return tuple(row[2] for row in self.coordinates)
        return None

    @property
    def xy_coords(self) -> tuple[tuple[float, float], ...]:
        """(X, Y) coordinates of each borehole [m]."""
        return tuple((row[0], row[1]) for row in self.coordinates)

    @property
    def radius_borehole(self) -> float:
        """Borehole radius [m], derived from borehole.diameter_outer."""
        return float(self.borehole.diameter_outer) / 2.0

    def to_pygfunction_boreholes(self) -> list[gt.boreholes.Borehole]:
        xy = np.asarray(self.xy_coords, dtype=float)
        unique_xy = np.unique(xy, axis=0)
        if len(unique_xy) != len(xy):
            raise ValueError(
                f"Duplicate borehole coordinates detected: "
                f"{len(xy) - len(unique_xy)} duplicate(s)."
            )

        boreholes: list[gt.boreholes.Borehole] = []

        for row in self.coordinates:
            x = float(row[0])
            y = float(row[1])

            if self.use_z_as_depth:
                if len(row) < 3:
                    raise ValueError(
                        "use_z_as_depth=True requires 3D coordinates [x, y, z]."
                    )
                D_use = float(row[2])
            else:
                D_use = self.burial_depth

            boreholes.append(
                gt.boreholes.Borehole(
                    H=self.length_borehole,
                    D=D_use,
                    r_b=self.radius_borehole,
                    x=x,
                    y=y,
                    tilt=self.tilt_rad,
                    orientation=self.orientation_rad,
                )
            )

        return boreholes

    def compute_pygfunctions(
        self,
        times_s,
        alpha_m2_s: float,
        method: str = "equivalent",
        boundary_condition: str = "UHTR",
        options: dict | None = None,
    ) -> np.ndarray:
        times_arr = np.asarray(times_s, dtype=float).reshape(-1)

        if times_arr.size == 0:
            raise ValueError("times_s must be non-empty.")
        if np.any(times_arr <= 0.0):
            raise ValueError("All times in times_s must be > 0.")
        if alpha_m2_s <= 0.0:
            raise ValueError("alpha_m2_s must be > 0.")

        boreholes = self.to_pygfunction_boreholes()

        gfunc = gt.gfunction.gFunction(
            boreholes,
            float(alpha_m2_s),
            time=times_arr,
            method=method,
            boundary_condition=boundary_condition,
            options=options or {},
        )

        return np.asarray(gfunc.gFunc, dtype=float)

    def compute_gfunctions_from_soil(
        self,
        times_s,
        soil,
        method: str = "equivalent",
        boundary_condition: str = "UHTR",
        options: dict | None = None,
    ) -> np.ndarray:
        alpha_m2_s = float(soil.thermal_conductivity) / (float(soil.density) * float(soil.specific_heat))

        return self.compute_gfunctions(
            times_s=times_s,
            alpha_m2_s=alpha_m2_s,
            method=method,
            boundary_condition=boundary_condition,
            options=options,
        )


@dataclass
class VHEFieldParameters:
    """Construction parameters for a vertical heat exchanger (BHE) field.

    Excludes `length_borehole` and the u-pipe's `PipeSegment.length` —
    both are placeholders overwritten by the borehole-length bisection
    solver (`pythermonet.dimensioning.BHE.borehole_length`) regardless of
    their starting value, so neither is a genuine input.

    Parameters
    ----------
    shank_spacing : float
        Center-to-center spacing between the U-pipe legs [m].
    burial_depth : float
        Burial depth of the borehole top [m].
    tilt_rad : float
        Borehole tilt from vertical [rad].
    orientation_rad : float
        Borehole tilt azimuth [rad].
    heat_exchanger_type : str
        Heat exchanger configuration, e.g. ``"1U"``, ``"2U"``, ``"CX"``.

    """

    shank_spacing: float       # m
    burial_depth: float        # m
    tilt_rad: float
    orientation_rad: float
    heat_exchanger_type: str


def build_vhe_field(
    segment_parameters: PipeSegmentParameters,
    field_parameters: VHEFieldParameters,
    pipe_material: Material,
    borehole: Annulus,
    grout: Material,
    coordinates: Sequence[Sequence[float]],
) -> VHEField:
    """Assemble a `VHEField` from flat construction parameters.

    Builds the u-pipe `PipeSegment` internally — `id_` and `length` are
    fixed placeholders (`length` is overwritten by the borehole-length
    bisection solver regardless of its value).

    Parameters
    ----------
    segment_parameters : PipeSegmentParameters
        Outer diameter, SDR, and roughness of the U-pipe.
    field_parameters : VHEFieldParameters
        Shank spacing, burial depth, tilt, orientation, and heat exchanger
        type shared by every borehole in the field.
    pipe_material : Material
        Thermal properties of the U-pipe wall material.
    borehole : Annulus
        Borehole diameter and SDR.
    grout : Material
        Thermal properties of the borehole grout.
    coordinates : sequence of sequence of float
        (x, y) or (x, y, z) position of each borehole, e.g. from
        `localize_borefield_coordinates`.

    Returns
    -------
    VHEField
        `length_borehole` holds a placeholder value (`120.0`) — sized later
        by the BHE bisection solver.

    """
    pipe = PipeSegment(
        diameter_outer=segment_parameters.diameter_outer,
        sdr=segment_parameters.sdr,
        material=pipe_material,
        roughness=segment_parameters.roughness,
        id_=0,
        length=100.0,  # placeholder — unused by VHEField/BHE sizing
    )

    return VHEField(
        id_=1,
        pipe=pipe,
        borehole=borehole,
        grout=grout,
        coordinates=coordinates,
        shank_spacing=field_parameters.shank_spacing,
        length_borehole=120.0,  # placeholder — will be sized
        burial_depth=field_parameters.burial_depth,
        tilt_rad=field_parameters.tilt_rad,
        orientation_rad=field_parameters.orientation_rad,
        heat_exchanger_type=field_parameters.heat_exchanger_type,
    )