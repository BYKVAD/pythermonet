from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any

import numpy as np
import pygfunction as gt
from ..core.pipe_segment import PipeSegment
from ..core.annulus import Annulus
from ..core.material import Material


@dataclass(frozen=True, slots=True)
class VHEField:
    ID: int
    HE: str
    pipe: PipeSegment
    borehole: Annulus
    grout: Material
    coordinates: Sequence[Sequence[float]]
    shankSpacing: float

    H_m: float
    D_m: float
    use_z_as_depth: bool = False
    tilt_rad: float = 0.0
    orientation_rad: float = 0.0

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
        if self.shankSpacing <= 0.0:
            raise ValueError("shankSpacing must be > 0.")

        if self.H_m <= 0.0:
            raise ValueError("H_m must be > 0.")
        if self.D_m < 0.0:
            raise ValueError("D_m must be >= 0.")

        object.__setattr__(
            self,
            "coordinates",
            tuple(tuple(float(v) for v in row) for row in coords),
        )
        object.__setattr__(self, "shankSpacing", float(self.shankSpacing))
        object.__setattr__(self, "H_m", float(self.H_m))
        object.__setattr__(self, "D_m", float(self.D_m))
        object.__setattr__(self, "use_z_as_depth", bool(self.use_z_as_depth))
        object.__setattr__(self, "tilt_rad", float(self.tilt_rad))
        object.__setattr__(self, "orientation_rad", float(self.orientation_rad))

    @property
    def ndim(self) -> int:
        return len(self.coordinates[0])

    @property
    def n_boreholes(self) -> int:
        return len(self.coordinates)

    @property
    def x_m(self) -> tuple[float, ...]:
        return tuple(row[0] for row in self.coordinates)

    @property
    def y_m(self) -> tuple[float, ...]:
        return tuple(row[1] for row in self.coordinates)

    @property
    def z_m(self) -> tuple[float, ...] | None:
        if self.ndim == 3:
            return tuple(row[2] for row in self.coordinates)
        return None

    @property
    def xy_m(self) -> tuple[tuple[float, float], ...]:
        return tuple((row[0], row[1]) for row in self.coordinates)

    @property
    def r_b_m(self) -> float:
        """Borehole radius [m], derived from borehole.outer_diameter."""
        return float(self.borehole.outer_diameter) / 2.0

    def to_pygfunction_boreholes(self) -> list[gt.boreholes.Borehole]:
        xy = np.asarray(self.xy_m, dtype=float)
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
                D_use = self.D_m

            boreholes.append(
                gt.boreholes.Borehole(
                    H=self.H_m,
                    D=D_use,
                    r_b=self.r_b_m,
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