from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Optional

from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.core.annulus import Annulus
from pythermonet.core.material import Material

HeatExchangerType = Literal["1U", "2U", "COAX"]


@dataclass
class VHEField:
    ID: int
    HE: HeatExchangerType                 # "1U", "2U", "COAX"

    pipe: PipeSegment                     # ét fysisk rør (geometri + materiale)
    borehole: Annulus                     # borehul (evt. casing)
    grout: Material                       # udfyldning mellem rør og borehul

    coordinates: List[list]               # [[x, y, z], ...] (én pr. HE)
    shankSpacing: float                   # m (kun relevant for 2U)

    # -----------------------------
    # Pygfunction adapter
    # -----------------------------
    def to_pygfunction_boreholes(
        self,
        *,
        H_m: float,
        D_m: float = 0.0,
        tilt_deg: float = 0.0,
        orientation_deg: float = 0.0,
        use_z_as_depth: bool = False,
        r_b_m: Optional[float] = None,
    ):
        """
        Returnerer en liste af pygfunction Borehole-objekter for borehole field layout.

        Parametre
        ---------
        H_m:
            Borehole-længde (m) (pygfunction: H).
        D_m:
            Dybde til borehole top (m) under terræn (pygfunction: D). Default 0.
        tilt_deg, orientation_deg:
            Borehole tilt/orientation (grader). Default 0 (vertikal).
        use_z_as_depth:
            Hvis True tolkes koordinatens z som "D" (dybde til top) pr borehole.
            (Typisk: z=0 betyder terræn; z>0 nedad). Hvis z i stedet er kote/elevation,
            så hold denne False og brug D_m.
        r_b_m:
            Overstyring af borehole-radius (m). Hvis None forsøges radius udledt af self.borehole.

        Bemærk
        ------
        - pygfunction bruger kun (x, y) til placering i plan.
        - z håndteres kun, hvis du eksplicit beder om det via use_z_as_depth.
        """
        import math

        # Lokal import for at holde afhængigheden isoleret
        import pygfunction as gt

        coords = self._validate_coordinates(self.coordinates)

        # Borehole radius
        r_b = r_b_m if r_b_m is not None else self._infer_borehole_radius_m()

        # Konverter grader til radianer som pygfunction forventer
        tilt = math.radians(float(tilt_deg))
        orientation = math.radians(float(orientation_deg))

        boreholes = []
        for c in coords:
            x, y, z = float(c[0]), float(c[1]), float(c[2])

            D = float(z) if use_z_as_depth else float(D_m)

            boreholes.append(
                gt.boreholes.Borehole(
                    H=float(H_m),
                    D=D,
                    r_b=float(r_b),
                    x=x,
                    y=y,
                    tilt=tilt,
                    orientation=orientation,
                )
            )

        return boreholes

    def boreholes_signature(self, *, H_m: float, D_m: float = 0.0, r_b_m: Optional[float] = None) -> str:
        """
        Stabil signatur til caching: layout + nøgleparametre.
        (Bevidst enkel; kan udvides senere).
        """
        r_b = r_b_m if r_b_m is not None else self._infer_borehole_radius_m()
        xy = [(float(c[0]), float(c[1])) for c in self._validate_coordinates(self.coordinates)]
        return f"VHEField:{self.ID}|HE:{self.HE}|n:{len(xy)}|H:{float(H_m)}|D:{float(D_m)}|r_b:{float(r_b)}|xy:{xy}"

    # -----------------------------
    # Intern hjælp
    # -----------------------------
    @staticmethod
    def _validate_coordinates(coords: Sequence[Sequence[float]]) -> List[List[float]]:
        if not isinstance(coords, (list, tuple)) or len(coords) == 0:
            raise ValueError("VHEField.coordinates skal være en ikke-tom liste af [x, y, z].")

        out: List[List[float]] = []
        for i, c in enumerate(coords):
            if not isinstance(c, (list, tuple)) or len(c) != 3:
                raise ValueError(f"coordinates[{i}] skal have format [x, y, z]. Fik: {c}")
            out.append([float(c[0]), float(c[1]), float(c[2])])
        return out

    def _infer_borehole_radius_m(self) -> float:
        """
        Forsøger at udlede borehole radius (m) fra Annulus.
        Du skal evt. justere felt-navn afhængigt af din Annulus-implementering.

        Typiske navne:
        - outerDiameter (m)
        - OD (m)
        - outer_diameter_m (m)

        Fallback: kaster fejl, så radius eksplicit gives via r_b_m.
        """
        candidates = ["outerDiameter", "OD", "outer_diameter_m", "outer_diameter"]
        for name in candidates:
            if hasattr(self.borehole, name):
                od = float(getattr(self.borehole, name))
                if od <= 0:
                    break
                return 0.5 * od

        raise ValueError(
            "Kunne ikke udlede borehole radius fra VHEField.borehole (Annulus). "
            "Angiv r_b_m eksplicit eller tilpas _infer_borehole_radius_m() til dine feltnavne."
        )