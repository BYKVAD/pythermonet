from dataclasses import dataclass
from typing import List, Literal

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
