from dataclasses import dataclass
from pythermonet.core.annulus import Annulus

@dataclass(frozen=True)
class AnnulusSpec:
    std_type: str
    nominal_width_mm: float
    outerDiameter: float   # m
    SDR: float

    def to_annulus(self) -> Annulus:
        return Annulus(outerDiameter=self.outerDiameter, SDR=self.SDR)
