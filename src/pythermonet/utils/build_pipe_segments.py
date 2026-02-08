from dataclasses import dataclass

from core.annulus import Annulus
from core.material import Material
from core.pipe import Pipe
from core.pipe_segment import PipeSegment
  
@dataclass(frozen=True)
class PipeFactory:
    material: Material
    roughnessHeight: float  # m

    def pipe_from_annulus(self, a: Annulus) -> Pipe:
        return Pipe(
            outerDiameter=a.outerDiameter,
            SDR=a.SDR,
            material=self.material,
            roughnessHeight=self.roughnessHeight,
        )

    def segment_from_annulus(self, a: Annulus, *, ID: int, length: float) -> PipeSegment:
        return PipeSegment(
            ID=ID,
            length=length,
            outerDiameter=a.outerDiameter,
            SDR=a.SDR,
            material=self.material,
            roughnessHeight=self.roughnessHeight,
        )
