from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipeCatalogueEntry:
    """
    Single pipe catalogue entry.

    Represents one commercially available pipe geometry,
    typically defined by outer diameter and SDR.
    """
    outer_diameter: float     # m
    SDR: float                # -
    
    # Optional metadata (can be None)
    name: str | None = None   # e.g. "PE100 DN63 SDR11"
    material: str | None = None
