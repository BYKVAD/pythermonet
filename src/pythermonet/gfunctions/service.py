from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import numpy as np

from .models import GFunctionRequest, GFunctionSet
from .pygfunction_impl import compute_gvalues_pygfunction

def _hash_request(req: GFunctionRequest) -> str:
    # Stabil hash uden at være afhængig af numpy repræsentation tilfældigt
    payload = {
        "evaluation_times": np.asarray(req.evaluation_times, dtype=float).tolist(),
        "thermal_diffusivity_soil": float(req.thermal_diffusivity_soil),
        "method": req.method,
        "boundary_condition": req.boundary_condition,
        "pygfunction_options": req.pygfunction_options or {},
        "boreholes_signature": getattr(req.boreholes, "signature", None),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

class GFunctionsService:
    def __init__(self, cache=None):
        self.cache = cache  # fx DiskCache eller simpel dict

    def compute(self, req: GFunctionRequest) -> GFunctionSet:
        key = _hash_request(req)

        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        g = compute_gvalues_pygfunction(req)

        if self.cache is not None:
            self.cache.set(key, g)

        return g
