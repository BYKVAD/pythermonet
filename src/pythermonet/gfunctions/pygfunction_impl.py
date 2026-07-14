from __future__ import annotations
import numpy as np
from .models import GFunctionRequest, GFunctionSet

def compute_gvalues_pygfunction(req: GFunctionRequest) -> GFunctionSet:
    import pygfunction as gt

    times = np.asarray(req.evaluation_times, dtype=float)
    boreholes = req.boreholes
    options = req.pygfunction_options or {}

    # Gyldig pygfunction solver-metode
    solver = req.method

    gfunc = gt.gfunction.gFunction(
        boreholes,          # boreholes_or_network (positionel)
        req.thermal_diffusivity_soil,     # alpha (positionel)
        time=times,
        method=solver,
        boundary_condition=req.boundary_condition,
        options=options,
    )

    g_values = getattr(gfunc, "gFunc", None)
    if g_values is None:
        g_values = getattr(gfunc, "gFunction", None)
    if g_values is None:
        raise AttributeError("Could not extract g-values from pygfunction gFunction object.")

    g_values = np.asarray(g_values, dtype=float)

    # --- Audit metadata (serialiserbart) ---
    meta = {
        "engine": "pygfunction",
        "pygfunction_version": getattr(gt, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "solver_method": solver,
        "boundary_condition": req.boundary_condition,
        "thermal_diffusivity_soil": float(req.thermal_diffusivity_soil),
        "n_boreholes": int(len(boreholes)),
        "time_grid": {
            "n": int(times.size),
            "t_min_s": float(times.min()),
            "t_max_s": float(times.max()),
        },
        "options": options,
    }

    # Geometri-snapshot (udtræk det vi kan uden at afhænge af interne pygfunction typer)
    try:
        meta["boreholes_geometry"] = {
            "H_m": float(getattr(boreholes[0], "H")),
            "D_m": float(getattr(boreholes[0], "D")),
            "r_b_m": float(getattr(boreholes[0], "r_b")),
            "xy": [[float(b.x), float(b.y)] for b in boreholes],
        }
    except Exception:
        # Hvis pygfunction ændrer API, vil vi stadig have et brugbart meta-sæt
        meta["boreholes_geometry"] = "unavailable"

    return GFunctionSet(evaluation_times=times, g_values=g_values, metadata=meta)
