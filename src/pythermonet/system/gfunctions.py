"""System-level g-function gateways for pythermonet.

Includes support for both pygfunction and HHE horizontal-loop gfunction.
"""
from __future__ import annotations

import numpy as np

from pythermonet.gfunctions.pygfunction_impl import compute_gvalues_pygfunction
from pythermonet.simulation.HHE import gfunction as compute_gvalues_hhe


def compute_gfunction_pygfunction(
    coords_xy_m,
    time_s,
    alpha_m2s,
    borehole_length,
    r_b_m,
    method="equivalent",
    options=None,
):
    # The existing pyro module expects a pygfunction style request object
    # but this function is a lightweight compatibility adapter for legacy code.
    from pythermonet.gfunctions.models import GFunctionRequest
    from pythermonet.gfunctions.models import GFunctionSet

    # For compatibility, treat coords_xy_m as list of boreholes in this context.
    # The actual pygfunction engine wants pygfunction borehole objects; callers
    # should still be responsible for correct types in this path.
    req = GFunctionRequest(
        evaluation_times=np.asarray(time_s, dtype=float),
        boreholes=coords_xy_m,
        thermal_diffusivity_soil=float(alpha_m2s),
        method=method,
        pygfunction_options=options or {},
    )
    gset = compute_gvalues_pygfunction(req)

    # Convert to compatibility object (duck-type minimal API)
    class _G:
        def __init__(self, evaluation_times, g_values, metadata):
            self.evaluation_times = evaluation_times
            self.g = np.asarray(g_values, dtype=float)
            self.metadata = metadata

    return _G(gset.evaluation_times, gset.g_values, gset.metadata)


def compute_gfunction_infinite_medium(
    coords_xy_m,
    time_s,
    alpha_m2s,
    borehole_length,
    r_b_m,
    D_large_m,
    method="equivalent",
    options=None,
):
    """Compatibility adapter used by thermal_dimensioning.HHE path.

    - If input is a pythermonet HHE horizontal loop (coords), then run HHE gfunction.
    - Otherwise fallback to pygfunction full-space calculation.
    """
    # If coords_xy_m are raw np coordinates, attempt to run HHE field gfunction.
    if isinstance(coords_xy_m, np.ndarray) and coords_xy_m.ndim == 2:
        # Build a minimal PipeInfrastructure representation for HHE
        # A single-lateral “equivalent” line is still roughly approximated as 1 pipe.
        from pythermonet.components.pipe_infrastructure import PipeInfrastructure
        from pythermonet.core.pipe_segment import PipeSegment
        from pythermonet.core.material import Material

        # 1D areal field approximated as an equivalent 1-pipe segment.
        # This is conservative and can be improved as needed by caller.
        material = Material(density=950.0, specific_heat=1900.0, thermal_conductivity=0.4)
        segment = PipeSegment(diameter_outer=2*0.016, sdr=11, material=material, roughness=1.5e-5, id_=0, length=float(borehole_length))

        pi = PipeInfrastructure(
            n_pipes_parallel=max(1, int(coords_xy_m.shape[0])),
            segments_trace=[segment],
            pipe_spacing=1.0,
            burial_depth=float(r_b_m),
        )

        g_values = compute_gvalues_hhe(pipe_infrastructure=pi, soil_thermal_conductivity=1.0, soil_thermal_diffusivity=float(alpha_m2s), evaluation_times=np.asarray(time_s, dtype=float))

        class _G:
            def __init__(self, g):
                self.g = np.asarray(g, dtype=float)

        return _G(g_values)

    # Fallback: run pygfunction-based infinite-medium model
    return compute_gfunction_pygfunction(
        coords_xy_m=coords_xy_m,
        time_s=time_s,
        alpha_m2s=alpha_m2s,
        borehole_length=borehole_length,
        r_b_m=r_b_m,
        method=method,
        options=options,
    )
