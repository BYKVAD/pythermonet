from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil

from pythermonet.simulation.distribution_pipe_thermal import (
    PipeGroupThermalInput,
    PulseResponseSpec,
    PipeThermalResponseModeInput,
    DistributionPipeThermalInput,
    DistributionPipeThermalResult,
    compute_distribution_pipe_thermal_response,
)


def _first_attr(obj, names: list[str], default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def _times_3pulse_s(*, year_days: float = 365.25, season_days: float = 180.0, peak_h: float = 4.0) -> np.ndarray:
    # Konvention: [year, season, peak]
    return np.asarray([year_days * 86400.0, season_days * 86400.0, peak_h * 3600.0], dtype=float)


def _pulse_spec(times_s: np.ndarray, powers_W: np.ndarray) -> PulseResponseSpec:
    t = np.asarray(times_s, dtype=float)
    p = np.asarray(powers_W, dtype=float)
    if t.shape != p.shape:
        raise ValueError("PulseResponseSpec requires times_s and powers_W of same shape")
    if t.ndim != 1 or t.size < 1:
        raise ValueError("times_s must be 1D length>=1")
    return PulseResponseSpec(times_s=t, powers_W=p)


def _build_pipe_groups_from_network(
    *,
    network,
    k_pipe_W_mK: float,
    burial_depth_m: float,
    n_pipes: int,
    pipe_spacing_m: Optional[float],
    use_mode: str,  # "heating" | "cooling"
) -> list[PipeGroupThermalInput]:
    """
    Mapper et dimensioneret DistributionNetwork til PipeGroupThermalInput pr trace-gruppe.

    Kræver efter run_pipedimensioning:
      - network.L_traces (one-way) [m]
      - network.SDR
      - network.infrastructure.traceSegments[i].outerDiameter (dimensioneret) [m]
      - network.dimensionedPipeReynoldsNumberHeating / Cooling
    """
    L_oneway = np.asarray(network.L_traces, dtype=float)
    SDR = np.asarray(network.SDR, dtype=float)

    if use_mode == "heating":
        Re_arr = np.asarray(network.dimensionedPipeReynoldsNumberHeating, dtype=float)
    elif use_mode == "cooling":
        Re_arr = np.asarray(network.dimensionedPipeReynoldsNumberCooling, dtype=float)
    else:
        raise ValueError("use_mode must be 'heating' or 'cooling'")

    Do = np.asarray([float(seg.outerDiameter) for seg in network.infrastructure.traceSegments], dtype=float)

    if not (L_oneway.size == SDR.size == Re_arr.size == Do.size):
        raise ValueError("Network arrays must have same length (trace group count)")

    # Termisk model bruger L_m i denominatoren.
    # Vi matcher din hydrauliske antagelse: forward+return => 2*one-way.
    L_m = 2.0 * L_oneway
    Di = Do * (1.0 - 2.0 / SDR)

    out: list[PipeGroupThermalInput] = []
    for i in range(L_m.size):
        out.append(
            PipeGroupThermalInput(
                ID=int(i),
                L_m=float(L_m[i]),
                Di_m=float(Di[i]),
                Do_m=float(Do[i]),
                Re=float(Re_arr[i]),
                k_pipe_W_mK=float(k_pipe_W_mK),
                burial_depth_m=float(burial_depth_m),
                n_pipes=int(n_pipes),
                pipe_spacing_m=float(pipe_spacing_m) if (n_pipes > 1 and pipe_spacing_m is not None) else None,
            )
        )
    return out


def run_distribution_pipe_thermal(
    *,
    network,
    brine: HeatCarrier,
    soil: Soil,
    heat_pumps,  # HeatPumps (din nuværende klasse)
    # Mode-temperaturer (de Tm-værdier der skal bruges i responsen)
    Ti_heat_C: float,
    To_heat_C: float,
    Ti_cool_C: Optional[float] = None,
    To_cool_C: Optional[float] = None,
    # Pulsvarigheder (3-puls)
    peak_heating_h: float = 4.0,
    peak_cooling_h: float = 4.0,
    season_days: float = 180.0,
    year_days: float = 365.25,
    # Rør-/site antagelser (kan overstyres eller hentes fra network hvis du gemmer dem dér)
    burial_depth_m: Optional[float] = None,
    k_pipe_W_mK: Optional[float] = None,
    n_parallel_pipes: Optional[int] = None,
    pipe_spacing_m: Optional[float] = None,
    # Boundary conditions
    T0_C: Optional[float] = None,
    surface_amp_C: Optional[float] = None,
    # Cooling toggle
    enable_cooling: bool = True,
) -> DistributionPipeThermalResult:
    """
    One-liner runner til distributionsrør-termik.

    Forventer at heat_pumps indeholder 3-puls loads:
      - heating_ground_load_eff_W (eller heating_ground_load_W i v2)
      - cooling_ground_load_eff_W (optional)
    """

    # --- Defaults fra objekter hvis muligt ---
    # T0 og amplitude: fra soil hvis ikke givet
    T0_C = float(T0_C if T0_C is not None else _first_attr(soil, ["surfaceTemp"], 9.0))
    surface_amp_C = float(surface_amp_C if surface_amp_C is not None else _first_attr(soil, ["surfaceTempAmp"], 7.0))

    # burial depth / pipe spacing / parallel pipes: prøv at hente fra network (hvis du har felter), ellers fallback
    burial_depth_m = float(
        burial_depth_m
        if burial_depth_m is not None
        else _first_attr(network, ["burial_depth", "burialDepth", "burial_depth_m", "burialDepth_m"], 1.2)
    )
    n_parallel_pipes = int(
        n_parallel_pipes
        if n_parallel_pipes is not None
        else _first_attr(network, ["n_parallel_pipes", "nParallelPipes"], 1)
    )
    pipe_spacing_m = (
        float(pipe_spacing_m)
        if pipe_spacing_m is not None
        else _first_attr(network, ["pipe_distance", "pipeDistance", "pipe_distance_m", "pipeDistance_m"], None)
    )

    # k_pipe: brug materialets k hvis ikke givet
    if k_pipe_W_mK is None:
        # forsøg: network.infrastructure.traceSegments[0].material.thermalCond
        seg0 = network.infrastructure.traceSegments[0]
        k_pipe_W_mK = float(_first_attr(seg0, ["material"], None).thermalCond) if hasattr(seg0, "material") else 0.4
    k_pipe_W_mK = float(k_pipe_W_mK)

    # --- Loads: brug dine eksisterende felter (du bruger heating_ground_load_eff_W i din main) ---
    if hasattr(heat_pumps, "heating_ground_load_eff_W"):
        P_heat_3_W = np.asarray(heat_pumps.heating_ground_load_eff_W, dtype=float)
    else:
        P_heat_3_W = np.asarray(heat_pumps.heating_ground_load_W, dtype=float)

    times_heat_s = _times_3pulse_s(year_days=year_days, season_days=season_days, peak_h=peak_heating_h)

    heating_mode = PipeThermalResponseModeInput(
        spec=_pulse_spec(times_heat_s, P_heat_3_W),
        Ti_C=float(Ti_heat_C),
        To_C=float(To_heat_C),
    )

    pipe_groups = _build_pipe_groups_from_network(
        network=network,
        k_pipe_W_mK=k_pipe_W_mK,
        burial_depth_m=burial_depth_m,
        n_pipes=n_parallel_pipes,
        pipe_spacing_m=pipe_spacing_m,
        use_mode="heating",
    )

    cooling_mode = None
    if enable_cooling and getattr(heat_pumps, "cooling_ground_load_eff_W", None) is not None:
        if Ti_cool_C is None or To_cool_C is None:
            raise ValueError("Cooling enabled, but Ti_cool_C/To_cool_C not provided")

        P_cool_3_W = np.asarray(heat_pumps.cooling_ground_load_eff_W, dtype=float)
        times_cool_s = _times_3pulse_s(year_days=year_days, season_days=season_days, peak_h=peak_cooling_h)

        cooling_mode = PipeThermalResponseModeInput(
            spec=_pulse_spec(times_cool_s, P_cool_3_W),
            Ti_C=float(Ti_cool_C),
            To_C=float(To_cool_C),
        )

        # (valgfrit) du kan vælge at bygge pipe_groups igen med cooling-Re
        # hvis du vil bruge cooling-Re i termikken. Default: heating-Re.
        # pipe_groups = _build_pipe_groups_from_network(... use_mode="cooling")

    inp = DistributionPipeThermalInput(
        brine=brine,
        soil=soil,
        T0_C=T0_C,
        surface_amp_C=surface_amp_C,
        pipe_groups=pipe_groups,
        heating=heating_mode,
        cooling=cooling_mode,
    )

    return compute_distribution_pipe_thermal_response(inp)
