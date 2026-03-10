from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.physics.thermal_resistances import pipe_thermal_resistance
from pythermonet.physics.sources import ils, csm


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class PipeGroup:
    """
    Fysisk og hydraulisk input for én rørgruppe.
    Alle størrelser i SI.
    """
    ID: int
    L_m: float
    Di_m: float
    Do_m: float
    Re: float
    k_pipe_W_mK: float
    burial_depth_m: float
    n_parallel_pipes: int
    n_traces: int
    pipe_spacing_m: float | None = None


@dataclass(frozen=True)
class ModeInput:
    """
    Input for én driftstilstand, fx heating eller cooling.
    """
    times_s: np.ndarray
    powers_W: np.ndarray
    Ti_C: float
    To_C: float


@dataclass(frozen=True)
class DistributionPipeModel:
    """
    Samlet modelinput for distributionsrør.
    """
    brine: HeatCarrier
    soil: Soil

    T0_C: float
    surface_amp_C: float

    pipe_groups: list[PipeGroup]

    heating: ModeInput
    cooling: ModeInput | None = None


@dataclass(frozen=True)
class ModeResult:
    """
    Resultat for én mode.
    """
    T_dimv_C: np.ndarray
    F_total: float
    F_per_group: np.ndarray


# -----------------------------
# Validation
# -----------------------------

def _validate_mode_input(mode_input: ModeInput, mode_name: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(mode_input.times_s, dtype=float)
    powers = np.asarray(mode_input.powers_W, dtype=float)

    if times.ndim != 1:
        raise ValueError(f"{mode_name}: times_s must be a 1D array")
    if powers.ndim != 1:
        raise ValueError(f"{mode_name}: powers_W must be a 1D array")
    if times.shape != powers.shape:
        raise ValueError(f"{mode_name}: times_s and powers_W must have same shape")
    if times.size < 1:
        raise ValueError(f"{mode_name}: times_s and powers_W must have length >= 1")
    if np.any(times <= 0):
        raise ValueError(f"{mode_name}: all times_s must be > 0")

    return times, powers


def _validate_pipe_group(pg: PipeGroup) -> None:
    if pg.L_m <= 0:
        raise ValueError(f"PipeGroup ID={pg.ID}: L_m must be > 0")
    if pg.Di_m <= 0 or pg.Do_m <= 0:
        raise ValueError(f"PipeGroup ID={pg.ID}: Di_m and Do_m must be > 0")
    if pg.Do_m < pg.Di_m:
        raise ValueError(f"PipeGroup ID={pg.ID}: Do_m must be >= Di_m")
    if pg.burial_depth_m < 0:
        raise ValueError(f"PipeGroup ID={pg.ID}: burial_depth_m must be >= 0")
    if pg.n_parallel_pipes not in (1, 2):
        raise ValueError(
            f"PipeGroup ID={pg.ID}: only one- and two-pipe systems are supported "
            f"(n_parallel_pipes must be 1 or 2)"
        )
    if pg.n_traces <= 0:
        raise ValueError(f"PipeGroup ID={pg.ID}: n_traces must be > 0")
    if pg.n_parallel_pipes == 2 and pg.pipe_spacing_m is None:
        raise ValueError(
            f"PipeGroup ID={pg.ID}: pipe_spacing_m must be set when n_parallel_pipes == 2"
        )
    if pg.pipe_spacing_m is not None and pg.pipe_spacing_m <= 0:
        raise ValueError(f"PipeGroup ID={pg.ID}: pipe_spacing_m must be > 0 when provided")


def _validate_model(model: DistributionPipeModel) -> None:
    if not model.pipe_groups:
        raise ValueError("pipe_groups must contain at least one PipeGroup")

    for pg in model.pipe_groups:
        _validate_pipe_group(pg)


# -----------------------------
# Helpers
# -----------------------------

def _prandtl(brine: HeatCarrier) -> float:
    """
    Pr = mu * cp / k
    """
    return float(brine.dynamicViscosity * brine.c / brine.thermalCond)


def _shallow_k(soil: Soil, mode: str) -> float:
    if mode == "heating":
        k = float(soil.thermalCondShallowHeating)
    elif mode == "cooling":
        k = float(soil.thermalCondShallowCooling)
    else:
        raise ValueError("mode must be 'heating' or 'cooling'")

    if k <= 0:
        raise ValueError(f"Shallow thermal conductivity must be > 0 for mode={mode}. Got {k}.")

    return k


def _soil_diffusivity_shallow(soil: Soil, mode: str) -> float:
    """
    a = k / (rho * cp)
    """
    k = _shallow_k(soil, mode)
    rho = float(soil.rho)
    cp = float(soil.c)

    if rho <= 0 or cp <= 0:
        raise ValueError(f"Soil rho and c must be > 0. Got rho={rho}, c={cp}.")

    return float(k / (rho * cp))


def _temp_penalty_at_depth(surface_amp_C: float, depth_m: float, diffusivity_m2_s: float) -> float:
    """
    Temperaturdæmpning af årlig overfladevariation i dybden.
    """
    omega = 2.0 * math.pi / (86400.0 * 365.25)
    return float(
        surface_amp_C * math.exp(-depth_m * math.sqrt(omega / (2.0 * diffusivity_m2_s)))
    )


def _delta_p(powers_W: np.ndarray) -> np.ndarray:
    """
    Legacy-kompatibel dP-definition:
      dP[0] = P[0]
      dP[1:] = diff(P)
    """
    P = np.asarray(powers_W, dtype=float)

    if P.ndim != 1 or P.size < 1:
        raise ValueError("powers_W must be a 1D array of length >= 1")

    dP = np.zeros_like(P, dtype=float)
    dP[0] = P[0]
    if P.size > 1:
        dP[1:] = np.diff(P)

    return dP


def _bundle_k1(
    diffusivity_m2_s: float,
    times_s: np.ndarray,
    depth_m: float,
    *,
    n_parallel_pipes: int,
    spacing_m: float | None,
) -> np.ndarray:
    """
    Repræsentativ K1 for 1- og 2-rørs system i samme dybde z med isoterm overflade.

    1-rør:
        K1(t) = -ILS(2z)

    2-rør:
        K1(t) = -ILS(2z) + ILS(s) - ILS(sqrt(s^2 + 4z^2))
    """
    if n_parallel_pipes == 1:
        return -ils(diffusivity_m2_s, times_s, 2.0 * depth_m)

    if n_parallel_pipes == 2:
        if spacing_m is None:
            raise ValueError("spacing_m must be set when n_parallel_pipes == 2")
        s = float(spacing_m)
        return (
            -ils(diffusivity_m2_s, times_s, 2.0 * depth_m)
            + ils(diffusivity_m2_s, times_s, s)
            - ils(diffusivity_m2_s, times_s, math.sqrt(s * s + 4.0 * depth_m * depth_m))
        )

    raise ValueError("Only one- and two-pipe systems are supported")


def _mode_numerator(T0_C: float, Tm_C: float, TP_C: float, mode: str) -> tuple[float, float]:
    """
    Returnerer:
      numerator
      sign for TP in temperature reconstruction

    heating:
      num = T0 - Tm - TP
      T = T0 - TP - ...

    cooling:
      num = Tm - T0 - TP
      T = T0 + TP - ...
    """
    if mode == "heating":
        return (T0_C - Tm_C - TP_C, -1.0)
    if mode == "cooling":
        return (Tm_C - T0_C - TP_C, +1.0)

    raise ValueError("mode must be 'heating' or 'cooling'")


# -----------------------------
# Core computation
# -----------------------------

def _compute_mode(
    model: DistributionPipeModel,
    mode_input: ModeInput,
    *,
    mode: str,
) -> ModeResult:
    times_s, powers_W = _validate_mode_input(mode_input, mode)

    Pr = _prandtl(model.brine)
    a_s = _soil_diffusivity_shallow(model.soil, mode)
    k_s = _shallow_k(model.soil, mode)

    dP = _delta_p(powers_W)

    Ti_C = float(mode_input.Ti_C)
    To_C = float(mode_input.To_C)
    Tm_C = 0.5 * (Ti_C + To_C)

    n_groups = len(model.pipe_groups)
    n_times = times_s.size

    F_per_group = np.zeros(n_groups, dtype=float)
    T_volume_weighted = np.zeros((n_groups, n_times), dtype=float)

    total_volume_m3 = 0.0

    for i, pg in enumerate(model.pipe_groups):
        TP_C = _temp_penalty_at_depth(
            surface_amp_C=model.surface_amp_C,
            depth_m=pg.burial_depth_m,
            diffusivity_m2_s=a_s,
        )

        R_pipe = pipe_thermal_resistance(
            Di=float(pg.Di_m),
            Do=float(pg.Do_m),
            Re=float(pg.Re),
            Pr=float(Pr),
            k_fluid=float(model.brine.thermalCond),
            k_pipe=float(pg.k_pipe_W_mK),
        )

        K1 = _bundle_k1(
            diffusivity_m2_s=a_s,
            times_s=times_s,
            depth_m=pg.burial_depth_m,
            n_parallel_pipes=pg.n_parallel_pipes,
            spacing_m=pg.pipe_spacing_m,
        )

        G_grid = csm(
            float(pg.Do_m) / 2.0,
            float(pg.Do_m) / 2.0,
            times_s,
            a_s,
        ) + K1

        kernel = ((G_grid / k_s) + R_pipe) / float(pg.L_m) / float(pg.n_traces)
        denom = float(np.dot(dP, kernel))

        numerator, tp_sign = _mode_numerator(
            T0_C=float(model.T0_C),
            Tm_C=Tm_C,
            TP_C=TP_C,
            mode=mode,
        )

        F_i = float(numerator / denom)
        F_per_group[i] = F_i

        T_seq_C = float(model.T0_C) + tp_sign * TP_C - F_i * np.cumsum(dP * kernel)

        volume_m3 = float(pg.L_m) * math.pi * float(pg.Di_m) ** 2 / 4.0
        total_volume_m3 += volume_m3
        T_volume_weighted[i, :] = T_seq_C * volume_m3

    T_dimv_C = np.sum(T_volume_weighted, axis=0) / float(total_volume_m3)
    F_total = float(np.sum(F_per_group))

    return ModeResult(
        T_dimv_C=T_dimv_C,
        F_total=F_total,
        F_per_group=F_per_group,
    )


def compute_distribution_pipe_thermal_response(
    model: DistributionPipeModel,
) -> dict[str, ModeResult]:
    """
    Hoved-entry point.

    Returnerer:
        {
            "heating": ModeResult(...),
            "cooling": ModeResult(...),   # kun hvis cooling er angivet
        }
    """
    _validate_model(model)

    results: dict[str, ModeResult] = {
        "heating": _compute_mode(model, model.heating, mode="heating")
    }

    if model.cooling is not None:
        results["cooling"] = _compute_mode(model, model.cooling, mode="cooling")

    return results