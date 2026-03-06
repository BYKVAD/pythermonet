from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.physics.thermal_resistances import pipe_thermal_resistance
from pythermonet.physics.sources import ils, csm  # dine funktioner (ils/csm) ligger her


# -----------------------------
# Data contracts
# -----------------------------

@dataclass(frozen=True)
class PipeGroupThermalInput:
    """
    Termisk input per pipe group (typisk output fra dimensionering + netværksgeometri).
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
    pipe_spacing_m: float | None = None  # kræves hvis n_pipes > 1


@dataclass(frozen=True)
class PulseResponseSpec:
    """
    Generisk pulse-spec til superposition:
      - times_s: evaluerings-tider (sekunder), længde N
      - powers_W: lastniveauer (W), længde N
        Legacy-kompatibelt: P=[annual, month, peak] => dP=[P0, diff(P)]
    """
    times_s: np.ndarray
    powers_W: np.ndarray


@dataclass(frozen=True)
class PipeThermalResponseModeInput:
    """
    Input for én mode (heating/cooling).
    """
    spec: PulseResponseSpec
    Ti_C: float
    To_C: float


@dataclass(frozen=True)
class DistributionPipeThermalInput:
    """
    Samlet input for distributionsrør-respons.
    """
    brine: HeatCarrier
    soil: Soil

    T0: float               # undisturbed ground temp (reference)
    surface_amp: float      # amplitude af surface temp variation (A)

    pipe_groups: list[PipeGroupThermalInput]

    heating: PipeThermalResponseModeInput
    cooling: PipeThermalResponseModeInput | None = None  # optional


@dataclass(frozen=True)
class DistributionPipeThermalModeResult:
    """
    Resultat for én mode.
    """
    T_dimv_C: np.ndarray            # shape (N_times,), volumen-vægtet middel
    F_total: float                  # total fraction (sum over groups)
    F_per_group: np.ndarray         # shape (N_groups,)
    T_per_group_C: np.ndarray       # shape (N_groups, N_times), ikke volumen-vægtet


@dataclass(frozen=True)
class DistributionPipeThermalResult:
    heating: DistributionPipeThermalModeResult
    cooling: DistributionPipeThermalModeResult | None


# -----------------------------
# Helpers
# -----------------------------

def _prandtl(brine: HeatCarrier) -> float:
    # Pr = mu*cp/k
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
    # a = k/(rho*cp)  (brug shallow k for distributionsrør)
    k = _shallow_k(soil, mode)
    rho = float(soil.rho)
    cp = float(soil.c)
    if rho <= 0 or cp <= 0:
        raise ValueError(f"Soil rho and c must be > 0. Got rho={rho}, c={cp}.")
    return float(k / (rho * cp))


def _temp_penalty_at_depth(A: float, z: float, a_s: float) -> float:
    # TP = A * exp(-z * sqrt(omega/(2a)))
    omega = 2.0 * math.pi / (86400.0 * 365.25)
    return float(A * math.exp(-z * math.sqrt(omega / (2.0 * a_s))))


def _delta_p(powers_W: np.ndarray) -> np.ndarray:
    """
    Legacy-kompatibel dP-definition for vilkårlig længde N>=1:
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


def k1_pipe_bundle(
    a_s: float,
    t,
    z: float,
    *,
    n_parallel_pipes: int,
    spacing: float | None,
):
    """
    Repræsentativ K1 for 1- og 2-rørs system i samme dybde z med isoterm overflade (spejlkilder).

    - 1-rør (one-pipe): n_parallel_pipes = 1
      K1(t) = -ILS(2z)

    - 2-rør (two-pipe): n_parallel_pipes = 2 (rørafstand = spacing)
      For hvert rør:
        K1(t) = -ILS(2z) + ILS(s) - ILS(sqrt(s^2 + 4z^2))
      (begge rør har samme respons pga. symmetri, så “bundle”-respons = samme værdi)
    """
    if n_parallel_pipes not in (1, 2):
        raise ValueError("Only one- and two-pipe systems are supported: n_parallel_pipes must be 1 or 2.")

    # One-pipe
    if n_parallel_pipes == 1:
        return -ils(a_s, t, 2.0 * z)

    # Two-pipe
    if spacing is None:
        raise ValueError("spacing must be set when n_parallel_pipes == 2")

    s = float(spacing)
    return (
        -ils(a_s, t, 2.0 * z)
        + ils(a_s, t, s)
        - ils(a_s, t, math.sqrt(s * s + 4.0 * z * z))
    )


# -----------------------------
# Core computation
# -----------------------------

def _compute_mode(
    *,
    mode_name: str,  # "heating" | "cooling"
    inp: DistributionPipeThermalInput,
    mode: PipeThermalResponseModeInput,
    #sign_TP_in_T0_term: float,
) -> DistributionPipeThermalModeResult:
    """
    sign_TP_in_T0_term:
      heating: -1.0  -> T = T0 - TP - ...
      cooling: +1.0  -> T = T0 + TP - ... (legacy-konvention i dit snippet)
    """
    Pr = _prandtl(inp.brine)
    a_s = _soil_diffusivity_shallow(inp.soil, mode_name)
    k_s = _shallow_k(inp.soil, mode_name)

    times = np.asarray(mode.spec.times_s, dtype=float)
    powers = np.asarray(mode.spec.powers_W, dtype=float)
    if times.shape != powers.shape:
        raise ValueError(f"{mode_name}: times_s and powers_W must have same shape")
    if times.ndim != 1:
        raise ValueError(f"{mode_name}: times_s must be a 1D array")
    if np.any(times <= 0):
        raise ValueError(f"{mode_name}: all times_s must be > 0")
    
    dP = _delta_p(powers)

    Ti = float(mode.Ti_C)
    To = float(mode.To_C)
    Tm = 0.5 * (Ti + To)

    N = len(inp.pipe_groups)
    Nt = times.size

    F_per = np.zeros(N, dtype=float)
    T_per = np.zeros((N, Nt), dtype=float)
    Tvol = np.zeros((N, Nt), dtype=float)

    V_total = 0.0

    for i, pg in enumerate(inp.pipe_groups):
        if pg.L_m <= 0:
            raise ValueError(f"PipeGroup ID={pg.ID}: L_m must be > 0")
        if pg.Di_m <= 0 or pg.Do_m <= 0:
            raise ValueError(f"PipeGroup ID={pg.ID}: Di_m/Do_m must be > 0")
        if pg.Do_m < pg.Di_m:
            raise ValueError(f"PipeGroup ID={pg.ID}: Do_m must be >= Di_m")
        if pg.n_parallel_pipes > 1 and pg.pipe_spacing_m is None:
            raise ValueError(f"PipeGroup ID={pg.ID}: pipe_spacing_m must be set when n_parallel_pipes>1")

        TP = _temp_penalty_at_depth(inp.surface_amp, pg.burial_depth_m, a_s)

        # R_pipe [m*K/W]
        R = pipe_thermal_resistance(
            Di=float(pg.Di_m),
            Do=float(pg.Do_m),
            Re=float(pg.Re),
            Pr=float(Pr),
            k_fluid=float(inp.brine.thermalCond),
            k_pipe=float(pg.k_pipe_W_mK),
        )

        # G_grid(t): CSM + K1 (image + inter-pipe)
        #print("Times:", times)
        
        K1 = k1_pipe_bundle(a_s, times, pg.burial_depth_m, n_parallel_pipes=pg.n_parallel_pipes, spacing=pg.pipe_spacing_m)
        G_grid = csm(float(pg.Do_m) / 2.0, float(pg.Do_m) / 2.0, times, a_s) + K1  # shape (Nt,)
        #print(f"G_grid={G_grid}, R={R}, np={pg.n_parallel_pipes}, Space={pg.pipe_spacing_m}, a_s={a_s}")

        # Denominator: dot(dP, (G/k_s + R)/L)
        denom = float(np.dot(dP, ((G_grid / k_s) + R) / float(pg.L_m) / float(pg.n_traces)))

        # Numerator og fraktion
        # heating legacy: (T0 - Tm - TP)
        # cooling legacy: (Tm - T0 - TP)
        if mode_name == "heating":
            num = (inp.T0 - Tm - TP)
            sign_TP_in_T0_term = -1.0
        else:
            num = (Tm - inp.T0 - TP)
            sign_TP_in_T0_term = +1.0
        #print(f"inp.to ID={inp.T0}: Tm={Tm}, TP={TP}, pg.L_m={pg.L_m}")
        F_per[i] = float(num / denom)

        # Temperatursekvens via superposition:
        # heating: T = T0 - TP - F * cumsum(dP * (G/k_s + R)/L)
        # cooling: T = T0 + TP - F * cumsum(...)  (samme algebra, men TP-signeret via sign_TP_in_T0_term)
        kernel = dP * (((G_grid / k_s) + R) / float(pg.L_m) / float(pg.n_traces))
        T_seq = float(inp.T0) + sign_TP_in_T0_term * TP - F_per[i] * np.cumsum(kernel)

        T_per[i, :] = T_seq

        V_i = float(pg.L_m) * math.pi * float(pg.Di_m) ** 2 / 4.0
        V_total += V_i
        Tvol[i, :] = T_seq * V_i

    T_dimv = np.sum(Tvol, axis=0) / float(V_total)
    F_total = float(np.sum(F_per))
    print(f"{mode_name.capitalize()} mode: F_total={F_total}")

    return DistributionPipeThermalModeResult(
        T_dimv_C=T_dimv,
        F_total=F_total,
        F_per_group=F_per,
        T_per_group_C=T_per,
    )


def compute_distribution_pipe_thermal_response(inp: DistributionPipeThermalInput) -> DistributionPipeThermalResult:
    """
    Hoved-entry point.
    Understøtter vilkårligt antal responstider (N>=1) via PulseResponseSpec.
    """

    heat_res = _compute_mode(
        mode_name="heating",
        inp=inp,
        mode=inp.heating,
    )

    if inp.cooling is None:
        cool_res = None
    else:
        cool_res = _compute_mode(
            mode_name="cooling",
            inp=inp,
            mode=inp.cooling,
        )

    return DistributionPipeThermalResult(heating=heat_res, cooling=cool_res)