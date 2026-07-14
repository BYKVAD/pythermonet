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
    id_: int
    length: float                       # [m]
    diameter_inner: float               # [m]
    diameter_outer: float               # [m]
    reynolds_number: float
    thermal_conductivity_pipe: float    # [W/m/K]
    burial_depth: float                 # [m]
    n_parallel_pipes: int
    n_traces: int
    pipe_spacing: float | None = None   # [m]


@dataclass(frozen=True)
class ModeInput:
    """
    Input for én driftstilstand, fx heating eller cooling.

    temperature_heat_pump_inlet : float
        Brine temperature at the heat pump inlet — the design limit temperature
        (minimum for heating, maximum for cooling) [°C].
    temperature_heat_pump_outlet : float
        Brine temperature at the heat pump outlet — more extreme than the inlet
        by the heat pump delta-T (lower for heating, higher for cooling) [°C].
    """
    pulse_timescales: np.ndarray  # [s]
    pulse_loads: np.ndarray       # [W]
    temperature_heat_pump_inlet: float   # [°C]
    temperature_heat_pump_outlet: float  # [°C]


@dataclass(frozen=True)
class DistributionPipeModel:
    """
    Samlet modelinput for distributionsrør.
    """
    brine: HeatCarrier
    soil: Soil

    temperature_ground_undisturbed: float   # [°C]
    temperature_surface_amplitude: float    # [°C]

    pipe_groups: list[PipeGroup]

    heating: ModeInput
    cooling: ModeInput | None = None


@dataclass(frozen=True)
class ThermonetPerformance:
    """
    Thermal performance of the thermonet distribution pipes for one operational mode.

    Attributes
    ----------
    temperatures_mean : ndarray, shape (3,)
        Volume-weighted mean brine temperature in the distribution pipe network
        at [annual, seasonal, peak] sizing timescales [°C].
        Does not include BHE or HHE sources — only the thermonet pipes.
    load_supply_fraction : float
        Fraction of the total ground load supplied by the thermonet pipes
        through heat exchange with the surrounding soil. Used to reduce the
        BHE or HHE sizing load.
    load_supply_fractions_per_group : ndarray
        Same fraction broken down per pipe group.
    """
    temperatures_mean: np.ndarray              # [°C]
    load_supply_fraction: float
    load_supply_fractions_per_group: np.ndarray


# -----------------------------
# Validation
# -----------------------------

def _validate_mode_input(mode_input: ModeInput, mode_name: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(mode_input.pulse_timescales, dtype=float)
    powers = np.asarray(mode_input.pulse_loads, dtype=float)

    if times.ndim != 1:
        raise ValueError(f"{mode_name}: pulse_timescales must be a 1D array")
    if powers.ndim != 1:
        raise ValueError(f"{mode_name}: pulse_loads must be a 1D array")
    if times.shape != powers.shape:
        raise ValueError(f"{mode_name}: pulse_timescales and pulse_loads must have same shape")
    if times.size < 1:
        raise ValueError(f"{mode_name}: pulse_timescales and pulse_loads must have length >= 1")
    if np.any(times <= 0):
        raise ValueError(f"{mode_name}: all pulse_timescales must be > 0")

    return times, powers


def _validate_pipe_group(pg: PipeGroup) -> None:
    if pg.length <= 0:
        raise ValueError(f"PipeGroup ID={pg.id_}: length must be > 0")
    if pg.diameter_inner <= 0 or pg.diameter_outer <= 0:
        raise ValueError(f"PipeGroup ID={pg.id_}: diameter_inner and diameter_outer must be > 0")
    if pg.diameter_outer < pg.diameter_inner:
        raise ValueError(f"PipeGroup ID={pg.id_}: diameter_outer must be >= diameter_inner")
    if pg.burial_depth < 0:
        raise ValueError(f"PipeGroup ID={pg.id_}: burial_depth must be >= 0")
    if pg.n_parallel_pipes not in (1, 2):
        raise ValueError(
            f"PipeGroup ID={pg.id_}: only one- and two-pipe systems are supported "
            f"(n_parallel_pipes must be 1 or 2)"
        )
    if pg.n_traces <= 0:
        raise ValueError(f"PipeGroup ID={pg.id_}: n_traces must be > 0")
    if pg.n_parallel_pipes == 2 and pg.pipe_spacing is None:
        raise ValueError(
            f"PipeGroup ID={pg.id_}: pipe_spacing must be set when n_parallel_pipes == 2"
        )
    if pg.pipe_spacing is not None and pg.pipe_spacing <= 0:
        raise ValueError(f"PipeGroup ID={pg.id_}: pipe_spacing must be > 0 when provided")


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
    return float(brine.dynamic_viscosity * brine.specific_heat / brine.thermal_conductivity)


def _shallow_k(soil: Soil, mode: str) -> float:
    if mode == "heating":
        k = float(soil.thermal_conductivity_shallow_heating)
    elif mode == "cooling":
        k = float(soil.thermal_conductivity_shallow_cooling)
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
    rho = float(soil.density)
    cp = float(soil.specific_heat)

    if rho <= 0 or cp <= 0:
        raise ValueError(f"Soil density and specific_heat must be > 0. Got density={rho}, specific_heat={cp}.")

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
) -> ThermonetPerformance:
    pulse_timescales, pulse_loads = _validate_mode_input(mode_input, mode)

    Pr = _prandtl(model.brine)
    a_s = _soil_diffusivity_shallow(model.soil, mode)
    k_s = _shallow_k(model.soil, mode)

    dP = _delta_p(pulse_loads)

    Ti_C = float(mode_input.temperature_heat_pump_inlet)
    To_C = float(mode_input.temperature_heat_pump_outlet)
    Tm_C = 0.5 * (Ti_C + To_C)

    n_groups = len(model.pipe_groups)
    n_times = pulse_timescales.size

    load_supply_fractions_per_group = np.zeros(n_groups, dtype=float)
    T_volume_weighted = np.zeros((n_groups, n_times), dtype=float)

    total_volume_m3 = 0.0

    for i, pg in enumerate(model.pipe_groups):
        TP_C = _temp_penalty_at_depth(
            surface_amp_C=model.temperature_surface_amplitude,
            depth_m=pg.burial_depth,
            diffusivity_m2_s=a_s,
        )

        R_pipe = pipe_thermal_resistance(
            Di=float(pg.diameter_inner),
            Do=float(pg.diameter_outer),
            Re=float(pg.reynolds_number),
            Pr=float(Pr),
            k_fluid=float(model.brine.thermal_conductivity),
            k_pipe=float(pg.thermal_conductivity_pipe),
        )

        K1 = _bundle_k1(
            diffusivity_m2_s=a_s,
            times_s=pulse_timescales,
            depth_m=pg.burial_depth,
            n_parallel_pipes=pg.n_parallel_pipes,
            spacing_m=pg.pipe_spacing,
        )

        G_grid = csm(
            float(pg.diameter_outer) / 2.0,
            float(pg.diameter_outer) / 2.0,
            pulse_timescales,
            a_s,
        ) + K1

        kernel = ((G_grid / k_s) + R_pipe) / float(pg.length) / float(pg.n_traces)
        denom = float(np.dot(dP, kernel))

        numerator, tp_sign = _mode_numerator(
            T0_C=float(model.temperature_ground_undisturbed),
            Tm_C=Tm_C,
            TP_C=TP_C,
            mode=mode,
        )

        F_i = float(numerator / denom)
        load_supply_fractions_per_group[i] = F_i

        T_seq_C = float(model.temperature_ground_undisturbed) + tp_sign * TP_C - F_i * np.cumsum(dP * kernel)

        volume_m3 = float(pg.n_traces) * float(pg.length) * math.pi * float(pg.diameter_inner) ** 2 / 4.0
        total_volume_m3 += volume_m3
        T_volume_weighted[i, :] = T_seq_C * volume_m3

    temperatures_mean = np.sum(T_volume_weighted, axis=0) / float(total_volume_m3)
    load_supply_fraction = float(np.sum(load_supply_fractions_per_group))

    return ThermonetPerformance(
        temperatures_mean=temperatures_mean,
        load_supply_fraction=load_supply_fraction,
        load_supply_fractions_per_group=load_supply_fractions_per_group,
    )


def compute_distribution_pipe_thermal_response(
    model: DistributionPipeModel,
) -> dict[str, ThermonetPerformance]:
    """
    Hoved-entry point.

    Returnerer:
        {
            "heating": ThermonetPerformance(...),
            "cooling": ThermonetPerformance(...),   # kun hvis cooling er angivet
        }
    """
    _validate_model(model)

    results: dict[str, ThermonetPerformance] = {
        "heating": _compute_mode(model, model.heating, mode="heating")
    }

    if model.cooling is not None:
        results["cooling"] = _compute_mode(model, model.cooling, mode="cooling")

    return results
