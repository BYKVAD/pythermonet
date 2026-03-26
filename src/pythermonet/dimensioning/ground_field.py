from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.components.vhe_field import VHEField
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.core.soil import Soil
from pythermonet.physics.bhe_resistance import compute_rb_for_vhe_field
from pythermonet.physics.hydraulics import darcy_friction_factor_beier, reynolds_number
from pythermonet.physics.sources import ils
import pythermonet.simulation.HHE as hhe


class GroundField(ABC):
    """
    Abstract interface for a ground heat exchanger field.

    Both BHE (borehole) and HHE (horizontal pipe) fields implement this
    interface so the same bisection sizing solver can be used for either.
    """

    @property
    @abstractmethod
    def n_elements(self) -> int:
        """Number of parallel elements (boreholes or pipes) — used for g-function geometry."""

    @property
    def n_thermal_elements(self) -> int:
        """
        Number of independent heat exchange paths — used as the divisor in
        q = P / (n_thermal_elements * L).

        For BHE: equals n_elements (one flow path per borehole).
        For HHE loops: equals n_elements // 2 (two physical pipes per loop share one flow path).
        Default implementation returns n_elements; override for loop topologies.
        """
        return self.n_elements

    @abstractmethod
    def T_undisturbed(self, L: float, soil: Soil) -> float:
        """
        Mean undisturbed ground temperature at element length L [°C].

        For BHE this increases with depth (geothermal gradient).
        For HHE it is constant (burial depth is fixed).
        """

    @abstractmethod
    def T_gradient(self, soil: Soil) -> float:
        """
        Rate of change of T_undisturbed with respect to L [K/m].

        BHE: Qgeo / (2 * k_s).
        HHE: 0.0.
        """

    @abstractmethod
    def compute_gfunction(self, L: float, times_s: np.ndarray, alpha: float) -> np.ndarray:
        """G-function for the field with element length L at given times [s]."""

    @abstractmethod
    def R_at_L(self, L: float, m_dot_per_element: float, brine: HeatCarrier, soil: Soil) -> float:
        """
        Effective thermal resistance per unit length [K·m/W] at element length L.

        BHE: Rb with flow/length correction.
        HHE: pipe wall + convective resistance (no correction needed).
        """

    @abstractmethod
    def R_simple(self, m_dot_per_element: float, brine: HeatCarrier, soil: Soil) -> float:
        """
        Thermal resistance for the ILS initial guess (no length correction).

        BHE: Rb without flow/length correction.
        HHE: same as R_at_L (no correction exists).
        """

    @abstractmethod
    def ils_gfunction(self, times_s: np.ndarray, alpha: float) -> np.ndarray:
        """
        ILS-based g-function for the bisection initial guess.

        Does not require a pygfunction/HHEGFunction call — used only to
        estimate the starting bracket cheaply.
        """

    @abstractmethod
    def seasonal_amplitude(self, soil: Soil) -> float:
        """
        Conservative seasonal temperature amplitude at the element depth [K].

        Used to shift the effective undisturbed temperature for worst-case
        sizing: subtract for heating (cold winter), add for cooling (hot summer).

        BHE: 0.0 — the seasonal signal is fully attenuated at borehole depth.
        HHE: A(D) = surfaceTempAmp · exp(−D/δ),  δ = √(2α/ω).
        """

    @abstractmethod
    def k_s_eff(self, soil: Soil) -> float:
        """
        Effective soil thermal conductivity [W/m/K] consistent with the
        g-function used by this field.

        BHE: deep conductivity (soil.thermalCond) — same value used by pygfunction.
        HHE: shallow conductivity supplied at construction — same value used by HHEGFunction.
        Must match the k_s baked into compute_gfunction / ils_gfunction so that
        ΔT = q · g / (2π · k_s) is dimensionally consistent.
        """



# ---------------------------------------------------------------------------
# BHE implementation
# ---------------------------------------------------------------------------

class BHEGroundField(GroundField):
    """Borehole heat exchanger field adapter wrapping VHEField."""

    def __init__(self, vhe_field: VHEField) -> None:
        self._field = vhe_field

    @property
    def n_elements(self) -> int:
        return self._field.n_boreholes

    def T_undisturbed(self, L: float, soil: Soil) -> float:
        return float(soil.surfaceTemp) + float(soil.Qgeo) * L / (2.0 * float(soil.thermalCond))

    def T_gradient(self, soil: Soil) -> float:
        return float(soil.Qgeo) / (2.0 * float(soil.thermalCond))

    def compute_gfunction(self, L: float, times_s: np.ndarray, alpha: float) -> np.ndarray:
        return replace(self._field, H_m=L).compute_pygfunctions(
            times_s=times_s, alpha_m2_s=alpha
        )

    def R_at_L(self, L: float, m_dot_per_element: float, brine: HeatCarrier, soil: Soil) -> float:
        return compute_rb_for_vhe_field(
            vhe_field=self._field, brine=brine, soil=soil,
            L_bhe_m=L, m_dot_kg_s=m_dot_per_element,
            use_flow_length_correction=True,
        ).Rb_K_m_W

    def R_simple(self, m_dot_per_element: float, brine: HeatCarrier, soil: Soil) -> float:
        return compute_rb_for_vhe_field(
            vhe_field=self._field, brine=brine, soil=soil,
            L_bhe_m=1.0, m_dot_kg_s=m_dot_per_element,
            use_flow_length_correction=False,
        ).Rb_K_m_W

    def seasonal_amplitude(self, soil: Soil) -> float:
        return 0.0

    def k_s_eff(self, soil: Soil) -> float:
        return float(soil.thermalCond)

    def ils_gfunction(self, times_s: np.ndarray, alpha: float) -> np.ndarray:
        r_b = float(self._field.r_b_m)
        xy = np.array(self._field.xy_m, dtype=float)
        N = self._field.n_boreholes
        TWO_PI = 2.0 * math.pi

        g = np.zeros(len(times_s))
        for k, t in enumerate(times_s):
            g_self = TWO_PI * ils(alpha, float(t), r_b)
            g_cross = sum(
                TWO_PI * ils(alpha, float(t), math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
                for i in range(N) for j in range(N) if i != j
            )
            g[k] = g_self + g_cross / N

        return g


# ---------------------------------------------------------------------------
# HHE implementation
# ---------------------------------------------------------------------------

class HHEGroundField(GroundField):
    """
    Horizontal heat exchanger field adapter wrapping PipeInfrastructure.

    Currently supports single-segment traces (straight parallel pipes).
    The segment length is the sizing variable (L).

    Parameters
    ----------
    pipe_infrastructure : PipeInfrastructure
        Template infrastructure — segment length is overridden at each
        bisection evaluation; all other properties are preserved.
    k_s : float
        Soil thermal conductivity [W/m/K] (required for HHE g-function).
    """

    def __init__(self, pipe_infrastructure: PipeInfrastructure, k_s: float) -> None:
        if len(pipe_infrastructure.traceSegments) != 1:
            raise ValueError(
                "HHEGroundField currently only supports single-segment traces "
                "(straight parallel pipes). Got "
                f"{len(pipe_infrastructure.traceSegments)} segments."
            )
        self._pi = pipe_infrastructure
        self._k_s = float(k_s)
        self._seg = pipe_infrastructure.traceSegments[0]

    @property
    def n_elements(self) -> int:
        return self._pi.NParallelPipes

    def T_undisturbed(self, L: float, soil: Soil) -> float:
        # Horizontal pipe: undisturbed temperature at burial depth, no L dependence
        D = float(self._pi.burialDepth)
        return float(soil.surfaceTemp) + float(soil.Qgeo) * D / (2.0 * float(soil.thermalCond))

    def T_gradient(self, soil: Soil) -> float:
        return 0.0

    def compute_gfunction(self, L: float, times_s: np.ndarray, alpha: float) -> np.ndarray:
        new_seg = PipeSegment(
            outerDiameter=self._seg.outerDiameter,
            SDR=self._seg.SDR,
            material=self._seg.material,
            roughnessHeight=self._seg.roughnessHeight,
            ID=self._seg.ID,
            length=L,
        )
        new_pi = PipeInfrastructure(
            NParallelPipes=self._pi.NParallelPipes,
            traceSegments=[new_seg],
            pipeDistance=self._pi.pipeDistance,
            burialDepth=self._pi.burialDepth,
        )
        return hhe.gfunction(new_pi, k_s=self._k_s, alpha=alpha, time=np.asarray(times_s))

    def R_at_L(self, L: float, m_dot_per_element: float, brine: HeatCarrier, soil: Soil) -> float:
        return self._r_pipe(m_dot_per_element, brine)

    def R_simple(self, m_dot_per_element: float, brine: HeatCarrier, soil: Soil) -> float:
        return self._r_pipe(m_dot_per_element, brine)

    def _r_pipe(self, m_dot: float, brine: HeatCarrier) -> float:
        """Pipe thermal resistance per unit length [K·m/W]: wall + convective."""
        Do = float(self._seg.outerDiameter)
        Di = Do * (1.0 - 2.0 / float(self._seg.SDR))
        k_pipe = float(self._seg.material.thermalCond)

        v = m_dot / (brine.rho * math.pi * Di ** 2 / 4.0)
        Re = float(reynolds_number(brine.rho, brine.dynamicViscosity, v, Di))

        if Re < 2300.0:
            Nu = 4.36  # laminar, constant heat flux
        else:
            Pr = brine.c * brine.dynamicViscosity / brine.thermalCond
            f = float(darcy_friction_factor_beier(Re))
            Nu = (f / 8.0) * (Re - 1000.0) * Pr / (
                1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0)
            )

        h = Nu * brine.thermalCond / Di
        R_conv = 1.0 / (h * math.pi * Di)
        R_wall = math.log(Do / Di) / (2.0 * math.pi * k_pipe)
        return R_conv + R_wall

    def seasonal_amplitude(self, soil: Soil) -> float:
        alpha = self._k_s / (float(soil.rho) * float(soil.c))
        omega = 2.0 * math.pi / (365.25 * 24.0 * 3600.0)
        delta = math.sqrt(2.0 * alpha / omega)
        D = float(self._pi.burialDepth)
        return float(soil.surfaceTempAmp) * math.exp(-D / delta)

    def k_s_eff(self, soil: Soil) -> float:
        return self._k_s

    def ils_gfunction(self, times_s: np.ndarray, alpha: float) -> np.ndarray:
        """ILS g averaged over N parallel horizontal pipes at lateral spacing d."""
        r_pipe = float(self._seg.outerDiameter) / 2.0
        N = self._pi.NParallelPipes
        d = float(self._pi.pipeDistance or 0.0)
        TWO_PI = 2.0 * math.pi

        g = np.zeros(len(times_s))
        for k, t in enumerate(times_s):
            g_self = TWO_PI * ils(alpha, float(t), r_pipe)
            g_cross = 0.0
            if d > 0.0 and N > 1:
                g_cross = sum(
                    TWO_PI * ils(alpha, float(t), abs(i - j) * d)
                    for i in range(N) for j in range(N) if i != j
                )
            g[k] = g_self + g_cross / N

        return g
