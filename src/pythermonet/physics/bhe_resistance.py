from __future__ import annotations
from dataclasses import dataclass
import math

from pythermonet.components.vhe_field import VHEField
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil

from pythermonet.physics.thermal_resistances import rb_multipole, rb_multipole_flow_length

@dataclass(frozen=True)
class BHEResistanceResult:
    Rb_K_m_W: float
    Re: float
    Pr: float
    Q_m3_s: float
    v_m_s: float
    r_b_m: float
    r_p_m: float
    r_i_m: float
    s_m: float
    used_flow_length_correction: bool

def _inner_diameter_from_od_sdr(od_m: float, sdr: float) -> float:
    t = od_m / sdr
    return od_m - 2.0 * t

def _prandtl(mu: float, cp: float, k: float) -> float:
    return mu * cp / k

def _reynolds(rho: float, mu: float, v: float, Di: float) -> float:
    return rho * v * Di / mu

def compute_rb_for_vhe_field(
    *,
    vhe_field: VHEField,
    brine: HeatCarrier,
    soil: Soil,
    L_bhe_m: float,
    m_dot_kg_s: float,
    use_flow_length_correction: bool = True,
) -> BHEResistanceResult:
    # --- Geometri ---
    od_p = float(vhe_field.pipe.outer_diameter)
    sdr = float(vhe_field.pipe.sdr)
    Di = _inner_diameter_from_od_sdr(od_p, sdr)

    r_p = 0.5 * od_p
    r_i = 0.5 * Di
    r_b = 0.5 * float(vhe_field.borehole.outer_diameter)
    s = float(vhe_field.shankSpacing)

    # --- Materialer ---
    k_fluid = float(brine.thermal_conductivity)
    k_pipe = float(vhe_field.pipe.material.thermal_conductivity)
    k_grout = float(vhe_field.grout.thermal_conductivity)
    k_soil = float(soil.thermal_conductivity)

    # --- Flow og dimensionløse tal ---
    rho = float(brine.density)
    mu = float(brine.dynamic_viscosity)
    cp = float(brine.specific_heat)

    Q = m_dot_kg_s / rho
    A = math.pi * (Di**2) / 4.0
    v = Q / A

    Pr = _prandtl(mu, cp, k_fluid)
    Re = _reynolds(rho, mu, v, Di)

    if use_flow_length_correction:
        Rb = rb_multipole_flow_length(
            k_fluid=k_fluid, k_pipe=k_pipe, k_grout=k_grout, k_soil=k_soil,
            rho_b=rho, c_b=cp,
            r_b=r_b, r_p=r_p, r_i=r_i,
            L_bhe=L_bhe_m, s=s,
            Q_bhe=Q, Re_bhe=Re, Pr=Pr,
        )
    else:
        Rb = rb_multipole(
            k_fluid=k_fluid, k_pipe=k_pipe, k_grout=k_grout, k_soil=k_soil,
            r_b=r_b, r_p=r_p, r_i=r_i,
            s=s, Re_bhe=Re, Pr=Pr,
        )

    return BHEResistanceResult(
        Rb_K_m_W=float(Rb),
        Re=float(Re),
        Pr=float(Pr),
        Q_m3_s=float(Q),
        v_m_s=float(v),
        r_b_m=float(r_b),
        r_p_m=float(r_p),
        r_i_m=float(r_i),
        s_m=float(s),
        used_flow_length_correction=bool(use_flow_length_correction),
    )
