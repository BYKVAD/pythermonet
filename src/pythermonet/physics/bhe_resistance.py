from __future__ import annotations
from dataclasses import dataclass
import math

from pythermonet.components.vhe_field import VHEField
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil

from pythermonet.physics.thermal_resistances import rb_multipole, rb_multipole_flow_length

@dataclass(frozen=True)
class BHEResistanceResult:
    """
    Output of a single borehole thermal resistance calculation.

    ''thermal_resistance_borehole'' is the primary result; the remaining
    fields are the intermediate flow and geometry values used to compute it,
    stored for inspection and diagnostics.

    Attributes
    ----------
    thermal_resistance_borehole : float
        Effective borehole thermal resistance Rb [K·m/W].
    reynolds_number : float
        Reynolds number of the brine in the U-pipe [-].
    prandtl_number : float
        Prandtl number of the brine [-].
    volume_flow : float
        Volumetric flow rate per borehole [m³/s].
    velocity_brine_mean : float
        Mean cross-sectional brine velocity in the U-pipe [m/s].
    radius_borehole : float
        Borehole radius [m].
    radius_pipe_outer : float
        U-pipe outer radius [m].
    radius_pipe_inner : float
        U-pipe inner radius [m].
    shank_spacing : float
        Center-to-center distance between the two U-pipe legs [m].
    used_flow_length_correction : bool
        Whether the flow-length correction was applied when computing Rb.
    """

    thermal_resistance_borehole: float  # [K·m/W]
    reynolds_number: float
    prandtl_number: float
    volume_flow: float                  # [m³/s]
    velocity_brine_mean: float          # [m/s]
    radius_borehole: float              # [m]
    radius_pipe_outer: float            # [m]
    radius_pipe_inner: float            # [m]
    shank_spacing: float                # [m]
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
    od_p = float(vhe_field.pipe.diameter_outer)
    sdr = float(vhe_field.pipe.sdr)
    Di = _inner_diameter_from_od_sdr(od_p, sdr)

    r_p = 0.5 * od_p
    r_i = 0.5 * Di
    r_b = 0.5 * float(vhe_field.borehole.diameter_outer)
    s = float(vhe_field.shank_spacing)

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
        thermal_resistance_borehole=float(Rb),
        reynolds_number=float(Re),
        prandtl_number=float(Pr),
        volume_flow=float(Q),
        velocity_brine_mean=float(v),
        radius_borehole=float(r_b),
        radius_pipe_outer=float(r_p),
        radius_pipe_inner=float(r_i),
        shank_spacing=float(s),
        used_flow_length_correction=bool(use_flow_length_correction),
    )
