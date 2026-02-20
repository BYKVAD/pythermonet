from __future__ import annotations
import math
import numpy as np
import mpmath as mp

from .hydraulics import darcy_friction_factor_beier

def pipe_thermal_resistance(Di: float, Do: float, Re: float, Pr: float, k_fluid: float, k_pipe: float) -> float:
    """
    Combined convective + conductive resistance of a pipe [m*K/W].
    """
    if Re > 2300:
        fD = darcy_friction_factor_beier(Re)
        h = k_fluid * 0.125 * fD * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(0.125 * fD) * (Pr ** (2 / 3) - 1)) / Di
    else:
        # Laminar: average of Nu=3.66 and 4.36
        h = (3.657 + 4.364) / 2 * k_fluid / Di

    R_conv = 1 / (2 * math.pi * 0.5 * Di * h)
    R_cond = 1 / (2 * math.pi * k_pipe) * math.log(Do / Di)
    return R_conv + R_cond

def rb_multipole(k_fluid: float, k_pipe: float, k_grout: float, k_soil: float,
                 r_b: float, r_p: float, r_i: float, s: float,
                 Re_bhe: float, Pr: float) -> float:
    """
    Multipole borehole thermal resistance [K*m/W] (RbMP).
    Naming here follows your existing notation.
    """
    b = 2 * math.pi * k_grout * pipe_thermal_resistance(2*r_i, 2*r_p, Re_bhe, Pr, k_fluid, k_pipe)
    C1 = s / (2 * r_b)
    C2 = r_b / r_p
    C3 = 1 / (2 * C1 * C2)
    si = (k_grout - k_soil) / (k_grout + k_soil)

    num = (1 - (4 * si * C1**4) / (1 - C1**4))**2
    den = ((1 + b) / (1 - b) + C3**2 * (1 + 16 * si * C1**4 / (1 - C1**4)**2))
    return 1 / (4 * math.pi * k_grout) * (
        b
        + math.log(C2 / (2 * C1) / (1 - C1**4)**si)
        - C3**2 * num / den
    )

def rb_multipole_flow_length(k_fluid: float, k_pipe: float, k_grout: float, k_soil: float,
                            rho_b: float, c_b: float,
                            r_b: float, r_p: float, r_i: float,
                            L_bhe: float, s: float, Q_bhe: float,
                            Re_bhe: float, Pr: float) -> float:
    """
    RbMP with flow and length corrections (RbMPflc). Returns [K*m/W].
    """
    b = 2 * math.pi * k_grout * pipe_thermal_resistance(2*r_i, 2*r_p, Re_bhe, Pr, k_fluid, k_pipe)
    C1 = s / (2 * r_b)
    C2 = r_b / r_p
    C3 = 1 / (2 * C1 * C2)
    si = (k_grout - k_soil) / (k_grout + k_soil)

    Rb1 = rb_multipole(k_fluid, k_pipe, k_grout, k_soil, r_b, r_p, r_i, s, Re_bhe, Pr)

    Ra = 1 / (math.pi * k_grout) * (
        b
        + math.log((1 + C1**2)**si / (C3 * (1 - C1**2)**si))
        - C3**2 * (1 - C1**4 + 4 * si * C1**2)**2 /
          (((1 + b) / (1 - b) * (1 - C1**4)**2 - C3**2 * (1 - C1**4)**2 + 8 * si * C1**2 * C3**2 * (1 + C1**4)))
    )

    dRb1 = (1 / 3) / Ra * (L_bhe / (rho_b * c_b * Q_bhe))**2
    R1b = 2 * Rb1
    R12 = 2 * Ra * R1b / (2 * R1b - Ra)
    nu = (L_bhe / (rho_b * c_b * Q_bhe)) * (1 / Rb1) * math.sqrt(1 + 4 * Rb1 / R12)

    Rb2 = Rb1 * nu * mp.coth(nu)
    Rb1_corr = Rb1 + dRb1
    return float(0.5 * (Rb1_corr + Rb2))