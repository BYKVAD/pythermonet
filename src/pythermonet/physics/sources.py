from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad
import scipy.special as sp

def ils(a: float, t: float, r: float) -> float:
    """Dimensionless infinite line source (ILS)."""
    return 1 / (4 * math.pi) * sp.exp1(r**2 / (4 * a * t))

def csm(r: float, r0: float, t, a: float):
    """Cylindrical Source Model (CSM). Supports scalar or array t."""
    p = r / r0
    t_arr = np.atleast_1d(t)
    z = a * t_arr / (r**2)

    def integrand(b, p_, z_):
        num = (np.exp(-b*b*z_) - 1) * (sp.jv(0, p_*b) * sp.yn(1, b) - sp.yn(0, p_*b) * sp.jv(1, b))
        den = (b*b) * (sp.jv(1, b)**2 + sp.yn(1, b)**2)
        return num / den

    G = np.array([quad(integrand, 0, np.inf, args=(p, zi))[0] for zi in z])
    out = G / (math.pi**2)

    if np.ndim(t) == 0:
        return float(out[0])
    return out

def vfls(r, H: float, a: float, U: float, t):
    """
    VFLS implementation. Returns scalar, vector, or matrix depending on inputs.
    (Keeps your scalar/array behavior.)
    """
    r = np.atleast_1d(r)
    t = np.atleast_1d(t)
    NT, NR = len(t), len(r)

    G = np.zeros((NR, NT))
    UU = U * U
    aa = a * a

    def erfi(x):
        return x * sp.erf(x) - (1 - np.exp(-x**2)) / np.sqrt(np.pi)

    def fun(s, rr_val):
        return np.exp(-UU / (16 * aa * s * s) - rr_val * s * s) * 2 * erfi(H * s) / (H * s * s)

    import scipy.integrate as integrate
    for j, r_val in enumerate(r):
        rr_val = r_val**2
        for i, t_val in enumerate(t):
            G[j, i], _ = integrate.quad(fun, 1 / np.sqrt(4 * a * t_val), np.inf, args=(rr_val,))

    G = sp.iv(0, 0 * U / (2 * a)) * G
    G /= (4 * math.pi)

    if G.shape == (1, 1):
        return float(G.item())
    if G.shape[0] == 1:
        return G.flatten()
    if G.shape[1] == 1:
        return G[:, 0]
    return G