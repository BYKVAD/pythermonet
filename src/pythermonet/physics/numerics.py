from __future__ import annotations

def halley_iteration(x: float, dx: float, f1: float, f2: float, f3: float) -> float:
    """
    One iteration of Halley's method using numerical derivatives based on f(x-dx), f(x), f(x+dx).
    """
    df = (f3 - f1) / (2 * dx)
    ddf = (f3 - 2 * f2 + f1) / (dx**2)
    return x - 2 * f2 * df / (2 * df**2 - f2 * ddf)