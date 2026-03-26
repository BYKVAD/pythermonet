"""
gfunction.py
------------
G-function assembly for a PipeInfrastructure (horizontal ground loop).

Equal load distribution
-----------------------
All NParallelPipes pipes carry the same heat extraction rate per unit length
q' (W/m). The g-function is therefore scalar:

    g(t) = 2*pi*k_s * DeltaT_avg / q'

where DeltaT_avg is the average pipe wall temperature rise across all pipes.

Each pipe-pair interaction is expressed as:

    DeltaT_pair = q' / (2*pi*k_s) * h

so:

    g(t) = (1 / N_total) * sum_{recv} sum_{src} h_{src,recv}(t)

Single-integral FLS
-------------------
For single-segment traces (all pipes same length L, starting at x = 0),
each h_{src,recv} is evaluated with the single-integral HFLS formula in
heat_transfer.py, keyed only by the lateral separation dy between the pipes.

Spatial aggregation
-------------------
Pairs of pipes with the same |dy| produce the same h value and are computed
only once. Pairs outside the thermal propagation distance are skipped
(their erfc contribution is negligible).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .aggregation import build_interaction_map
from .heat_transfer import hfls_pipe_interaction


class HHEGFunction:
    """
    G-function calculator for a horizontal ground loop defined by a
    PipeInfrastructure.

    Parameters
    ----------
    pipe_infrastructure : PipeInfrastructure
        The horizontal ground loop definition.
    k_s : float
        Soil thermal conductivity (W/m/K).
    alpha : float
        Soil thermal diffusivity (m²/s).
    time : array-like, optional
        Time values (s). If provided, g-function is evaluated immediately.
    n_sigma : float
        Thermal propagation distance multiplier for pair cutoff.
    disp : bool
        Print progress messages.
    """

    def __init__(
        self,
        pipe_infrastructure,
        k_s: float,
        alpha: float,
        time: Optional[np.ndarray] = None,
        n_sigma: float = 3.0,
        disp: bool = False,
    ):
        self.pi = pipe_infrastructure
        self.k_s = k_s
        self.alpha = alpha
        self.n_sigma = n_sigma
        self.disp = disp
        self.gFunc: Optional[np.ndarray] = None
        self._time: Optional[np.ndarray] = None

        self._n_pipes = pipe_infrastructure.NParallelPipes
        self._n_seg = len(pipe_infrastructure.traceSegments)
        self._n_total = self._n_pipes * self._n_seg
        self._depth = float(pipe_infrastructure.burialDepth)
        self._pipe_dist = float(pipe_infrastructure.pipeDistance or 0.0)
        self._trace = pipe_infrastructure.traceSegments
        self._r_pipe = float(self._trace[0].outerDiameter / 2.0) if self._trace[0].outerDiameter is not None else 0.0

        if time is not None:
            self.evaluate(np.asarray(time, dtype=float))

    def evaluate(self, time: np.ndarray) -> np.ndarray:
        """
        Evaluate the g-function at each value in time (s).

        Returns
        -------
        np.ndarray, shape (K,)
        """
        self._time = time
        g = np.zeros(len(time))

        for k, t in enumerate(time):
            if self.disp:
                print(f"  [{k+1}/{len(time)}]  t = {t:.3e} s", flush=True)
            g[k] = self._eval_single(t)

        self.gFunc = g
        return g

    def _eval_single(self, t: float) -> float:
        """
        Evaluate g at a single time step using field symmetry.

        The field is symmetric about its midpoint: pipe i and pipe N-1-i
        have identical temperatures.  Only the first n_half receivers are
        computed; their contributions are doubled, except for the centre
        pipe (when N is odd) which is unique and counted once.
        """
        n_pipes = self._n_pipes
        n_half = (n_pipes + 1) // 2  # number of unique receiver pipes

        interaction_map = build_interaction_map(
            n_parallel=n_pipes,
            pipe_distance=self._pipe_dist,
            trace_segments=self._trace,
            t=t,
            alpha=self.alpha,
            n_sigma=self.n_sigma,
            n_recv=n_half,
        )

        h_sum = np.zeros((n_half, self._n_seg))

        for geom, pairs in interaction_map.items():
            L = float(self._trace[geom.s_src].length)
            dy = abs(geom.dy)

            h_val = hfls_pipe_interaction(
                t=t,
                L=L,
                dy=dy,
                depth=self._depth,
                r_pipe=self._r_pipe,
                alpha=self.alpha,
            )

            for _, recv_id in pairs:
                h_sum[recv_id.p, recv_id.s] += h_val

        # Each of the first n_half pipes represents itself and its mirror
        # on the other side of the field — weight 2 for all, except the
        # centre pipe (last entry when N is odd) which is unique — weight 1.
        weights = np.full(n_half, 2.0)
        if n_pipes % 2 == 1:
            weights[-1] = 1.0

        return float(np.dot(weights, h_sum.sum(axis=1))) / self._n_total

    def visualize_g_function(self, ax=None):
        """Plot the g-function. Returns matplotlib figure."""
        import matplotlib.pyplot as plt

        if self.gFunc is None:
            raise RuntimeError("Call evaluate() first.")

        fig, axis = (None, ax) if ax is not None else plt.subplots()
        if fig is None:
            fig = axis.get_figure()

        t_s = self._depth**2 / (9.0 * self.alpha)
        ln_t = np.log(self._time / t_s)

        axis.plot(ln_t, self.gFunc, lw=1.5)
        axis.set_xlabel(r'$\ln(t\,/\,t_s)$')
        axis.set_ylabel(r'$g$')
        axis.set_title(
            f'HHE g-function  —  {self._n_pipes} pipes × '
            f'{self._n_seg} segments, D = {self._depth} m'
        )
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def gfunction(
    pipe_infrastructure,
    k_s: float,
    alpha: float,
    time: np.ndarray,
    n_sigma: float = 3.0,
    disp: bool = False,
) -> np.ndarray:
    """
    Compute the g-function for a horizontal ground loop.

    Parameters
    ----------
    pipe_infrastructure : PipeInfrastructure
        Ground loop definition.
    k_s : float
        Soil thermal conductivity (W/m/K).
    alpha : float
        Soil thermal diffusivity (m²/s).
    time : np.ndarray
        Evaluation times (s).
    n_sigma : float
        Thermal propagation cutoff multiplier.
    disp : bool
        Print progress.

    Returns
    -------
    np.ndarray
        g-function values at each time.
    """
    return HHEGFunction(
        pipe_infrastructure, k_s, alpha,
        time=time,
        n_sigma=n_sigma,
        disp=disp,
    ).gFunc
