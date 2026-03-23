"""
gfunction.py
------------
G-function assembly for a PipeInfrastructure (horizontal ground loop).

Equal load distribution
-----------------------
All NParallelPipes pipes carry the same heat extraction rate per unit length
q' (W/m). The g-function is therefore scalar:

    g(t) = 2*pi*k_s * DeltaT_avg / q'

where DeltaT_avg is the average pipe wall temperature rise across all pipes
and all trace segments:

    DeltaT_avg = (1 / (N_pipes * N_seg)) * sum_{recv} sum_{src} h_{src,recv}(t)
                 * q' / (2*pi*k_s)

So:
    g(t) = (1 / (N_pipes * N_seg)) * sum_{recv} sum_{src} h_{src,recv}(t)

Spatial aggregation
-------------------
Pairs of segments with the same (dy, dx) geometry produce the same h value
and are computed only once. Pairs outside the thermal propagation distance
are skipped entirely (their erfc contribution is < 0.002%).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .aggregation import build_interaction_map, _segment_x_start
from .heat_transfer import hfls_segment_interaction


class HHEGFunction:
    """
    G-function calculator for a horizontal ground loop defined by a
    PipeInfrastructure.

    Parameters
    ----------
    pipe_infrastructure : PipeInfrastructure
        The horizontal ground loop definition. pipeDistance must be set
        (not None) for multi-pipe fields.
    k_s : float
        Soil thermal conductivity (W/m/K).
    alpha : float
        Soil thermal diffusivity (m2/s).
    time : array-like, optional
        Time values (s). If provided, g-function is evaluated immediately.
    dis_tol : float
        Lateral/longitudinal distance tolerance for pair grouping (m).
    n_sigma : float
        Thermal propagation distance multiplier for pair cutoff.
    n_gauss : int
        Gauss-Legendre quadrature points per integral.
    disp : bool
        Print progress messages.
    """

    def __init__(
        self,
        pipe_infrastructure,
        k_s: float,
        alpha: float,
        time: Optional[np.ndarray] = None,
        dis_tol: float = 0.01,
        n_sigma: float = 3.0,
        n_gauss: int = 21,
        disp: bool = False,
    ):
        self.pi = pipe_infrastructure
        self.k_s = k_s
        self.alpha = alpha
        self.dis_tol = dis_tol
        self.n_sigma = n_sigma
        self.n_gauss = n_gauss
        self.disp = disp
        self.gFunc: Optional[np.ndarray] = None
        self._time: Optional[np.ndarray] = None

        self._n_pipes = pipe_infrastructure.NParallelPipes
        self._n_seg = len(pipe_infrastructure.traceSegments)
        self._n_total = self._n_pipes * self._n_seg
        self._depth = pipe_infrastructure.burialDepth
        self._pipe_dist = pipe_infrastructure.pipeDistance or 0.0
        self._trace = pipe_infrastructure.traceSegments
        self._r_pipe = (self._trace[0].outerDiameter / 2.0) if self._trace[0].outerDiameter is not None else 0.0

        # Pre-compute x start positions along the trace
        self._x_starts = [
            _segment_x_start(s, self._trace) for s in range(self._n_seg)
        ]

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
        """Evaluate g at a single time step."""

        # Build map of unique geometries -> list of (src, recv) pairs
        interaction_map = build_interaction_map(
            n_parallel=self._n_pipes,
            pipe_distance=self._pipe_dist,
            trace_segments=self._trace,
            t=t,
            alpha=self.alpha,
            dis_tol=self.dis_tol,
            n_sigma=self.n_sigma,
        )

        # h_sum accumulates sum of h values for each receiver segment
        # indexed as h_sum[p_recv, s_recv]
        h_sum = np.zeros((self._n_pipes, self._n_seg))

        for geom, pairs in interaction_map.items():
            # Compute h for this unique geometry (using first pair as representative)
            src_id, recv_id = pairs[0]
            p_src, s_src = src_id
            p_recv, s_recv = recv_id

            h_val = hfls_segment_interaction(
                t=t,
                x_src=self._x_starts[s_src],
                L_src=self._trace[s_src].length,
                y_src=p_src * self._pipe_dist,
                x_recv=self._x_starts[s_recv],
                L_recv=self._trace[s_recv].length,
                y_recv=p_recv * self._pipe_dist,
                depth=self._depth,
                r_pipe=self._r_pipe,
                alpha=self.alpha,
                n_gauss=self.n_gauss,
            )

            # Distribute to all pairs that share this geometry
            for src_id, recv_id in pairs:
                h_sum[recv_id.p, recv_id.s] += h_val

        # g = (1 / N_total) * sum of all h values
        return float(np.sum(h_sum)) / self._n_total

    def visualize_g_function(self, ax=None):
        """Plot the g-function. Returns matplotlib figure."""
        import matplotlib.pyplot as plt

        if self.gFunc is None:
            raise RuntimeError("Call evaluate() first.")

        fig, axis = (None, ax) if ax is not None else plt.subplots()
        if fig is None:
            fig = axis.get_figure()

        # Normalise time axis to t_s = D^2 / (9*alpha) (Eskilson time scale)
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
    dis_tol: float = 0.01,
    n_sigma: float = 3.0,
    n_gauss: int = 21,
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
        Soil thermal diffusivity (m2/s).
    time : np.ndarray
        Evaluation times (s).
    dis_tol : float
        Pair grouping tolerance (m).
    n_sigma : float
        Thermal propagation cutoff multiplier.
    n_gauss : int
        Gauss-Legendre quadrature points.
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
        dis_tol=dis_tol,
        n_sigma=n_sigma,
        n_gauss=n_gauss,
        disp=disp,
    ).gFunc
