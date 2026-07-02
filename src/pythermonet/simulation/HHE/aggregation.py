"""
aggregation.py
--------------
Spatial aggregation for fields of parallel horizontal pipe segments.

A PipeInfrastructure defines:
  - n_parallel_pipes parallel pipes (indexed p = 0 .. n_parallel_pipes-1)
  - Each pipe follows the same trace of NSegments segments
  - Pipe p is offset laterally by p * pipe_distance from pipe 0
  - All pipes at the same burial_depth

The full field therefore has N = n_parallel_pipes * NSegments source/receiver
elements. The interaction between segment (p_src, s_src) and segment
(p_recv, s_recv) depends only on:

    dy   = (p_recv - p_src) * pipe_distance     lateral separation
    dx   = x_recv_start - x_src_start          longitudinal start offset

For a regular parallel field, (dy, dx) takes at most
n_parallel_pipes * NSegments unique values (typically far fewer after cutoff).
Each unique (dy, dx) pair is computed once and reused for all matching pairs.

Cutoff: pairs where the minimum distance between the two segments exceeds
thermal_propagation_distance(t, alpha, n_sigma) are skipped (erfc ~ 0).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, NamedTuple

from .heat_transfer import thermal_propagation_distance


class SegmentID(NamedTuple):
    """Identifies one segment: pipe index p, segment index s."""
    p: int   # pipe index  (0 .. n_parallel_pipes-1)
    s: int   # segment index (0 .. NSegments-1)


@dataclass(frozen=True)
class PairGeometry:
    """
    Hashable key for a unique source-receiver geometry.
    dy and dx are rounded to dis_tol buckets.
    """
    dy: float    # lateral separation (m)
    dx: float    # longitudinal start-position offset (m)
    # src and recv segment indices (same for all pairs sharing this geometry
    # in a uniform field, but stored so we can look up lengths etc.)
    s_src: int
    s_recv: int


def _round(value: float, tol: float) -> float:
    if tol <= 0:
        return value
    return round(value / tol) * tol


def _segment_x_start(seg_index: int, trace_segments) -> float:
    """x-position of the start of segment seg_index along the trace."""
    return sum(seg.length for seg in trace_segments[:seg_index])


def _min_distance_segments(
    x1: float, L1: float, y1: float,
    x2: float, L2: float, y2: float,
) -> float:
    """
    Minimum distance between two parallel horizontal line segments at the
    same depth. Only lateral and longitudinal separation matter.
    """
    dy = abs(y2 - y1)
    # Longitudinal gap: 0 if segments overlap in x, otherwise end-to-end gap
    x1_end = x1 + L1
    x2_end = x2 + L2
    long_gap = max(0.0, max(x1, x2) - min(x1_end, x2_end))
    return np.sqrt(dy**2 + long_gap**2)


def build_interaction_map(
    n_parallel: int,
    pipe_distance: float,
    trace_segments,          # list of PipeSegment
    t: float,
    alpha: float,
    dis_tol: float = 0.01,
    n_sigma: float = 3.0,
    n_recv: int | None = None,
) -> Dict[PairGeometry, List[Tuple[SegmentID, SegmentID]]]:
    """
    Build a map from unique pair geometries to lists of (src, recv) segment ID
    pairs that share that geometry.

    Parameters
    ----------
    n_parallel : int
        Number of parallel pipes (PipeInfrastructure.n_parallel_pipes).
    pipe_distance : float
        Lateral centre-to-centre pipe spacing (m).
    trace_segments : list of PipeSegment
        Segment definitions (PipeInfrastructure.trace_segments).
    t : float
        Current time (s) for cutoff computation.
    alpha : float
        Soil thermal diffusivity (m2/s).
    dis_tol : float
        Distance tolerance for grouping (m). Default 0.01.
    n_sigma : float
        Propagation distance multiplier. Default 3.
    n_recv : int, optional
        Number of receiver pipes to include (pipes 0 .. n_recv-1).
        Sources always span the full 0 .. n_parallel-1 range.
        Defaults to n_parallel (all receivers).

    Returns
    -------
    dict : PairGeometry -> list of (src SegmentID, recv SegmentID)
    """
    r_max = thermal_propagation_distance(t, alpha, n_sigma)
    n_seg = len(trace_segments)
    if n_recv is None:
        n_recv = n_parallel

    # Pre-compute x start positions for all segments
    x_starts = [_segment_x_start(s, trace_segments) for s in range(n_seg)]

    interaction_map: Dict[PairGeometry, List[Tuple[SegmentID, SegmentID]]] = {}

    for p_recv in range(n_recv):
        y_recv = p_recv * pipe_distance
        for s_recv in range(n_seg):
            x_recv = x_starts[s_recv]
            L_recv = trace_segments[s_recv].length

            for p_src in range(n_parallel):
                y_src = p_src * pipe_distance
                for s_src in range(n_seg):
                    x_src = x_starts[s_src]
                    L_src = trace_segments[s_src].length

                    # Cutoff check
                    min_dist = _min_distance_segments(
                        x_src, L_src, y_src,
                        x_recv, L_recv, y_recv,
                    )
                    # For self-pair, min_dist = 0 — always include
                    if p_src != p_recv or s_src != s_recv:
                        if min_dist > r_max:
                            continue

                    dy = _round(y_recv - y_src, dis_tol)
                    dx = _round(x_recv - x_src, dis_tol)

                    key = PairGeometry(dy=dy, dx=dx, s_src=s_src, s_recv=s_recv)

                    if key not in interaction_map:
                        interaction_map[key] = []
                    interaction_map[key].append(
                        (SegmentID(p_src, s_src), SegmentID(p_recv, s_recv))
                    )

    return interaction_map
