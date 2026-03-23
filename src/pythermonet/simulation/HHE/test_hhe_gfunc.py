"""
tests/test_hhe_gfunc.py
-----------------------
Unit and validation tests for hhe_gfunc.

Tests
-----
1. ILS convergence        — single long pipe self-response vs infinite line source
2. Image source effect    — g with image < g without image (surface cools ground)
3. Finite length effect   — short pipe g < long pipe g (end effects)
4. Monotonicity           — g strictly increasing in time for all field sizes
5. Multi-pipe interaction — N-pipe g >= 1-pipe g at all times
6. Segment additivity     — one 200m segment == two 100m segments in a trace
7. Spatial cutoff         — distant pipes are correctly excluded at early times

Run with:
    python -m pytest tests/test_hhe_gfunc.py -v          (with pytest)
    python tests/test_hhe_gfunc.py                        (standalone)

Plots are saved to tests/plots/ when run standalone or with --plots flag.
"""

import sys
import os
import argparse
import numpy as np
from scipy.special import exp1

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import hhe_gfunc as hhe
from hhe_gfunc._pythermonet_stubs import (
    Material, PipeSegment, PipeInfrastructure
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ALPHA  = 1.0e-6   # m2/s  soil thermal diffusivity
K_S    = 2.0      # W/m/K soil thermal conductivity
DEPTH  = 1.5      # m     burial depth
R_PIPE = 0.016    # m     pipe outer radius (32mm OD)
HDPE   = Material(rho=950, c=1900, thermalCond=0.4)


def make_segment(length: float, seg_id: int = 0) -> PipeSegment:
    return PipeSegment(
        outerDiameter=2 * R_PIPE,
        SDR=11,
        material=HDPE,
        roughnessHeight=1.5e-5,
        ID=seg_id,
        length=length,
    )


def make_field(
    n_pipes: int = 1,
    seg_length: float = 100.0,
    n_segments: int = 1,
    spacing: float = 1.0,
    depth: float = DEPTH,
) -> PipeInfrastructure:
    segs = [make_segment(seg_length, i) for i in range(n_segments)]
    return PipeInfrastructure(
        NParallelPipes=n_pipes,
        traceSegments=segs,
        pipeDistance=spacing,
        burialDepth=depth,
    )


def ils_gfunc(time: np.ndarray, r: float, alpha: float) -> np.ndarray:
    """
    Infinite line source g-function.
    g_ILS(t) = 0.5 * E1(r^2 / (4*alpha*t))
    Valid when image source is negligible: sqrt(4*alpha*t) << depth.
    """
    return 0.5 * exp1(r**2 / (4.0 * alpha * time))


def fls_gfunc_single(
    time: np.ndarray,
    r: float,
    L: float,
    alpha: float,
) -> np.ndarray:
    """
    Finite line source g-function for a single pipe, no image source
    (infinite medium). Evaluated by direct numerical integration for reference.

    DeltaT(r,t) = q'/(4*pi*k_s) * integral_{-L/2}^{L/2} erfc(R/sqrt(4at))/R dxi
    g = 0.5 * (1/L) * integral_{-L/2}^{L/2} integral_{-L/2}^{L/2} [...] dxi_r dxi_s

    For a long pipe (L >> sqrt(4at)) this approaches the ILS solution.
    Here we compute the simpler point-evaluation form at the pipe centre:
    g_approx = 0.5 * integral_{-L/2}^{L/2} erfc(sqrt(u^2+r^2)/sqrt(4at))/sqrt(u^2+r^2) du
    """
    from scipy.integrate import quad
    from scipy.special import erfc

    g = np.zeros(len(time))
    for k, t in enumerate(time):
        sqrt_4at = np.sqrt(4.0 * alpha * t)
        hw = max(6.0 * sqrt_4at, 10.0 * r)
        lo = max(-L / 2.0, -hw)
        hi = min(L / 2.0,   hw)

        def integrand(u):
            R = np.sqrt(u**2 + r**2)
            return erfc(R / sqrt_4at) / R

        val, _ = quad(integrand, lo, hi, limit=200, points=[0.0])
        g[k] = 0.5 * val

    return g


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _assert_close(a, b, rtol, label):
    err = np.abs(a - b) / np.abs(b)
    max_err = err.max()
    assert max_err < rtol, (
        f"{label}: max relative error {max_err:.4f} exceeds tolerance {rtol}"
    )
    return max_err


def _assert_monotone(g, label):
    diffs = np.diff(g)
    assert np.all(diffs > 0), (
        f"{label}: g-function not strictly increasing (min diff = {diffs.min():.3e})"
    )


# ---------------------------------------------------------------------------
# Test 1: ILS convergence (short-to-medium times, single long pipe)
# ---------------------------------------------------------------------------

def test_ils_convergence():
    """
    Self-response of a single long pipe must match the ILS solution
    at times where sqrt(4*alpha*t) << depth (image source negligible)
    and sqrt(4*alpha*t) << pipe_length (end effects negligible).
    Tolerance: 1% relative error.
    """
    # Only use times where thermal front << depth (image negligible)
    # and thermal front << L/2 (end effects negligible)
    L = 200.0
    time_all = np.geomspace(60, 3600 * 24 * 365, 30)
    mask = (
        (np.sqrt(4 * ALPHA * time_all) / DEPTH < 0.25) &
        (np.sqrt(4 * ALPHA * time_all) / (L / 2) < 0.1)
    )
    time = time_all[mask]
    assert len(time) >= 5, "Too few time points satisfy the ILS validity mask"

    g_ref  = ils_gfunc(time, R_PIPE, ALPHA)
    g_hfls = hhe.gfunction(
        make_field(1, seg_length=L),
        k_s=K_S, alpha=ALPHA, time=time,
    )

    max_err = _assert_close(g_hfls, g_ref, rtol=0.01, label="ILS convergence")
    print(f"  [PASS] ILS convergence: max error = {max_err*100:.3f}%")
    return time, g_ref, g_hfls


# ---------------------------------------------------------------------------
# Test 2: Image source effect
# ---------------------------------------------------------------------------

def test_image_source_effect():
    """
    At long times (thermal front reaches the surface), the image source
    reduces the temperature rise compared to the infinite medium (no image).
    g_with_image < g_no_image at late times.

    We approximate g_no_image using the ILS (which has no image source)
    at late times where end effects are also negligible.
    """
    # Use times where sqrt(4*alpha*t) is a significant fraction of depth
    L = 500.0
    time = np.geomspace(3600 * 24 * 30, 3600 * 24 * 365 * 10, 10)
    mask = np.sqrt(4 * ALPHA * time) / DEPTH > 1.0
    time = time[mask]

    g_hfls = hhe.gfunction(
        make_field(1, seg_length=L),
        k_s=K_S, alpha=ALPHA, time=time,
    )
    g_ils = ils_gfunc(time, R_PIPE, ALPHA)

    # Image source reduces g — HFLS should be below ILS at late times
    assert np.all(g_hfls < g_ils), (
        "Image source effect: g_hfls should be < g_ILS at late times"
    )
    print(f"  [PASS] Image source effect: g_HFLS < g_ILS at late times ✓")
    return time, g_ils, g_hfls


# ---------------------------------------------------------------------------
# Test 3: Finite length effect
# ---------------------------------------------------------------------------

def test_finite_length_effect():
    """
    A short pipe (end effects significant) has lower g than a long pipe
    at times where sqrt(4*alpha*t) is comparable to pipe length.
    """
    time = np.geomspace(3600, 3600 * 24 * 365, 20)
    g_short = hhe.gfunction(make_field(1, seg_length=10.0),
                             k_s=K_S, alpha=ALPHA, time=time)
    g_long  = hhe.gfunction(make_field(1, seg_length=200.0),
                             k_s=K_S, alpha=ALPHA, time=time)

    # At times where sqrt(4at) ~ L_short, end effects reduce g
    # — long pipe always has g >= short pipe
    assert np.all(g_long >= g_short - 1e-10), (
        "Finite length: long pipe g should be >= short pipe g"
    )
    print(f"  [PASS] Finite length: g_long >= g_short at all times ✓")
    return time, g_short, g_long


# ---------------------------------------------------------------------------
# Test 4: Monotonicity
# ---------------------------------------------------------------------------

def test_monotonicity():
    """g must be strictly increasing in time for all configurations."""
    time = np.geomspace(3600, 3600 * 24 * 365 * 25, 30)
    configs = {
        "1 pipe":         make_field(1),
        "3 pipes 1m":     make_field(3, spacing=1.0),
        "5 pipes 0.8m":   make_field(5, spacing=0.8),
        "2 segments":     make_field(1, n_segments=2, seg_length=50.0),
    }
    for label, field in configs.items():
        g = hhe.gfunction(field, k_s=K_S, alpha=ALPHA, time=time)
        _assert_monotone(g, label)
        print(f"  [PASS] Monotonicity: {label} ✓")
    return time, configs


# ---------------------------------------------------------------------------
# Test 5: Multi-pipe interaction
# ---------------------------------------------------------------------------

def test_multi_pipe_interaction():
    """
    Adding more parallel pipes increases mutual thermal interference,
    so g_N >= g_1 for all N > 1 at all times.
    """
    time = np.geomspace(3600 * 24, 3600 * 24 * 365 * 10, 15)
    g1 = hhe.gfunction(make_field(1), k_s=K_S, alpha=ALPHA, time=time)

    for n in [2, 3, 5]:
        gn = hhe.gfunction(make_field(n), k_s=K_S, alpha=ALPHA, time=time)
        assert np.all(gn >= g1 - 1e-10), (
            f"Multi-pipe: g({n} pipes) should be >= g(1 pipe)"
        )
        print(f"  [PASS] Multi-pipe: g({n}) >= g(1) at all times ✓")

    return time, g1


# ---------------------------------------------------------------------------
# Test 6: Segment additivity
# ---------------------------------------------------------------------------

def test_segment_additivity():
    """
    A trace of two 100m segments must give the same g as one 200m segment
    (since load is distributed equally and both represent the same total pipe).
    Tolerance: 0.5% relative error.
    """
    time = np.geomspace(3600, 3600 * 24 * 365, 15)

    g_one   = hhe.gfunction(make_field(1, seg_length=200.0, n_segments=1),
                             k_s=K_S, alpha=ALPHA, time=time)
    g_two   = hhe.gfunction(make_field(1, seg_length=100.0, n_segments=2),
                             k_s=K_S, alpha=ALPHA, time=time)

    max_err = _assert_close(g_two, g_one, rtol=0.005, label="Segment additivity")
    print(f"  [PASS] Segment additivity: max error = {max_err*100:.3f}%")
    return time, g_one, g_two


# ---------------------------------------------------------------------------
# Test 7: Spatial cutoff
# ---------------------------------------------------------------------------

def test_spatial_cutoff():
    """
    At very early times, the thermal propagation distance is small.
    A distant pipe (separation >> sqrt(4*alpha*t)) should contribute
    nothing, so g(1 pipe far away) == g(1 pipe) at early times.

    Concretely: a field of 2 pipes separated by 50m at t=1h should give
    the same g as 1 pipe, since sqrt(4*alpha*1h) = 0.12m << 50m.
    """
    time_early = np.array([3600.0, 7200.0, 3600.0 * 6])

    g1       = hhe.gfunction(make_field(1),
                              k_s=K_S, alpha=ALPHA, time=time_early)
    g2_close = hhe.gfunction(make_field(2, spacing=1.0),
                              k_s=K_S, alpha=ALPHA, time=time_early)
    g2_far   = hhe.gfunction(make_field(2, spacing=50.0),
                              k_s=K_S, alpha=ALPHA, time=time_early)

    # Far pipe at early times: g2_far should be very close to g1
    # (interaction is zero, g averages over 2 identical uncoupled pipes)
    max_err = np.abs(g2_far - g1).max() / g1.max()
    assert max_err < 0.001, (
        f"Spatial cutoff: distant pipe at early time should not contribute "
        f"(max error = {max_err:.4f})"
    )

    # Close pipe: should be larger than g1 at longer times
    t_long = np.array([3600.0 * 24 * 30])
    g1_l  = hhe.gfunction(make_field(1),          k_s=K_S, alpha=ALPHA, time=t_long)
    g2_l  = hhe.gfunction(make_field(2, spacing=1.0), k_s=K_S, alpha=ALPHA, time=t_long)
    assert g2_l[0] > g1_l[0], "Nearby pipe should increase g at longer times"

    print(f"  [PASS] Spatial cutoff: distant pipe negligible at early times ✓")
    return time_early, g1, g2_close, g2_far


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(outdir: str):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs(outdir, exist_ok=True)
    t_s = DEPTH**2 / (9.0 * ALPHA)   # Eskilson time scale

    def ln_t(time):
        return np.log(time / t_s)

    # ------------------------------------------------------------------
    # Figure 1: ILS convergence
    # ------------------------------------------------------------------
    print("  Plotting: ILS convergence...")
    L = 200.0
    time = np.geomspace(60, 3600 * 24 * 365, 40)
    g_ils  = ils_gfunc(time, R_PIPE, ALPHA)
    g_hfls = hhe.gfunction(make_field(1, seg_length=L),
                            k_s=K_S, alpha=ALPHA, time=time)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(ln_t(time), g_ils,  'k--', lw=1.5, label='ILS (infinite medium)')
    ax.plot(ln_t(time), g_hfls, 'b-',  lw=1.5, label='HFLS (with image source)')
    ax.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax.set_ylabel(r'$g$')
    ax.set_title('Single pipe: HFLS vs ILS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark where image source becomes significant (sqrt(4at)/D = 1)
    t_image = DEPTH**2 / (4.0 * ALPHA)
    ax.axvline(np.log(t_image / t_s), color='gray', lw=0.8, ls=':')
    ax.text(np.log(t_image / t_s) + 0.1, ax.get_ylim()[0] + 0.2,
            r'$\sqrt{4\alpha t} = D$', fontsize=8, color='gray')

    ax2 = axes[1]
    err_pct = (g_hfls - g_ils) / g_ils * 100
    ax2.plot(ln_t(time), err_pct, 'r-', lw=1.5)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.axvline(np.log(t_image / t_s), color='gray', lw=0.8, ls=':',
                label=r'$\sqrt{4\alpha t} = D$')
    ax2.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax2.set_ylabel('(HFLS − ILS) / ILS  [%]')
    ax2.set_title('Deviation from ILS')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f'Test 1: ILS convergence  —  single pipe, L={L}m, D={DEPTH}m, '
        r'$r_b$' + f'={R_PIPE*1000:.0f}mm',
        fontsize=10
    )
    fig.tight_layout()
    path = os.path.join(outdir, 'test1_ils_convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved → {path}")

    # ------------------------------------------------------------------
    # Figure 2: Image source effect & finite length
    # ------------------------------------------------------------------
    print("  Plotting: Image source & finite length effects...")
    time = np.geomspace(60, 3600 * 24 * 365 * 25, 40)

    g_ils_ref = ils_gfunc(time, R_PIPE, ALPHA)
    g_L10   = hhe.gfunction(make_field(1, seg_length=10.0),
                             k_s=K_S, alpha=ALPHA, time=time)
    g_L50   = hhe.gfunction(make_field(1, seg_length=50.0),
                             k_s=K_S, alpha=ALPHA, time=time)
    g_L200  = hhe.gfunction(make_field(1, seg_length=200.0),
                             k_s=K_S, alpha=ALPHA, time=time)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ln_t(time), g_ils_ref, 'k--', lw=1.2, label='ILS (inf. medium, inf. length)')
    ax.plot(ln_t(time), g_L10,  lw=1.5, label='HFLS L=10 m')
    ax.plot(ln_t(time), g_L50,  lw=1.5, label='HFLS L=50 m')
    ax.plot(ln_t(time), g_L200, lw=1.5, label='HFLS L=200 m')

    # Annotate where sqrt(4at) = L/2 for each pipe length
    for L_ann, g_ann in [(10.0, g_L10), (50.0, g_L50), (200.0, g_L200)]:
        t_end = (L_ann / 2)**2 / (4.0 * ALPHA)
        if time[0] < t_end < time[-1]:
            ax.axvline(np.log(t_end / t_s), color='gray', lw=0.5, ls=':')

    # Mark where image source kicks in
    t_image = DEPTH**2 / (4.0 * ALPHA)
    ax.axvline(np.log(t_image / t_s), color='red', lw=0.8, ls='--', alpha=0.5,
               label=r'$\sqrt{4\alpha t} = D$ (image source)')

    ax.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax.set_ylabel(r'$g$')
    ax.set_title(f'Test 2 & 3: Image source and finite length effects (D={DEPTH}m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'test2_image_and_length.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved → {path}")

    # ------------------------------------------------------------------
    # Figure 3: Multi-pipe field
    # ------------------------------------------------------------------
    print("  Plotting: Multi-pipe field...")
    time = np.geomspace(3600, 3600 * 24 * 365 * 25, 40)
    n_list = [1, 2, 3, 5, 10]
    g_multi = {n: hhe.gfunction(make_field(n, spacing=1.0),
                                 k_s=K_S, alpha=ALPHA, time=time)
               for n in n_list}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for n, g in g_multi.items():
        ax.plot(ln_t(time), g, lw=1.5, label=f'N={n}')
    ax.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax.set_ylabel(r'$g$')
    ax.set_title(f'Multi-pipe g-functions (spacing=1.0 m, D={DEPTH}m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    g_base = g_multi[1]
    for n in n_list[1:]:
        delta = g_multi[n] - g_base
        ax2.plot(ln_t(time), delta, lw=1.5, label=f'N={n} − N=1')
    ax2.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax2.set_ylabel(r'$g_N - g_1$  (mutual interference)')
    ax2.set_title('Additional temperature rise from pipe interactions')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Test 5: Multi-pipe field interactions', fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, 'test3_multi_pipe.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved → {path}")

    # ------------------------------------------------------------------
    # Figure 4: Spacing sensitivity
    # ------------------------------------------------------------------
    print("  Plotting: Pipe spacing sensitivity...")
    time = np.geomspace(3600, 3600 * 24 * 365 * 25, 40)
    spacings = [0.5, 1.0, 1.5, 2.0, 3.0]
    n_fixed = 5

    fig, ax = plt.subplots(figsize=(7, 5))
    for sp in spacings:
        g = hhe.gfunction(make_field(n_fixed, spacing=sp),
                          k_s=K_S, alpha=ALPHA, time=time)
        ax.plot(ln_t(time), g, lw=1.5, label=f's={sp} m')

    g1 = hhe.gfunction(make_field(1), k_s=K_S, alpha=ALPHA, time=time)
    ax.plot(ln_t(time), g1, 'k--', lw=1.2, label='N=1 (no interaction)')

    ax.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax.set_ylabel(r'$g$')
    ax.set_title(f'Spacing sensitivity: N={n_fixed} pipes, D={DEPTH}m')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'test4_spacing_sensitivity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved → {path}")

    # ------------------------------------------------------------------
    # Figure 5: Segment additivity
    # ------------------------------------------------------------------
    print("  Plotting: Segment additivity...")
    time = np.geomspace(3600, 3600 * 24 * 365 * 10, 30)
    g_one = hhe.gfunction(make_field(1, seg_length=200.0, n_segments=1),
                          k_s=K_S, alpha=ALPHA, time=time)
    g_two = hhe.gfunction(make_field(1, seg_length=100.0, n_segments=2),
                          k_s=K_S, alpha=ALPHA, time=time)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(ln_t(time), g_one, 'b-',  lw=1.5, label='1 segment × 200 m')
    ax.plot(ln_t(time), g_two, 'r--', lw=1.5, label='2 segments × 100 m')
    ax.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax.set_ylabel(r'$g$')
    ax.set_title('Test 6: Segment additivity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    err = (g_two - g_one) / g_one * 100
    ax2.plot(ln_t(time), err, 'k-', lw=1.5)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax2.set_ylabel('Relative difference [%]')
    ax2.set_title('Difference between discretisations')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'test5_segment_additivity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved → {path}")

    # ------------------------------------------------------------------
    # Figure 6: Spatial cutoff demo
    # ------------------------------------------------------------------
    print("  Plotting: Spatial cutoff...")
    time = np.geomspace(3600, 3600 * 24 * 365 * 5, 40)
    g1      = hhe.gfunction(make_field(1),           k_s=K_S, alpha=ALPHA, time=time)
    g2_1m   = hhe.gfunction(make_field(2, spacing=1.0),  k_s=K_S, alpha=ALPHA, time=time)
    g2_5m   = hhe.gfunction(make_field(2, spacing=5.0),  k_s=K_S, alpha=ALPHA, time=time)
    g2_50m  = hhe.gfunction(make_field(2, spacing=50.0), k_s=K_S, alpha=ALPHA, time=time)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ln_t(time), g1,     'k--', lw=1.2, label='N=1')
    ax.plot(ln_t(time), g2_1m,  lw=1.5, label='N=2, s=1 m')
    ax.plot(ln_t(time), g2_5m,  lw=1.5, label='N=2, s=5 m')
    ax.plot(ln_t(time), g2_50m, lw=1.5, label='N=2, s=50 m')

    # Mark where sqrt(4at) = spacing for each
    for s_ann, g_ann in [(1.0, g2_1m), (5.0, g2_5m), (50.0, g2_50m)]:
        t_ann = s_ann**2 / (4.0 * ALPHA)
        if time[0] < t_ann < time[-1]:
            ax.axvline(np.log(t_ann / t_s), color='gray', lw=0.5, ls=':')

    ax.set_xlabel(r'$\ln(t\,/\,t_s)$')
    ax.set_ylabel(r'$g$')
    ax.set_title(
        f'Test 7: Spatial cutoff — interaction onset vs pipe spacing\n'
        f'(dotted lines: $\\sqrt{{4\\alpha t}} = s$)'
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'test6_spatial_cutoff.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved → {path}")

    print(f"\n  All plots saved to: {outdir}/")


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

def test_1_ils_convergence():
    test_ils_convergence()

def test_2_image_source_effect():
    test_image_source_effect()

def test_3_finite_length_effect():
    test_finite_length_effect()

def test_4_monotonicity():
    test_monotonicity()

def test_5_multi_pipe_interaction():
    test_multi_pipe_interaction()

def test_6_segment_additivity():
    test_segment_additivity()

def test_7_spatial_cutoff():
    test_spatial_cutoff()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hhe_gfunc test suite')
    parser.add_argument(
        '--plots', action='store_true',
        help='Generate validation plots in tests/plots/'
    )
    parser.add_argument(
        '--plotdir', default=os.path.join(os.path.dirname(__file__), 'plots'),
        help='Output directory for plots'
    )
    args = parser.parse_args()

    tests = [
        ("1: ILS convergence",        test_ils_convergence),
        ("2: Image source effect",     test_image_source_effect),
        ("3: Finite length effect",    test_finite_length_effect),
        ("4: Monotonicity",            test_monotonicity),
        ("5: Multi-pipe interaction",  test_multi_pipe_interaction),
        ("6: Segment additivity",      test_segment_additivity),
        ("7: Spatial cutoff",          test_spatial_cutoff),
    ]

    passed = 0
    failed = 0
    print("=" * 60)
    print("hhe_gfunc test suite")
    print("=" * 60)

    for name, fn in tests:
        print(f"\nTest {name}")
        print("-" * 40)
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if args.plots or True:   # always plot when running standalone
        print("\nGenerating plots...")
        make_plots(args.plotdir)

    sys.exit(0 if failed == 0 else 1)
