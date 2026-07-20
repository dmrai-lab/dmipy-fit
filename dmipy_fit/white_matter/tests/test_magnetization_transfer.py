"""Tests for the two-pool MT kernel (dmipy_fit.white_matter.magnetization_transfer).

Validates: (i) the degenerate limit reduces to the surface-relaxivity closed form
(rho -> kappa_MT), i.e. MT is degenerate with surface relaxivity in the diffusion
signal; (ii) the two-pool steady state is algebraically correct; (iii) the qMT
Z-spectrum / MTR is the degeneracy-lifting observable (MTR=0 without a bound pool);
(iv) the geometric forward rate is tied to the substrate S/V; and (v) the analytic
Z-spectrum matches the dmipy-sim Monte-Carlo two-pool oracle (analytic <-> MC), when
a dmipy_sim with the MT kernel is installed.
"""
import numpy as np
import pytest

from dmipy_fit.white_matter import magnetization_transfer as mt
from dmipy_fit.white_matter import surface


# ── (i) degenerate limit == surface relaxivity with rho -> kappa_MT ─────────────
def test_degenerate_limit_equals_surface_interior():
    alpha, scale, kappa_MT = 2.0, 0.304e-6, 1.16e-6
    tau = np.linspace(0, 0.1, 20)
    mt_att = mt.mt_transverse_attenuation("intra", tau, kappa_MT=kappa_MT,
                                          alpha=alpha, scale_diameter=scale)
    surf_att = surface.b_hat_ia(alpha, scale, kappa_MT, tau)   # rho := kappa_MT
    assert np.allclose(mt_att, surf_att)


def test_degenerate_limit_equals_surface_exterior():
    kappa_MT, sv = 1.16e-6, 4.0e5
    tau = np.linspace(0, 0.1, 20)
    mt_att = mt.mt_transverse_attenuation("extra", tau, kappa_MT=kappa_MT,
                                          S_ext_over_V_EA=sv)
    assert np.allclose(mt_att, surface.b_hat_ea_long(kappa_MT, sv, tau))


# ── (ii) two-pool steady state is algebraically correct ────────────────────────
def test_steady_state_matches_linear_solve():
    k_f, k_r, R1a, R1b = 30.0, 100.0, 1.0, 1.0
    M0a = 1.0
    M0b = M0a * k_f / k_r
    for W_a, W_b in [(0.0, 0.0), (5.0, 5000.0), (0.5, 20000.0)]:
        # independent 2x2 solve of the same longitudinal system
        A = np.array([[R1a + k_f + W_a, -k_r],
                      [-k_f, R1b + k_r + W_b]])
        rhs = np.array([R1a * M0a, R1b * M0b])
        Mza_ref = np.linalg.solve(A, rhs)[0] / M0a
        Mza = mt.two_pool_steady_state(k_f, k_r, R1a, R1b, W_a, W_b, M0a=M0a)
        assert Mza == pytest.approx(Mza_ref, rel=1e-10)


def test_no_saturation_is_equilibrium():
    z = mt.two_pool_steady_state(40.0, 120.0, 1.0, 1.0, 0.0, 0.0)
    assert z == pytest.approx(1.0, rel=1e-12)


# ── (iii) the degeneracy-lifting observable: MTR ────────────────────────────────
def test_mtr_zero_without_bound_pool():
    """A pure surface-relaxivity sink (no bound pool, k_f=0) has NO off-resonance
    MTR -- the qMT axis is what distinguishes MT from surface relaxivity.  (At an
    offset >> the narrow free-pool line and modest B1, direct free saturation is
    negligible, so any MTR is the bound pool.)"""
    r = mt.mtr(5000.0, 50.0, k_f=0.0, k_r=100.0, T1a=1.0, T1b=1.0,
               T2a=0.08, T2b=1e-5)
    assert abs(r) < 0.01
    # a bound pool at the SAME (offset, B1) gives a real MT dip -> lifts degeneracy
    r_mt = mt.mtr(5000.0, 50.0, k_f=30.0, k_r=100.0, T1a=1.0, T1b=1.0,
                  T2a=0.08, T2b=1e-5)
    assert r_mt > 0.02


def test_zspectrum_shape():
    kw = dict(k_f=30.0, k_r=100.0, T1a=1.0, T1b=1.0, T2a=0.08, T2b=1e-5)
    offs = np.array([0.0, 500.0, 5000.0, 1e6])
    Z = mt.z_spectrum(offs, 100.0, **kw)
    assert Z[0] < Z[1] < Z[2] < Z[3]           # dip deepest on resonance, recovers
    # the bound line is broad (~1/(2 pi T2b) ~ 16 kHz), so recovery needs a FAR
    # offset -- at 1 MHz the free pool is fully relaxed
    assert Z[-1] == pytest.approx(1.0, abs=5e-3)


# ── (iv) geometric forward rate tied to the substrate S/V ──────────────────────
def test_forward_rate_tracks_substrate_geometry():
    kappa_MT, alpha, scale = 1.5e-5, 2.0, 0.304e-6
    S_ext_over_V_EA = 4.0e5
    # exterior: k_f = kappa_MT * S_ext/V_EA (same geometry as surface relaxivity)
    k_ext = mt.forward_rate_exterior(kappa_MT, S_ext_over_V_EA)
    assert k_ext == pytest.approx(kappa_MT * S_ext_over_V_EA, rel=1e-12)
    assert k_ext > 0
    # interior: k_f = kappa_MT * <4/d>_volume (same <4/d> as the interior surface rate)
    k_int = mt.forward_rate_interior(alpha, scale, kappa_MT)
    assert k_int == pytest.approx(
        kappa_MT * surface.mean_inv_diameter_4(alpha, scale), rel=1e-12)


# ── (v) analytic Z-spectrum matches the dmipy-sim MC two-pool oracle ───────────
def test_analytic_matches_mc_oracle():
    sim_mt = pytest.importorskip("dmipy_sim.mt")   # needs a dmipy_sim with the MT kernel
    k_f, k_r = 30.0, 100.0
    T1a, T1b, T2a, T2b = 1.0, 1.0, 0.08, 1e-5
    # qMT regime where the longitudinal quasi-steady-state holds: modest B1 and
    # offsets >> the narrow free-pool line, so free direct saturation is negligible
    # and the bound pool dominates (a strong B1 near the free line stresses the QSS).
    w1_hz = 50.0
    offs = np.array([2000.0, 5000.0, 10000.0, 30000.0])
    Z_fit = mt.z_spectrum(offs, w1_hz, k_f=k_f, k_r=k_r, T1a=T1a, T1b=T1b,
                          T2a=T2a, T2b=T2b)
    # dmipy-sim oracle: evolve the full two-pool Bloch system to (near) steady state
    M0b = k_f / k_r
    Z_sim = []
    for off in offs:
        dw = 2.0 * np.pi * off
        A = sim_mt.two_pool_generator(R1a=1 / T1a, R2a=1 / T2a, R1b=1 / T1b,
                                      R2b=1 / T2b, k_f=k_f, k_r=k_r, M0a=1.0,
                                      M0b=M0b, dw_a=dw, dw_b=dw,
                                      w1=2 * np.pi * w1_hz)
        s0 = np.array([0, 0, 1.0, 0, 0, M0b, 1.0])
        # t=10s >> T1a=1s so the transient (~e^{-t/T1}) is gone -> true steady state
        Z_sim.append(sim_mt.evolve_two_pool(s0, 10.0, A)[2])
    Z_sim = np.array(Z_sim)
    assert np.allclose(Z_fit, Z_sim, atol=0.02), f"fit {Z_fit} vs sim {Z_sim}"
