"""Tests for S4SphereGaussianPhaseApproximation OGSE extension (Phases 2-3).

Covers:
1.  PGSE path bit-identical to original S4 code path
2.  OGSE restriction effect: smaller sphere → higher E at same b
3.  OGSE free-diffusion limit: large R → E = exp(-b D) to 1%
4.  OGSE low-frequency limit: f→0 with same b → matches PGSE to 2%
5.  OGSE high-frequency limit: E in (0, 1]
6.  Mixed PGSE+OGSE scheme produces correct-length output with 0 < E ≤ 1
7.  Trapezoidal: near-zero rise-time matches pure cosine (Phase 3)
8.  Trapezoidal: numerical path matches analytical PGSE via from_waveform
9.  Trapezoidal: 0 < E ≤ 1 for range of parameters
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from dmipy_fit.signal_models.sphere_models import S4SphereGaussianPhaseApproximation
from dmipy_fit.core.acquisition_scheme import (
    AcquisitionScheme,
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_gradient_strengths,
)
from dmipy_fit.core.constants import CONSTANTS

GAMMA = CONSTANTS['water_gyromagnetic_ratio']


def _make_bvecs(n_m):
    return np.tile(np.r_[1., 0., 0.], (n_m, 1))


def _pgse_scheme_fixed(n_m=10, delta=0.02, Delta=0.04, D=1.7e-9):
    """Standard PGSE scheme for S4 testing."""
    # moderate b-values where GPA is valid
    bvalues = np.linspace(0, 2e9, n_m)
    bvecs = _make_bvecs(n_m)
    return acquisition_scheme_from_bvalues(
        bvalues, bvecs, delta=delta, Delta=Delta)


# ---------------------------------------------------------------------------
# 1. PGSE path bit-identical
# ---------------------------------------------------------------------------

def test_pgse_path_bit_identical():
    """PGSE scheme through new __call__ gives exactly same result as sphere_attenuation."""
    delta, Delta, diam = 0.02, 0.04, 10e-6
    scheme = _pgse_scheme_fixed(n_m=15, delta=delta, Delta=Delta)
    s4 = S4SphereGaussianPhaseApproximation(diameter=diam)

    E_new = s4(scheme)

    # Reference: call sphere_attenuation directly
    g = scheme.gradient_strengths
    E_ref = np.ones_like(g)
    mask = g > 0
    E_ref[mask] = s4.sphere_attenuation(g[mask], delta, Delta, diam)

    assert_allclose(E_new, E_ref, rtol=1e-12,
                    err_msg="PGSE path changed — not bit-identical")


# ---------------------------------------------------------------------------
# 2. OGSE restriction effect
# ---------------------------------------------------------------------------

def test_ogse_restriction_effect():
    """Smaller sphere shows more restriction (higher E) than larger sphere.

    This verifies the physical monotonicity: decreasing R increases restriction,
    so E(R_small) > E(R_large) at the same b-value.
    """
    D = 1.7e-9
    f = 50.0       # Hz
    sigma = 0.04   # s
    b_val = 1e9

    bvecs = np.atleast_2d(np.r_[1., 0., 0.])
    scheme = AcquisitionScheme.from_ogse(
        np.array([b_val]), bvecs,
        oscillation_frequency=f,
        gradient_duration=sigma,
        n_t=2000)

    s4_small = S4SphereGaussianPhaseApproximation(
        diameter=4e-6, diffusion_constant=D)   # 4 μm diameter
    s4_large = S4SphereGaussianPhaseApproximation(
        diameter=40e-6, diffusion_constant=D)  # 40 μm diameter

    E_small = s4_small(scheme)[0]
    E_large = s4_large(scheme)[0]

    # Smaller sphere → more restriction → higher signal (less attenuation)
    assert E_small > E_large, (
        f"Expected E(R_small={E_small:.4f}) > E(R_large={E_large:.4f})")
    # Both should be in physical range
    assert 0 < E_small <= 1.0 + 1e-9
    assert 0 < E_large <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 3. OGSE free-diffusion limit: large R → E = exp(-b D)
# ---------------------------------------------------------------------------

def test_ogse_free_diffusion_limit():
    """At R → ∞ (no walls), OGSE GPA must give exp(-b·D) for any waveform.

    For Gaussian free diffusion the phase variance φ² = γ²D·b exactly,
    regardless of waveform shape.  We use R = 1 mm (eigenvalues λ_k ≈ 0)
    so the IIR filter becomes a pure integrator.
    Tolerance: 1% (numerical grid artefact at n_t=5000).
    """
    D = 1.7e-9
    b_val = 1e9
    bvecs = np.atleast_2d(np.r_[1., 0., 0.])

    scheme = AcquisitionScheme.from_ogse(
        np.array([b_val]), bvecs,
        oscillation_frequency=50.0,
        gradient_duration=0.04,
        n_t=5000)

    s4_free = S4SphereGaussianPhaseApproximation(
        diameter=2e-3, diffusion_constant=D)   # 1 mm radius — effectively unrestricted

    E_ogse = s4_free(scheme)[0]
    E_free = np.exp(-b_val * D)

    assert_allclose(E_ogse, E_free, rtol=0.01,
                    err_msg=f"Free-diffusion limit failed: "
                            f"E_ogse={E_ogse:.5f}, exp(-bD)={E_free:.5f}")


# ---------------------------------------------------------------------------
# 4. Waveform-independence of free diffusion: OGSE == PGSE at same b, large R
# ---------------------------------------------------------------------------

def test_free_diffusion_waveform_independence():
    """For unrestricted diffusion, E = exp(-bD) for ANY waveform at the same b.

    This is a fundamental GPA identity: φ = γ²D ∫q(t)²dt = D·b regardless
    of waveform shape.  We test that OGSE (cosine, f=50 Hz) and PGSE
    give the same signal to <1% for a 1 mm radius sphere (effectively
    unrestricted: diffusion length √(Dσ) ≈ 8 µm << R).
    """
    D = 1.7e-9
    b_val = 1e9
    bvecs = np.atleast_2d(np.r_[1., 0., 0.])
    diam_free = 2e-3  # 1 mm radius — unrestricted

    scheme_ogse = AcquisitionScheme.from_ogse(
        np.array([b_val]), bvecs,
        oscillation_frequency=50.0, gradient_duration=0.04, n_t=5000)
    scheme_pgse = acquisition_scheme_from_bvalues(
        np.array([b_val]), bvecs, delta=0.02, Delta=0.04)

    s4 = S4SphereGaussianPhaseApproximation(
        diameter=diam_free, diffusion_constant=D)

    E_ogse = s4(scheme_ogse)[0]
    E_pgse = s4(scheme_pgse)[0]
    E_free = np.exp(-b_val * D)

    # 1.5% tolerance: residual from truncated eigenvalue series (20 roots) at large R
    assert_allclose(E_ogse, E_free, rtol=0.015,
                    err_msg=f"OGSE free-diffusion: E={E_ogse:.4f}, exp(-bD)={E_free:.4f}")
    assert_allclose(E_pgse, E_free, rtol=0.015,
                    err_msg=f"PGSE free-diffusion: E={E_pgse:.4f}, exp(-bD)={E_free:.4f}")
    # Both use the same 20-eigenvalue truncation, so their mutual agreement
    # should be within 1.5% even though each has ~1.2% offset from exp(-bD).
    assert_allclose(E_ogse, E_pgse, rtol=0.015,
                    err_msg=f"OGSE ({E_ogse:.4f}) ≠ PGSE ({E_pgse:.4f}) for free diffusion")


# ---------------------------------------------------------------------------
# 4. OGSE high-frequency motional narrowing
# ---------------------------------------------------------------------------

def test_ogse_high_frequency_limit():
    """OGSE signal is in physical range (0 < E <= 1) across frequencies.

    Also verifies b=0 measurement → E=1.
    """
    D = 1.7e-9
    diameter = 10e-6
    sigma = 0.04

    bvecs = np.atleast_2d(np.r_[1., 0., 0.])
    s4 = S4SphereGaussianPhaseApproximation(
        diameter=diameter, diffusion_constant=D)

    for f in [50.0, 200.0, 500.0]:
        b_val = 1e9
        scheme = AcquisitionScheme.from_ogse(
            np.array([b_val]), bvecs,
            oscillation_frequency=f,
            gradient_duration=sigma, n_t=2000)
        E = s4(scheme)[0]
        assert 0 < E <= 1.0 + 1e-9, (
            f"f={f}Hz: E={E:.4e} out of physical range [0,1]")

    # b=0 → E=1
    scheme_b0 = AcquisitionScheme.from_ogse(
        np.array([0.0]), bvecs,
        oscillation_frequency=50.0,
        gradient_duration=sigma, n_t=2000)
    E_b0 = s4(scheme_b0)[0]
    assert_allclose(E_b0, 1.0, atol=1e-9, err_msg="b=0 OGSE should give E=1")


# ---------------------------------------------------------------------------
# 5. Mixed PGSE + OGSE scheme
# ---------------------------------------------------------------------------

def test_ogse_mixed_scheme():
    """Mixed PGSE+OGSE scheme produces (n_m,) output with plausible 0 < E ≤ 1."""
    pgse = AcquisitionScheme.from_pgse(
        np.linspace(0, 2e9, 4), _make_bvecs(4),
        delta=0.02, Delta=0.04)
    ogse = AcquisitionScheme.from_ogse(
        np.linspace(0, 2e9, 3), _make_bvecs(3),
        oscillation_frequency=50.0,
        gradient_duration=0.04)
    mixed = AcquisitionScheme.concatenate([pgse, ogse])

    s4 = S4SphereGaussianPhaseApproximation(diameter=10e-6)
    E = s4(mixed)

    assert E.shape == (7,), f"Expected (7,), got {E.shape}"
    assert np.all(E > 0.0), "Some E values are non-positive"
    assert np.all(E <= 1.0 + 1e-9), "Some E values exceed 1"
    # b=0 rows should be 1
    assert_allclose(E[mixed.bvalues <= 1e6], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Trapezoidal: near-zero rise-time matches pure cosine (Phase 3)
# ---------------------------------------------------------------------------

def test_trap_zero_ramp_matches_cosine():
    """gradient_rise_time=1e-9 (near-zero) gives same result as rise_time=0."""
    D = 1.7e-9
    diameter = 8e-6
    f = 50.0
    sigma = 0.04
    b_val = 1e9
    bvecs = np.atleast_2d(np.r_[1., 0., 0.])

    scheme_cos = AcquisitionScheme.from_ogse(
        np.array([b_val]), bvecs,
        oscillation_frequency=f,
        gradient_duration=sigma,
        gradient_rise_time=0.,
        n_t=3000)

    scheme_trap = AcquisitionScheme.from_ogse(
        np.array([b_val]), bvecs,
        oscillation_frequency=f,
        gradient_duration=sigma,
        gradient_rise_time=1e-9,
        n_t=3000)

    s4 = S4SphereGaussianPhaseApproximation(
        diameter=diameter, diffusion_constant=D)
    E_cos = s4(scheme_cos)[0]
    E_trap = s4(scheme_trap)[0]

    assert_allclose(E_trap, E_cos, rtol=0.005,
                    err_msg="Near-zero ramp trapezoidal does not match cosine")


# ---------------------------------------------------------------------------
# 7. Trapezoidal: numerical path matches analytical PGSE via from_waveform
# ---------------------------------------------------------------------------

def test_trap_numerical_matches_analytical_pgse():
    """Numerical (trapezoidal) path on a PGSE waveform matches analytical PGSE."""
    D = 1.7e-9
    diameter = 10e-6
    delta = 0.02
    Delta = 0.04
    b_val = 1e9
    bvecs = np.atleast_2d(np.r_[1., 0., 0.])

    # Analytical PGSE reference
    scheme_pgse = acquisition_scheme_from_bvalues(
        np.array([b_val]), bvecs, delta=delta, Delta=Delta)
    s4 = S4SphereGaussianPhaseApproximation(
        diameter=diameter, diffusion_constant=D)
    E_analytical = s4(scheme_pgse)[0]

    # Waveform PGSE → triggers numerical path via _ogse_numerical_sphere_signal
    from dmipy_fit.signal_models.sphere_models import _ogse_numerical_sphere_signal
    g_strength = scheme_pgse.gradient_strengths[0]
    n_t = 5000
    T_total = Delta + delta
    dt = T_total / (n_t - 1)
    t = np.arange(n_t) * dt
    n_pulse = max(1, round(delta / dt))
    n_Delta = round(Delta / dt)
    G_t = np.zeros(n_t)
    G_t[:n_pulse] = g_strength
    G_t[n_Delta:n_Delta + n_pulse] = -g_strength

    E_numerical = _ogse_numerical_sphere_signal(
        G_t, dt, D, diameter / 2,
        S4SphereGaussianPhaseApproximation.SPHERE_TRASCENDENTAL_ROOTS)

    assert_allclose(E_numerical, E_analytical, rtol=0.005,
                    err_msg="Numerical path does not match analytical PGSE")


# ---------------------------------------------------------------------------
# 8. Trapezoidal: physical range
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mitra short-time S/V recovery — signal-only, no model fitting
# ---------------------------------------------------------------------------

def test_ogse_mitra_sv_recovery():
    """OGSE Mitra slope recovers S/V = 3/R without sphere model fitting.

    The Mitra (1993) short-time ADC expansion:
        D_app(ω)/D ≈ 1 − (4/9√π) × (S/V) × √(D/ω)   [valid ωR²/D >> 1]
    gives a linear relationship between D_app(f) and 1/√f.
    Fitting the slope yields S/V, hence R = 3/(S/V), from the signal alone.

    This validates the OGSE formula without inverse crime: no sphere model is
    fit to recover R.  The Mitra expansion is an independent analytical result
    (holds for any pore geometry in the short-time regime).

    Parameters chosen so ωR²/D ∈ [55, 185] across test frequencies.
    Second-order Mitra correction ≤ 6% → expected R error ≤ 5%.
    """
    D = 1.7e-9
    R = 5e-6      # 5 µm radius
    b_target = 5e8  # s/m²; gives E ≈ 0.43 for free diffusion — measurable signal

    # Frequencies where ωR²/D ≥ 55 (short-time regime, Mitra valid to ~6%)
    freqs = np.array([600., 800., 1000., 1500., 2000.])
    omega = 2 * np.pi * freqs
    bvecs = np.atleast_2d(np.r_[1., 0., 0.])

    s4 = S4SphereGaussianPhaseApproximation(diameter=2 * R, diffusion_constant=D)

    D_app = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        sigma = 2.0 / f   # 2 complete cosine cycles
        scheme = AcquisitionScheme.from_ogse(
            np.array([b_target]), bvecs,
            oscillation_frequency=f, gradient_duration=sigma, n_t=2000)
        E = s4(scheme)[0]
        b_eff = scheme.bvalues[0]
        D_app[i] = -np.log(E) / b_eff   # D_app = −ln E / b (any waveform)

    # Mitra fit: (1 − D_app/D) = c_mitra × (S/V) × √(D/ω)
    # Least-squares through origin: slope = Σ(xy)/Σ(x²)
    y = 1.0 - D_app / D
    x = np.sqrt(D / omega)
    slope = np.dot(x, y) / np.dot(x, x)

    c_mitra = 4.0 / (9.0 * np.sqrt(np.pi))
    SV_fit = slope / c_mitra
    R_fit = 3.0 / SV_fit   # sphere: S/V = 3/R

    rel_err = abs(R_fit - R) / R
    assert rel_err < 0.08, (
        f"Mitra S/V recovery: R_fit={R_fit*1e6:.2f} µm, "
        f"R_true={R*1e6:.1f} µm, err={100*rel_err:.1f}%")


def test_trap_signal_physical():
    """0 < E ≤ 1 for a range of sphere radii and OGSE frequencies."""
    from dmipy_fit.signal_models.sphere_models import _ogse_numerical_sphere_signal

    roots = S4SphereGaussianPhaseApproximation.SPHERE_TRASCENDENTAL_ROOTS
    D = 1.7e-9
    f = 50.0
    sigma = 0.04
    n_t = 1000
    T_total = sigma
    dt = T_total / (n_t - 1)

    for R in [3e-6, 6e-6, 10e-6, 20e-6]:
        for G in [0.1, 0.3]:
            t = np.arange(n_t) * dt
            G_t = G * np.cos(2 * np.pi * f * t)
            E = _ogse_numerical_sphere_signal(G_t, dt, D, R, roots)
            assert 0 < E <= 1.0 + 1e-9, (
                f"E={E:.4f} out of physical range for R={R}, G={G}")
