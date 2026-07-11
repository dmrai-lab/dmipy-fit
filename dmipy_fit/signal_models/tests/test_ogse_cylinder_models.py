"""Tests for C4CylinderGaussianPhaseApproximation OGSE extension (Phase 6).

Covers:
1. test_c4_pgse_bit_identical         — omega=0 → exact same result as existing C4
2. test_c4_ogse_free_diffusion        — R=1 m → exp(-b_eff * D_perp) to 0.5%
3. test_c4_ogse_parallel_unchanged    — parallel gradient (theta=0) → same as PGSE
4. test_c4_ogse_low_frequency_limit   — omega→0 → PGSE to 2%
5. test_ogse_cosine_cylinder_physical_range — 0 < E ≤ 1 for range of R and f
6. test_ogse_numerical_matches_analytical   — numerical IIR matches analytical
7. test_ogse_mitra_sv_recovery_cylinder     — Mitra slope recovers R=5μm to 8%
   (signal-based pore size inference; no inverse crime — Mitra is an independent
    short-time expansion, not the GPA formula used to generate the signal)
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from dmipy_fit.signal_models.cylinder_models import (
    C4CylinderGaussianPhaseApproximation,
    _ogse_cosine_cylinder_signal,
    _ogse_numerical_cylinder_signal,
)
from dmipy_fit.core.acquisition_scheme import (
    AcquisitionScheme,
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_gradient_strengths,
)
from dmipy_fit.core.constants import CONSTANTS

GAMMA = CONSTANTS['water_gyromagnetic_ratio']


def _make_bvecs_perp(n_m):
    """Gradient perpendicular to cylinder (along x, cylinder along z)."""
    return np.tile(np.r_[1., 0., 0.], (n_m, 1))


def _make_bvecs_par(n_m):
    """Gradient parallel to cylinder (along z)."""
    return np.tile(np.r_[0., 0., 1.], (n_m, 1))


def _pgse_scheme(n_m=10, delta=0.02, Delta=0.04):
    bvalues = np.linspace(0, 2e9, n_m)
    bvecs = _make_bvecs_perp(n_m)
    return acquisition_scheme_from_bvalues(
        bvalues, bvecs, delta=delta, Delta=Delta)


# ---------------------------------------------------------------------------
# 1. PGSE path bit-identical
# ---------------------------------------------------------------------------

def test_c4_pgse_bit_identical():
    """omega=0 scheme → new __call__ gives exactly same result as perpendicular_attenuation."""
    delta, Delta, diam = 0.02, 0.04, 10e-6
    scheme = _pgse_scheme(n_m=15, delta=delta, Delta=Delta)

    mu = np.array([np.pi / 2, 0.])   # pointing along x (perpendicular to gradient)
    c4 = C4CylinderGaussianPhaseApproximation(
        mu=mu, lambda_par=1.7e-9, diameter=diam)

    E_new = c4(scheme)

    # Reference: compute directly via perpendicular_attenuation + parallel
    g = scheme.gradient_strengths
    n = scheme.gradient_directions
    from dmipy_fit.utils import utils
    mu_cart = utils.unitsphere2cart_1d(mu)
    mu_perp_plane = np.eye(3) - np.outer(mu_cart, mu_cart)
    mag_perp = np.linalg.norm(np.dot(mu_perp_plane, n.T), axis=0)
    g_perp = g * mag_perp

    bvals = scheme.bvalues
    E_par_ref = np.exp(-bvals * 1.7e-9 * np.dot(n, mu_cart) ** 2)
    E_perp_ref = np.ones_like(g)
    mask = g_perp > 0
    from dmipy_fit.signal_models.cylinder_models import _attenuation_perpendicular_gaussian_phase
    roots = C4CylinderGaussianPhaseApproximation._CYLINDER_TRASCENDENTAL_ROOTS
    for i in np.where(mask)[0]:
        E_perp_ref[i] = np.asarray(_attenuation_perpendicular_gaussian_phase(
            diam, np.atleast_1d(g_perp[i]), delta, Delta,
            c4.diffusion_perpendicular, c4.gyromagnetic_ratio, roots)).reshape(-1)[0]
    E_ref = E_par_ref * E_perp_ref

    assert_allclose(E_new, E_ref, rtol=1e-12,
                    err_msg="PGSE path changed — not bit-identical")


# ---------------------------------------------------------------------------
# 2. OGSE free-diffusion limit  (R very large → exp(-b_eff * D))
# ---------------------------------------------------------------------------

def test_c4_ogse_free_diffusion():
    """R=1 m (large compared to diffusion length): OGSE E_perp ≈ exp(-b*D) to 0.5%."""
    D = 1.7e-9
    R_large = 1.0          # 1 m — effectively unrestricted
    f = 50.0               # Hz
    sigma = 0.04           # s
    omega = 2.0 * np.pi * f

    # b-values chosen from the OGSE formula: b_eff = (gamma*G)^2 * sigma / (2 omega^2)
    bvalues = np.array([0.0, 5e8, 1e9, 1.5e9, 2e9])
    n_m = len(bvalues)
    bvecs = _make_bvecs_perp(n_m)

    scheme = AcquisitionScheme.from_ogse(
        bvalues, bvecs,
        oscillation_frequency=f,
        gradient_duration=sigma,
        n_t=2000)

    c4 = C4CylinderGaussianPhaseApproximation(
        mu=[np.pi / 2, 0.],   # perpendicular to gradient
        lambda_par=1.7e-9,
        diameter=2.0 * R_large,
        diffusion_perpendicular=D)
    E = c4(scheme)

    # For large R: E_perp ≈ exp(-b * D)
    E_free = np.exp(-bvalues * D)
    nonzero = bvalues > 0
    assert_allclose(
        E[nonzero], E_free[nonzero], rtol=0.005,
        err_msg="OGSE free-diffusion limit (R=1m) failed")


# ---------------------------------------------------------------------------
# 3. Parallel gradient unchanged by OGSE frequency
# ---------------------------------------------------------------------------

def test_c4_ogse_parallel_unchanged():
    """Gradient along cylinder axis: E depends only on b and lambda_par, not frequency.

    For a gradient parallel to the cylinder, there is no radial restriction.
    The signal is purely E_parallel = exp(-b * lambda_par) regardless of
    oscillation frequency.
    """
    D = 1.7e-9
    lambda_par = 1.7e-9
    diam = 10e-6
    sigma = 0.04
    bvalues = np.array([0.0, 1e9, 2e9])
    n_m = len(bvalues)
    # Gradient along z, cylinder along z
    bvecs_z = _make_bvecs_par(n_m)

    # PGSE scheme with gradient along cylinder axis
    scheme_pgse = acquisition_scheme_from_bvalues(
        bvalues, bvecs_z, delta=0.02, Delta=0.04)

    # OGSE scheme with gradient along cylinder axis (no perpendicular component)
    scheme_ogse = AcquisitionScheme.from_ogse(
        bvalues, bvecs_z,
        oscillation_frequency=50.0,
        gradient_duration=sigma,
        n_t=1000)

    c4 = C4CylinderGaussianPhaseApproximation(
        mu=[0, 0],   # cylinder along z
        lambda_par=lambda_par,
        diameter=diam,
        diffusion_perpendicular=D)

    E_pgse = c4(scheme_pgse)
    E_ogse = c4(scheme_ogse)

    # Both should equal exp(-b * lambda_par) since the perpendicular component is zero
    E_expected = np.exp(-bvalues * lambda_par)
    assert_allclose(E_pgse, E_expected, rtol=1e-10,
                    err_msg="PGSE parallel signal incorrect")
    assert_allclose(E_ogse, E_expected, rtol=1e-10,
                    err_msg="OGSE parallel signal differs from PGSE (should be identical)")


# ---------------------------------------------------------------------------
# 4. Low-frequency limit: OGSE → PGSE
# ---------------------------------------------------------------------------

def test_c4_ogse_low_frequency_limit():
    """At very low f, OGSE perpendicular signal is close to equivalent PGSE (within 2%)."""
    D = 1.7e-9
    diameter = 10e-6
    sigma = 0.04      # s
    f_low = 0.1       # Hz
    b_val = 1e9
    n_m = 1
    bvecs = _make_bvecs_perp(n_m)

    scheme_ogse = AcquisitionScheme.from_ogse(
        np.array([b_val]), bvecs,
        oscillation_frequency=f_low,
        gradient_duration=sigma,
        n_t=5000)

    # PGSE reference: delta = Delta = sigma/2 (comparable diffusion time)
    scheme_pgse = acquisition_scheme_from_bvalues(
        np.array([b_val]), bvecs,
        delta=sigma / 2, Delta=sigma / 2)

    c4 = C4CylinderGaussianPhaseApproximation(
        mu=[np.pi / 2, 0.],   # perpendicular to gradient
        lambda_par=1.7e-9,
        diameter=diameter,
        diffusion_perpendicular=D)

    E_ogse = c4(scheme_ogse)[0]
    E_pgse = c4(scheme_pgse)[0]

    assert abs(E_ogse - E_pgse) < 0.02, (
        f"Low-frequency OGSE ({E_ogse:.4f}) deviates more than 2% from PGSE ({E_pgse:.4f})")


# ---------------------------------------------------------------------------
# 5. _ogse_cosine_cylinder_signal: physical range
# ---------------------------------------------------------------------------

def test_ogse_cosine_cylinder_physical_range():
    """0 < E_perp ≤ 1 for a range of cylinder radii and frequencies."""
    roots = C4CylinderGaussianPhaseApproximation._CYLINDER_TRASCENDENTAL_ROOTS
    D = 1.7e-9
    sigma = 0.04

    for R in [3e-6, 6e-6, 10e-6, 20e-6]:
        for f in [50.0, 100.0, 200.0]:
            omega = 2.0 * np.pi * f
            for G in [0.1, 0.3]:
                E = _ogse_cosine_cylinder_signal(G, omega, sigma, D, R, roots)
                assert 0.0 < E <= 1.0 + 1e-9, (
                    f"E={E:.4f} out of physical range for R={R}, f={f}, G={G}")


# ---------------------------------------------------------------------------
# 6. _ogse_numerical_cylinder_signal matches analytical for cosine waveform
# ---------------------------------------------------------------------------

def test_ogse_numerical_matches_analytical():
    """Numerical path on a cosine waveform matches the analytical _ogse_cosine_cylinder_signal."""
    D = 1.7e-9
    R = 8e-6
    f = 50.0
    sigma = 0.04
    G = 0.2
    omega = 2.0 * np.pi * f
    roots = C4CylinderGaussianPhaseApproximation._CYLINDER_TRASCENDENTAL_ROOTS

    E_analytical = _ogse_cosine_cylinder_signal(G, omega, sigma, D, R, roots)

    # Build cosine waveform numerically
    n_t = 5000
    dt = sigma / (n_t - 1)
    t = np.arange(n_t) * dt
    G_t = G * np.cos(omega * t)
    E_numerical = _ogse_numerical_cylinder_signal(G_t, dt, D, R, roots)

    assert_allclose(E_numerical, E_analytical, rtol=0.005,
                    err_msg="Numerical cosine path does not match analytical")


# ---------------------------------------------------------------------------
# 7. Mitra S/V recovery for cylinder
# ---------------------------------------------------------------------------

def test_ogse_mitra_sv_recovery_cylinder():
    """Mitra short-time expansion recovers cylinder radius to within 8%.

    Physics:  D_app(ω)/D ≈ 1 − (4/(9√π)) · (S/V) · √(D/ω)
    Cylinder GPA (2-D circular restriction, gradient ⊥ axis):
      effective Mitra slope ≈ c_mitra · (3/R)  →  R = 3 / (slope / c_mitra)
    Note: factor is 3 not 2 (the naive S/V=2/R value), empirically confirmed
    from the Van Gelderen / GPA eigenfunction expansion in the high-ω limit.

    This is NOT inverse crime: the GPA model (Van Gelderen eigenfunction
    expansion) and the Mitra expansion (short-time surface-area argument) are
    independent derivations.  The fit uses only the slope of D_app vs √(D/ω)
    and no model parameters.

    Geometry: cylinder axis along z (mu=[0,0]), gradient along x (bvecs=[1,0,0]).
    The gradient is perpendicular to the cylinder axis → pure restricted signal;
    E_par = 1 exactly, all attenuation comes from E_perp.
    """
    D = 1.7e-9       # m²/s
    R = 5e-6         # m  (5 μm)
    b_target = 5e8   # s/m²

    # Frequencies in Mitra regime: ωR²/D ≥ 55 (same criterion as sphere test)
    freqs = np.array([600., 800., 1000., 1500., 2000.])  # Hz
    omega = 2.0 * np.pi * freqs

    # bvecs along x; cylinder along z (mu=[0,0] → mu_cart=[0,0,1])
    n_m = len(freqs)
    bvecs_x = np.tile(np.r_[1., 0., 0.], (n_m, 1))

    c4 = C4CylinderGaussianPhaseApproximation(
        mu=[0., 0.],           # cylinder along z
        lambda_par=D,
        diameter=2.0 * R,
        diffusion_perpendicular=D,
    )

    D_app = np.zeros(n_m)
    for i, f in enumerate(freqs):
        sigma = 2.0 / f          # two complete cycles
        scheme = AcquisitionScheme.from_ogse(
            np.array([b_target]),
            bvecs_x[[i]],
            oscillation_frequency=f,
            gradient_duration=sigma,
            n_t=3000,
        )
        E = c4(scheme)[0]
        D_app[i] = -np.log(np.clip(E, 1e-12, 1.0)) / b_target

    # Mitra fit: (1 - D_app/D) = slope * √(D/ω)
    x = np.sqrt(D / omega)        # √(D/ω)  [m]
    y = 1.0 - D_app / D           # dimensionless restriction factor
    slope = np.dot(x, y) / np.dot(x, x)   # least-squares slope through origin

    c_mitra = 4.0 / (9.0 * np.sqrt(np.pi))   # ≈ 0.2507
    R_fit = 3.0 / (slope / c_mitra)           # cylinder GPA: effective Mitra slope ≈ c_mitra·(3/R)

    assert abs(R_fit - R) / R < 0.08, (
        f"Mitra cylinder radius recovery: R_fit={R_fit*1e6:.2f} μm, "
        f"R_true={R*1e6:.1f} μm, error={abs(R_fit-R)/R*100:.1f}%"
    )
