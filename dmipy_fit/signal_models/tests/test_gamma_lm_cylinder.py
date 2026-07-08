"""Tests for Gamma_lm angular power spectrum and fast-eigenmode cylinder GPA path.

Tests:
1. test_C_geom_matches_pgse_perpendicular  — C_geom gives correct attenuation for
   purely perpendicular PGSE. Tolerance 40% because alpha1*delta~4.6 at R=5um
   means the fast-eigenmode approximation is marginal for PGSE at delta=20ms.
   This tests the formula implementation, not the approximation quality.
2. test_gamma_lm_fixed_direction_vs_iir   — Gamma_lm path vs IIR for rotating
   waveform (sigma=40ms, R=5um): fast-eigenmode gives ~12% overestimate of phi
   since alpha1*sigma=9.2 is marginal.  Tolerance 15% (tests formula, not approx).
3. test_gamma_lm_physical_range           — 0 < E <= 1 for rotating waveform
4. test_gamma_lm_vs_monte_carlo           — MC validation at R=2um where the
   fast-eigenmode approximation is valid (alpha1*sigma=57.6 >> 1).
   Tolerance: |E_gamma - E_mc| < 0.02 (2%).

Notes on the fast-eigenmode approximation quality:
- R=2um, sigma=40ms: alpha1*sigma=57.6; |E_gamma - E_mc| ~ 0.007 (0.7%) — VALID
- R=5um, sigma=40ms: alpha1*sigma=9.2;  |E_gamma - E_mc| ~ 0.031 (3%) — MARGINAL
- R=5um, delta=20ms: alpha1*delta=4.6;  phi error ~ 36% — NOT VALID for PGSE
The fast-eigenmode limit requires alpha1*T >> 1 where T is the waveform duration.
"""

import numpy as np
import numpy.testing as npt
import pytest
import sys

from dmipy_fit.signal_models.cylinder_models import (
    C4CylinderGaussianPhaseApproximation,
    _compute_C_geom,
    _cylinder_signal_from_gamma_lm,
    _eval_Y2m_at_direction,
    _attenuation_perpendicular_gaussian_phase,
    _ogse_cosine_cylinder_signal,
    _ogse_numerical_cylinder_signal,
)
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.constants import CONSTANTS

GAMMA = CONSTANTS['water_gyromagnetic_ratio']

# Physical parameters
R = 5e-6        # 5 um cylinder radius
R_small = 2e-6  # 2 um cylinder radius (fast-eigenmode approx valid)
D = 1.7e-9      # m^2/s perpendicular diffusivity
G0 = 0.1        # T/m gradient amplitude
SIGMA = 40e-3   # 40 ms gradient duration
N_T = 400       # 400 timesteps -> dt = 0.1 ms
DT = SIGMA / (N_T - 1)

_ROOTS = C4CylinderGaussianPhaseApproximation._CYLINDER_TRASCENDENTAL_ROOTS


def _build_rotating_waveform_and_scheme(R_use=R, G0_use=G0, sigma=SIGMA, n_t=N_T):
    """Build a single-measurement rotating gradient waveform.

    G(t) rotates linearly from x to y over [0, sigma].
    Returns: scheme, G_vec (n_t, 3), dt
    """
    dt = sigma / (n_t - 1)
    t = np.linspace(0.0, sigma, n_t)
    angle = (np.pi / 2.0) * t / sigma   # 0 -> pi/2
    Gx = G0_use * np.cos(angle)
    Gy = G0_use * np.sin(angle)
    Gz = np.zeros(n_t)
    G_vec = np.stack([Gx, Gy, Gz], axis=-1)
    G_arr = G_vec[None, :, :].astype(np.float32)

    grad_dir = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]])
    # The x->y rotating gradient is intentionally NOT moment-nulled (q(TE) != 0):
    # gamma_lm is the angular power spectrum int |G|^2 Y_lm dt, which does not
    # require refocusing, so the moment-null guard is bypassed here.
    scheme = AcquisitionScheme.from_waveform(G_arr, dt, grad_dir,
                                             allow_unrefocused=True)
    return scheme, G_vec, dt


def _build_pgse_scheme_perp(G_strength=G0, delta=10e-3, Delta=30e-3):
    """PGSE scheme: single measurement, gradient along x, cylinder along z."""
    bvecs = np.array([[1.0, 0.0, 0.0]])
    bvals_arr = np.array([
        GAMMA**2 * G_strength**2 * delta**2 * (Delta - delta / 3.0)
    ])
    scheme = AcquisitionScheme.from_pgse(
        bvals_arr, bvecs, delta=delta, Delta=Delta
    )
    return scheme, G_strength, delta, Delta


# ---------------------------------------------------------------------------
# Test 1: C_geom gives correct implementation (not testing approximation quality)
# ---------------------------------------------------------------------------

def test_C_geom_matches_pgse_perpendicular():
    """Verify Gamma_lm path computes C_geom × int|G|^2 dt correctly.

    For PGSE with gradient along x and cylinder along z (purely perpendicular),
    the SH expansion gives phi_perp = C_geom × int|G|^2 dt.

    At R=5um, delta=20ms: alpha1*delta=4.6 (marginal fast-eigenmode condition).
    The fast-eigenmode approximation underestimates the true GPA phi by ~36%,
    so the tolerance is set to 40% to verify the formula is correctly implemented
    without conflating approximation quality.

    The key check: C_geom * int|G|^2 dt is computed correctly from Gamma_lm.
    """
    G_strength = 0.1
    delta = 20e-3
    Delta = 40e-3

    scheme, G_s, delta_, Delta_ = _build_pgse_scheme_perp(
        G_strength=G_strength, delta=delta, Delta=Delta
    )

    # Full Van Gelderen GPA (reference)
    diameter = 2.0 * R
    E_gpa = _attenuation_perpendicular_gaussian_phase(
        diameter, np.array([G_strength]), delta_, Delta_,
        D, GAMMA, _ROOTS
    )
    E_gpa_scalar = float(np.atleast_1d(E_gpa).ravel()[0])

    # Gamma_lm path
    C_geom = _compute_C_geom(R, D, _ROOTS)
    gamma_lm = scheme.gamma_lm(l_max=4)
    mu_z = np.array([0.0, 0.0, 1.0])
    b_par = np.zeros(1)
    E_gamma = _cylinder_signal_from_gamma_lm(
        gamma_lm, b_par, C_geom, mu_z, lambda_par=0.0
    )

    phi_gpa = -np.log(E_gpa_scalar)
    phi_gamma = -np.log(float(E_gamma[0]))

    print(f"\nTest 1: PGSE C_geom formula check (G={G_strength} T/m, R=5um)")
    print(f"  E_gpa (Van Gelderen full) = {E_gpa_scalar:.6f}")
    print(f"  E_gamma (fast-eigenmode)  = {float(E_gamma[0]):.6f}")
    print(f"  phi_gpa = {phi_gpa:.6f}")
    print(f"  phi_gamma = {phi_gamma:.6f}")
    print(f"  phi_gamma/phi_gpa = {phi_gamma/phi_gpa:.4f}")
    print(f"  alpha1*delta = {((_ROOTS[0]/R)**2)*D*delta:.2f} (fast-eig criterion)")
    print(f"  C_geom = {C_geom:.6e}")

    # Sanity: Gamma_lm path produces attenuation in (0,1]
    assert 0.0 < E_gamma[0] <= 1.0, f"Signal out of range: {E_gamma[0]}"

    # Fast-eigenmode approximation is known to differ by ~36% from GPA for these params.
    # The 40% atol tests that the formula is implemented correctly (not approx quality).
    # For R=2um the approximation would be within 5%.
    npt.assert_allclose(
        float(E_gamma[0]), E_gpa_scalar, atol=0.40,
        err_msg=(
            "Gamma_lm path gives clearly wrong result. "
            "Expected ~36% phi discrepancy at R=5um, alpha1*delta=4.6"
        )
    )
    # Also check the C_geom value is in correct ballpark (within 50% of GPA phi)
    assert abs(phi_gamma / phi_gpa - 1.0) < 0.50, (
        f"C_geom phi error too large: {phi_gamma/phi_gpa:.3f}, "
        "likely a formula bug"
    )


# ---------------------------------------------------------------------------
# Test 2: Gamma_lm vs IIR for rotating waveform (sigma=40ms, R=5um)
# ---------------------------------------------------------------------------

def test_gamma_lm_fixed_direction_vs_iir():
    """Gamma_lm path vs exact IIR for rotating waveform (R=5um, sigma=40ms).

    The fast-eigenmode approximation at R=5um, sigma=40ms gives alpha1*sigma=9.2,
    which leads to ~12% overestimate of phi (C_geom is too large by ~12%).
    The tolerance is set to 15% to verify the formula is correctly implemented.

    For R=2um (alpha1*sigma=57.6), the approximation is accurate to <1%.
    """
    scheme, G_vec, dt = _build_rotating_waveform_and_scheme()

    # IIR reference: project gradient onto perpendicular plane (cylinder along z)
    G_perp_t = np.sqrt(G_vec[:, 0]**2 + G_vec[:, 1]**2)  # = G0 always
    E_iir = float(_ogse_numerical_cylinder_signal(G_perp_t, dt, D, R, _ROOTS))

    # Gamma_lm path at mu=z (purely perpendicular)
    C_geom = _compute_C_geom(R, D, _ROOTS)
    gamma_lm = scheme.gamma_lm(l_max=4)
    mu_z = np.array([0.0, 0.0, 1.0])
    b_par = np.zeros(1)
    E_gamma = float(_cylinder_signal_from_gamma_lm(
        gamma_lm, b_par, C_geom, mu_z, lambda_par=0.0
    )[0])

    phi_iir = -np.log(E_iir)
    phi_gamma = -np.log(E_gamma)

    print(f"\nTest 2: Rotating waveform Gamma_lm vs IIR (R=5um, sigma=40ms)")
    print(f"  E_iir   (exact IIR) = {E_iir:.6f}")
    print(f"  E_gamma (fast-eig)  = {E_gamma:.6f}")
    print(f"  phi_iir = {phi_iir:.6f}")
    print(f"  phi_gamma = {phi_gamma:.6f}")
    print(f"  phi_gamma/phi_iir = {phi_gamma/phi_iir:.4f}  (~1.12 expected at R=5um)")
    print(f"  alpha1*sigma = {((_ROOTS[0]/R)**2)*D*SIGMA:.2f}")

    # Verify signal is in physical range
    assert 0.0 < E_gamma <= 1.0, f"Signal out of range: {E_gamma}"

    # Tolerance 15%: fast-eigenmode overestimates phi by ~12% at R=5um, sigma=40ms
    npt.assert_allclose(
        E_gamma, E_iir, atol=0.15,
        err_msg=(
            f"Gamma_lm path disagrees with IIR by more than expected. "
            f"E_gamma={E_gamma:.4f}, E_iir={E_iir:.4f}"
        )
    )


# ---------------------------------------------------------------------------
# Test 3: Physical range for rotating waveform across multiple orientations
# ---------------------------------------------------------------------------

def test_gamma_lm_physical_range():
    """E in (0, 1] for rotating gradient at 20 cylinder axis orientations."""
    scheme, G_vec, dt = _build_rotating_waveform_and_scheme()
    C_geom = _compute_C_geom(R, D, _ROOTS)
    gamma_lm = scheme.gamma_lm(l_max=4)

    n_orient = 20
    thetas = np.full(n_orient, np.pi / 2.0)
    phis = np.linspace(0, 2.0 * np.pi, n_orient, endpoint=False)
    bvals = scheme.bvalues
    n_dirs = scheme.gradient_directions

    errors = []
    for i in range(n_orient):
        mu_cart = np.array([
            np.sin(thetas[i]) * np.cos(phis[i]),
            np.sin(thetas[i]) * np.sin(phis[i]),
            np.cos(thetas[i])
        ])
        b_par = bvals * np.dot(n_dirs, mu_cart) ** 2
        E = _cylinder_signal_from_gamma_lm(
            gamma_lm, b_par, C_geom, mu_cart, lambda_par=1.7e-9
        )
        if not (0.0 < E[0] <= 1.0):
            errors.append((i, float(E[0])))

    print(f"\nTest 3: Physical range check over {n_orient} orientations")
    if errors:
        print(f"  Violations: {errors}")
    else:
        print("  All E in (0, 1] — PASS")

    assert len(errors) == 0, f"Signal outside (0, 1]: {errors}"


# ---------------------------------------------------------------------------
# Test 4: Monte Carlo validation at R=2um (fast-eigenmode fully valid)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gamma_lm_vs_monte_carlo():
    """MC validation of Gamma_lm path for rotating gradient waveform at R=2um.

    R=2um is chosen because alpha1*sigma = 57.6 >> 1, so the fast-eigenmode
    approximation is fully valid. Expected |E_gamma - E_mc| < 0.02 (2%).

    Setup:
    - Cylinder R=2um along z axis, D=1.7e-9 m^2/s
    - Gradient rotates x->y over sigma=40ms, G0=0.3 T/m
    - Cylinder axis along z: gradient stays in xy-plane (purely perpendicular)
    - N_walkers=200000, seed=42

    Also reports R=5um results (expected |E_gamma - E_mc| ~0.03) for reference.
    """
    try:
        from dmipy_sim import simulate, Cylinder
        from dmipy_sim.waveforms import Waveform
        import jax.numpy as jnp
    except ImportError as e:
        pytest.skip(f"dmipy-sim not available: {e}")

    N_WALKERS = 200000
    SEED = 42
    G0_2um = 0.3  # T/m — larger gradient to get meaningful attenuation at R=2um

    def run_comparison(R_use, G0_use, label):
        n_t = N_T
        sigma = SIGMA
        dt = sigma / (n_t - 1)
        t = np.linspace(0.0, sigma, n_t)
        angle = (np.pi / 2.0) * t / sigma
        Gx = G0_use * np.cos(angle)
        Gy = G0_use * np.sin(angle)
        G_vec = np.stack([Gx, Gy, np.zeros(n_t)], axis=-1).astype(np.float32)
        G_arr = G_vec[None, :, :]

        wf = Waveform(G=jnp.array(G_arr), dt=float(dt), echo_idx=n_t - 1)
        E_mc_raw = simulate(
            n_walkers=N_WALKERS, diffusivity=D, waveform=wf,
            geometry=Cylinder(radius=R_use, orientation=[0.0, 0.0, 1.0]),
            seed=SEED,
        )
        E_mc = float(E_mc_raw[0])

        G_perp_t = np.sqrt(Gx**2 + Gy**2)
        E_iir = float(_ogse_numerical_cylinder_signal(
            G_perp_t, dt, D, R_use, _ROOTS))

        grad_dir = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]])
        # rotating x->y gradient is intentionally not moment-nulled (gamma_lm only
        # needs int|G|^2, not refocusing)
        scheme = AcquisitionScheme.from_waveform(G_arr, dt, grad_dir,
                                                 allow_unrefocused=True)
        C_geom = _compute_C_geom(R_use, D, _ROOTS)
        gamma_lm = scheme.gamma_lm(l_max=4)
        mu_z = np.array([0.0, 0.0, 1.0])
        bvals = scheme.bvalues
        b_par = bvals * np.dot(scheme.gradient_directions, mu_z) ** 2
        E_gamma = float(_cylinder_signal_from_gamma_lm(
            gamma_lm, b_par, C_geom, mu_z, lambda_par=D
        )[0])

        alpha1_sigma = ((_ROOTS[0] / R_use) ** 2) * D * sigma
        diff_gm = abs(E_gamma - E_mc)
        diff_im = abs(E_iir - E_mc)
        diff_gi = abs(E_gamma - E_iir)

        print(f"\n  {label} (R={R_use*1e6:.0f}um, G={G0_use} T/m):")
        print(f"    alpha1*sigma = {alpha1_sigma:.1f}")
        print(f"    bvalue = {float(bvals[0]):.3e} s/m^2")
        print(f"    E_mc    = {E_mc:.6f}")
        print(f"    E_iir   = {E_iir:.6f}")
        print(f"    E_gamma = {E_gamma:.6f}")
        print(f"    |E_gamma - E_mc|  = {diff_gm:.6f}")
        print(f"    |E_iir  - E_mc|   = {diff_im:.6f}")
        print(f"    |E_gamma - E_iir| = {diff_gi:.6f}")

        return E_mc, E_iir, E_gamma

    print("\nTest 4: MC validation — Gamma_lm path")

    # Primary: R=2um (fast-eigenmode fully valid)
    E_mc_2, E_iir_2, E_gamma_2 = run_comparison(R_small, G0_2um, "PRIMARY R=2um")

    # Reference: R=5um (fast-eigenmode marginal, documented for completeness)
    E_mc_5, E_iir_5, E_gamma_5 = run_comparison(R, G0, "REFERENCE R=5um")

    # Primary assertion: R=2um within 2% of MC
    npt.assert_allclose(
        E_gamma_2, E_mc_2, atol=0.02,
        err_msg=(
            f"R=2um: Gamma_lm vs MC failed. "
            f"E_gamma={E_gamma_2:.4f}, E_mc={E_mc_2:.4f}, "
            f"diff={abs(E_gamma_2 - E_mc_2):.4f} > 0.02"
        )
    )

    # IIR vs MC: should be < 2% (exact GPA vs MC)
    npt.assert_allclose(
        E_iir_2, E_mc_2, atol=0.02,
        err_msg=f"R=2um: IIR vs MC failed. diff={abs(E_iir_2 - E_mc_2):.4f}"
    )

    # R=5um: document that E_gamma differs from MC by ~3% (marginal fast-eigenmode)
    # This is NOT a hard assertion — it documents the known approximation limit.
    diff_5 = abs(E_gamma_5 - E_mc_5)
    print(f"\n  R=5um summary: |E_gamma - E_mc| = {diff_5:.4f} "
          f"({'within' if diff_5 < 0.05 else 'exceeds'} 5% tolerance)")
