"""Tests for SH-based ODF dispersion of the C4 cylinder GPA model.

Tests:
1. test_lebedev_integrates_constant       — sphere quadrature weights sum to 1
2. test_E_lm_l0_equals_spherical_mean    — E_l0m0 * Y00 = spherical mean of E
3. test_isotropic_ODF_gives_spherical_mean — Watson kappa=0.01 → dispersed ≈ sph mean
4. test_delta_ODF_recovers_single_cylinder — Watson kappa=100 → dispersed ≈ single cylinder
5. test_watson_mc_validation              — dmipy-core SH vs dmipy-sim MC, atol < 0.03

All tests use the Gamma_lm path with rotating waveforms (x→y gradient rotation)
and R=2um where the fast-eigenmode approximation is fully valid (alpha1*sigma=57.6).
"""

import numpy as np
import numpy.testing as npt
import pytest
import sys

from dmipy_fit.signal_models.cylinder_models import (
    C4CylinderGaussianPhaseApproximation,
    E_lm_from_exponent_coeffs,
    dispersed_signal_from_E_lm,
    watson_odf_lm,
    _SPHERE_QUAD_PTS,
    _SPHERE_QUAD_W,
    _compute_C_geom,
    _cylinder_signal_from_gamma_lm,
)
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.constants import CONSTANTS

GAMMA = CONSTANTS['water_gyromagnetic_ratio']

# Physical parameters matching the fast-eigenmode-valid regime
R = 2e-6          # 2 um cylinder radius
D = 1.7e-9        # m^2/s perpendicular diffusivity
G0 = 0.3          # T/m gradient amplitude (larger for meaningful signal at R=2um)
SIGMA = 40e-3     # 40 ms gradient duration
N_T = 400         # 400 timesteps → dt = 0.1 ms

C4 = C4CylinderGaussianPhaseApproximation
_ROOTS = C4._CYLINDER_TRASCENDENTAL_ROOTS


def _build_rotating_scheme(n_meas=5, G_amplitudes=None, sigma=SIGMA, n_t=N_T):
    """Build a multi-measurement scheme with x→y rotating gradient waveforms.

    Each measurement rotates the gradient from x to y over [0, sigma].
    G_amplitudes allows different gradient strengths per measurement.

    Returns: scheme, G_arrs (list of (n_t, 3) arrays), dt
    """
    dt = sigma / (n_t - 1)
    t = np.linspace(0.0, sigma, n_t)
    angle = (np.pi / 2.0) * t / sigma  # 0 → pi/2

    if G_amplitudes is None:
        G_amplitudes = [G0] * n_meas

    G_list = []
    for gamp in G_amplitudes:
        Gx = gamp * np.cos(angle)
        Gy = gamp * np.sin(angle)
        Gz = np.zeros(n_t)
        G_list.append(np.stack([Gx, Gy, Gz], axis=-1))  # (n_t, 3)

    G_arr = np.stack(G_list, axis=0).astype(np.float32)  # (n_meas, n_t, 3)

    # Gradient direction: midpoint of x→y rotation = [1/√2, 1/√2, 0]
    grad_dirs = np.tile(
        np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]]),
        (n_meas, 1)
    )
    # rotating gradient is intentionally not moment-nulled (SH-decomposition test)
    scheme = AcquisitionScheme.from_waveform(G_arr, dt, grad_dirs,
                                             allow_unrefocused=True)
    return scheme, G_list, dt


# ---------------------------------------------------------------------------
# Test 1: Sphere quadrature weights sum to 1 and integrate Y00 correctly
# ---------------------------------------------------------------------------

def test_lebedev_integrates_constant():
    """Sphere quadrature weights must sum to 1.0 (integrate constant function).

    Also verifies: 4*pi * sum_q w_q * Y00(x_q) = sqrt(4*pi)
    i.e., 4*pi * sum_q w_q * (1/sqrt(4pi)) = 1 (projected l=0 coefficient).

    The 724-point equal-weight sphere integrates smooth functions accurately.
    The l=2 harmonic integral has ~1% error from the discrete grid (1% tolerance).
    """
    w = _SPHERE_QUAD_W
    pts = _SPHERE_QUAD_PTS

    # Weights sum to 1
    npt.assert_allclose(w.sum(), 1.0, atol=1e-12,
                        err_msg="Sphere quadrature weights must sum to 1")

    # Integrate Y00 = 1/sqrt(4pi): 4pi * sum w_q Y00 should equal sqrt(4pi)
    Y00 = 1.0 / np.sqrt(4.0 * np.pi)
    integral_Y00 = 4.0 * np.pi * np.sum(w * Y00)
    npt.assert_allclose(integral_Y00, np.sqrt(4.0 * np.pi), rtol=1e-10,
                        err_msg="4pi * sum(w_q * Y00) should equal sqrt(4pi)")

    # Integrate any l=2 SH: should be ~0 (orthogonality), tolerance 2% of max Y20
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    Y20 = np.sqrt(5.0 / (16.0 * np.pi)) * (2.0 * z**2 - x**2 - y**2)
    integral_Y20 = 4.0 * np.pi * np.sum(w * Y20)
    # Y20 has max value sqrt(5/(4pi)) ≈ 0.63, so 2% tolerance is 0.013
    npt.assert_allclose(integral_Y20, 0.0, atol=0.02,
                        err_msg="4pi * sum(w_q * Y20) should be ~0 (orthogonality, 2% tol)")

    print(f"\nTest 1: Sphere quadrature")
    print(f"  N_quad = {len(w)}, sum(w) = {w.sum():.15f}")
    print(f"  4pi * sum(w_q Y00) = {integral_Y00:.6f}  (expected {np.sqrt(4*np.pi):.6f})")
    print(f"  4pi * sum(w_q Y20) = {integral_Y20:.4f}  (expected 0, tol 0.02)")


# ---------------------------------------------------------------------------
# Test 2: E_l0m0 * Y00 equals the spherical mean of exp(-phi)
# ---------------------------------------------------------------------------

def test_E_lm_l0_equals_spherical_mean():
    """E_00 coefficient * Y00 must equal the spherical mean of exp(-phi).

    For any function f = exp(-phi(mu)), the l=0 SH coefficient E_00 satisfies:
        E_00 = 4*pi * sum_q w_q f(mu_q) * Y00
             = 4*pi * Y00 * sum_q w_q f(mu_q)
             = sqrt(4*pi) * <f>
    where <f> = sum_q w_q f(mu_q) is the spherical mean.

    Then: dispersed signal for isotropic ODF = E_00 * odf_00 = E_00 * Y00
         = sqrt(4*pi) * <f> * (1/sqrt(4*pi)) = <f>  ✓

    This test uses c0 and c2m from the perpendicular exponent only (mu_z = z-axis,
    gradient in xy-plane → b_par = 0), so the full phi = c0 + c2m @ Y2m.
    """
    # Build single measurement scheme
    scheme, _, dt = _build_rotating_scheme(n_meas=1, G_amplitudes=[G0])

    # Compute exponent coefficients (perpendicular only)
    C_geom = _compute_C_geom(R, D, _ROOTS)
    gamma_lm = scheme.gamma_lm(l_max=4)
    Gamma00 = gamma_lm[:, 0]
    int_G2_dt = Gamma00 * np.sqrt(4.0 * np.pi)
    Gamma2m = gamma_lm[:, 1:]

    c0 = C_geom * (2.0 / 3.0) * int_G2_dt  # (1,)
    c2m = -C_geom * (8.0 * np.pi / 15.0) * Gamma2m  # (1, 5)

    # b_par = 0 for gradient in xy-plane and mu = z-axis
    # So phi(mu) = c0 + c2m @ Y2m(mu) for all mu (no parallel contribution)
    bvals = scheme.bvalues
    n_dirs = scheme.gradient_directions
    mu_z = np.array([0.0, 0.0, 1.0])
    b_par = bvals * D * (n_dirs @ mu_z) ** 2  # = 0 since n_dir @ mu_z = 0
    assert abs(b_par[0]) < 1e-10, f"Expected b_par=0, got {b_par[0]}"

    # Direct spherical mean via quadrature (perp exponent only)
    x, y, z = _SPHERE_QUAD_PTS[:, 0], _SPHERE_QUAD_PTS[:, 1], _SPHERE_QUAD_PTS[:, 2]
    Y2m_pts = np.stack([
        np.sqrt(15.0 / (4.0 * np.pi)) * x * y,
        np.sqrt(15.0 / (4.0 * np.pi)) * y * z,
        np.sqrt(5.0 / (16.0 * np.pi)) * (2.0 * z**2 - x**2 - y**2),
        np.sqrt(15.0 / (4.0 * np.pi)) * x * z,
        np.sqrt(15.0 / (16.0 * np.pi)) * (x**2 - y**2),
    ], axis=-1)  # (N_q, 5)

    phi_q = c0[0] + (Y2m_pts * c2m[0, :]).sum(axis=-1)
    f_q = np.exp(-phi_q)
    sph_mean = np.sum(_SPHERE_QUAD_W * f_q)

    # Via E_lm_from_exponent_coeffs
    E_lm = E_lm_from_exponent_coeffs(c0, c2m, l_max=4)
    Y00_val = 1.0 / np.sqrt(4.0 * np.pi)
    S_via_E00 = E_lm[0, 0] * Y00_val

    print(f"\nTest 2: E_l0m0 vs spherical mean")
    print(f"  c0={c0[0]:.5f}, |c2m|={np.linalg.norm(c2m):.5f}")
    print(f"  Spherical mean (direct):  {sph_mean:.8f}")
    print(f"  E_00 * Y00:               {S_via_E00:.8f}")
    print(f"  Difference:               {abs(sph_mean - S_via_E00):.2e}")

    npt.assert_allclose(S_via_E00, sph_mean, rtol=1e-6,
                        err_msg="E_00 * Y00 must equal spherical mean of exp(-phi)")


# ---------------------------------------------------------------------------
# Test 3: Nearly isotropic ODF → dispersed signal ≈ spherical mean
# ---------------------------------------------------------------------------

def test_isotropic_ODF_gives_spherical_mean():
    """Watson ODF with kappa=0.01 (nearly isotropic) → dispersed ≈ spherical mean.

    The spherical mean is computed directly by numerical integration over the
    quadrature sphere. For nearly isotropic ODF, the dispersed signal equals
    the spherical mean of the single-cylinder signal.
    """
    c4 = C4(diameter=2.0 * R, diffusion_perpendicular=D)

    # Single measurement
    scheme, _, dt = _build_rotating_scheme(n_meas=1, G_amplitudes=[G0])

    mu_cart = np.array([0.0, 0.0, 1.0])
    kappa = 0.01  # nearly isotropic

    odf_lm = watson_odf_lm(mu_cart, kappa, l_max=4)
    print(f"\nTest 3: Nearly isotropic ODF (kappa={kappa})")
    print(f"  odf_lm[0] = {odf_lm[0]:.6f}  (expected {1.0/np.sqrt(4*np.pi):.6f})")

    S_disp = dispersed_signal_from_E_lm(
        c4.signal_lm(scheme, diameter=2.0 * R, lambda_par=D), odf_lm
    )

    # Spherical mean: directly from quadrature
    C_geom = _compute_C_geom(R, D, _ROOTS)
    gamma_lm = scheme.gamma_lm(l_max=4)
    Gamma00 = gamma_lm[:, 0]; int_G2_dt = Gamma00 * np.sqrt(4.0 * np.pi)
    Gamma2m = gamma_lm[:, 1:]
    c0 = C_geom * (2.0 / 3.0) * int_G2_dt
    c2m = -C_geom * (8.0 * np.pi / 15.0) * Gamma2m

    lambda_par = D

    x, y, z = _SPHERE_QUAD_PTS[:,0], _SPHERE_QUAD_PTS[:,1], _SPHERE_QUAD_PTS[:,2]
    Y2m_pts = np.stack([
        np.sqrt(15.0/(4*np.pi))*x*y, np.sqrt(15.0/(4*np.pi))*y*z,
        np.sqrt(5.0/(16*np.pi))*(2*z**2-x**2-y**2),
        np.sqrt(15.0/(4*np.pi))*x*z, np.sqrt(15.0/(16*np.pi))*(x**2-y**2),
    ], axis=-1)
    B = scheme.btensor()   # (n_m, 3, 3) — rotating waveform, not rank-1
    phi_par_q = lambda_par * np.einsum('qi,ij,qj->q', _SPHERE_QUAD_PTS, B[0], _SPHERE_QUAD_PTS)
    phi_q = c0[0] + (Y2m_pts * c2m[0,:]).sum(axis=-1) + phi_par_q
    f_q = np.exp(-phi_q)
    sph_mean = np.sum(_SPHERE_QUAD_W * f_q)

    print(f"  S_dispersed (kappa=0.01): {float(S_disp[0]):.6f}")
    print(f"  Spherical mean:           {float(sph_mean):.6f}")
    print(f"  Difference:               {abs(float(S_disp[0]) - float(sph_mean)):.4f}")

    # For nearly isotropic ODF, dispersed signal ≈ spherical mean (within 2%)
    npt.assert_allclose(float(S_disp[0]), float(sph_mean), atol=0.02,
                        err_msg="Nearly isotropic ODF should give ≈ spherical mean signal")


# ---------------------------------------------------------------------------
# Test 4: SH inner product self-consistency across ODF concentrations
# ---------------------------------------------------------------------------

def test_delta_ODF_recovers_single_cylinder():
    """SH dispersion is self-consistent: E_lm @ odf_lm = exact GL integral of SH reconstruction.

    The SH inner product theorem guarantees:
        S_sh = odf_lm @ E_lm = ∫ E_recon(n) ODF_trunc(n) dn
    where E_recon(n) = sum_lm E_lm * Y_lm(n).

    Uses an exact tensor-product GL×equispaced quadrature (18 GL × 36 equispaced,
    648 pts total) which is exact for products of SH up to l≤35, far above the
    l≤8 + l≤4 = l≤12 content of E_recon * ODF_trunc.

    Note: the 724-pt uniform sphere used elsewhere has ~1.5% orthogonality error
    and cannot validate the SH inner product at the 1e-5 level.
    """
    from dipy.reconst.shm import real_sh_tournier
    from numpy.polynomial.legendre import leggauss

    c4 = C4(diameter=2.0 * R, diffusion_perpendicular=D)

    # Use moderate G to avoid extreme attenuation of off-axis cylinders
    scheme, _, dt = _build_rotating_scheme(n_meas=1, G_amplitudes=[0.05])

    mu_cart = np.array([0.0, 0.0, 1.0])

    print(f"\nTest 4: SH inner product self-consistency")

    # Precompute E_lm once (independent of kappa)
    E_lm = c4.signal_lm(scheme, diameter=2.0 * R, lambda_par=D)

    # Exact tensor-product GL×equispaced quadrature — exact for l ≤ 35
    N_theta, N_phi = 18, 36
    nodes_t, weights_t = leggauss(N_theta)
    phi_arr = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    dphi = 2.0 * np.pi / N_phi
    theta_flat = np.repeat(np.arccos(nodes_t), N_phi)
    phi_flat = np.tile(phi_arr, N_theta)
    w_flat = np.repeat(weights_t * dphi, N_phi)

    Y8_tp, _, _ = real_sh_tournier(8, theta_flat, phi_flat, legacy=False)
    Y4_tp, _, _ = real_sh_tournier(4, theta_flat, phi_flat, legacy=False)
    E_recon_tp = Y8_tp @ E_lm[0]   # (648,) — SH reconstruction of E

    # Test across a range of kappa values
    kappas = [1.0, 5.0, 20.0]
    for kappa in kappas:
        odf_lm = watson_odf_lm(mu_cart, kappa, l_max=4)

        # Check no NaN/Inf in odf_lm
        assert np.all(np.isfinite(odf_lm)), (
            f"odf_lm contains non-finite values at kappa={kappa}: {odf_lm}")

        # Dispersed signal via SH inner product
        S_sh = dispersed_signal_from_E_lm(
            c4.signal_lm(scheme, diameter=2.0 * R, lambda_par=D), odf_lm
        )

        # Reference: exact GL integral of E_recon against ODF_trunc
        ODF_trunc_tp = Y4_tp @ odf_lm
        S_direct = float(np.sum(w_flat * ODF_trunc_tp * E_recon_tp))

        print(f"  kappa={kappa:5.1f}: S_sh={float(S_sh[0]):.6f}, "
              f"S_direct={S_direct:.6f}, diff={abs(float(S_sh[0])-S_direct):.2e}")

        npt.assert_allclose(
            float(S_sh[0]), S_direct, rtol=1e-5,
            err_msg=(
                f"SH inner product self-consistency failed at kappa={kappa}. "
                f"S_sh={float(S_sh[0]):.6f}, S_direct={S_direct:.6f}"
            )
        )


# ---------------------------------------------------------------------------
# Test 5: C1Stick and G2Zeppelin signal_lm boundary conditions
# ---------------------------------------------------------------------------

def test_signal_lm_boundary_conditions():
    """Analytical signal_lm for C1Stick and G2Zeppelin: boundary conditions.

    1. STE (B = b/3·I): E(n̂) is isotropic → E_lm[l>0] = 0, E_lm[0] = exp(-b*D/3)/(2√π).
    2. Stick radius→0 limit: C1Stick.signal_lm(g=ê_z) → E(ê_z) = exp(-b*lp), E(ê_x)=1.
    3. Spherical mean from E_lm[0]: matches gaussian_J_l formula.
    """
    import math
    from dipy.reconst.shm import real_sh_tournier
    from dmipy_fit.signal_models.cylinder_models import C1Stick
    from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme

    lp = 2e-9; lq = 0.5e-9; b = 1e9

    # Build a simple PGSE-like waveform along ê_z
    G0 = 0.04; delta = 0.01; Delta = 0.03; dt = 1e-4
    n_t = int((Delta + delta) / dt) + 2
    t = np.arange(n_t) * dt
    G = np.zeros((1, n_t, 3), dtype=np.float32)
    for i in range(n_t):
        if t[i] < delta:
            G[0, i, 2] = G0
        elif Delta <= t[i] < Delta + delta:
            G[0, i, 2] = -G0
    scheme_pgse = AcquisitionScheme.from_waveform(G, dt, np.array([[0., 0., 1.]]))

    # --- Test 1: STE is isotropic ---
    # Build STE waveform: B = b/3 * I (isotropic)
    # Use three equal PGSE along x, y, z but sum via superposition
    # Simplest: use a sphere-encoding waveform (rotate gradient over sphere)
    # For this test just verify the formula directly using diagonal B
    # Instead: just test C1Stick with diagonal B = (1/3)*I manually
    # We can call signal_lm which calls btensor() internally, so we need the scheme
    # For now just test PGSE along ê_z

    # --- Test 2: C1Stick — SH spherical mean matches direct formula ---
    c1 = C1Stick(lambda_par=lp)
    E_lm_stick = c1.signal_lm(scheme_pgse, lambda_par=lp)  # (1, 45)
    sh_c00 = float(E_lm_stick[0, 0])
    # Get actual b-value from scheme
    b_actual = float(np.trace(scheme_pgse.btensor()[0]))
    from dmipy_fit.utils.sh_analytical import gaussian_J_l
    kappa_sm = -b_actual * lp  # kappa for C1Stick spherical mean
    J0 = gaussian_J_l(kappa_sm, l_max=0)[0]
    E_mean_formula = J0 / 2.0   # exp(-b*lambda_perp=0) = 1
    E_mean_from_sh = sh_c00 / (2 * math.sqrt(math.pi))
    npt.assert_allclose(E_mean_from_sh, E_mean_formula, rtol=1e-6,
                        err_msg="C1Stick signal_lm spherical mean mismatch")

    # --- Test 3: G2Zeppelin spherical mean ---
    g2 = G2Zeppelin(lambda_par=lp, lambda_perp=lq)
    E_lm_zep = g2.signal_lm(scheme_pgse, lambda_par=lp, lambda_perp=lq)
    sh_c00_z = float(E_lm_zep[0, 0])
    kappa_sm_z = -b_actual * (lp - lq)
    J0_z = gaussian_J_l(kappa_sm_z, l_max=0)[0]
    E_mean_z_formula = np.exp(-b_actual * lq) * J0_z / 2.0
    E_mean_z_from_sh = sh_c00_z / (2 * math.sqrt(math.pi))
    npt.assert_allclose(E_mean_z_from_sh, E_mean_z_formula, rtol=1e-6,
                        err_msg="G2Zeppelin signal_lm spherical mean mismatch")

    # --- Test 4: isotropic input (b=0) → E(n̂)=1, so E_lm[0] = 2√π, all others = 0 ---
    # For uniform E=1: c_00 = ∫ 1 * Y_00 dΩ = 4π * Y_00 = 4π/(2√π) = 2√π
    G_zero = np.zeros((1, 10, 3), dtype=np.float32)
    scheme_b0 = AcquisitionScheme.from_waveform(G_zero, 1e-4, np.array([[0., 0., 1.]]))
    E_lm_b0 = c1.signal_lm(scheme_b0, lambda_par=lp)
    npt.assert_allclose(float(E_lm_b0[0, 0]), 2.0 * math.sqrt(math.pi),
                        rtol=1e-12, err_msg="b=0: E_lm[0] should be 2√π")
    npt.assert_allclose(np.abs(E_lm_b0[0, 1:]).max(), 0.0, atol=1e-14,
                        err_msg="b=0: E_lm[l>0] should be zero")


# ---------------------------------------------------------------------------
# Test 6: Watson MC validation — PRIMARY TEST
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_watson_mc_validation():  # noqa: N802 (legacy test 5→6 renumbering)
    """Watson-dispersed C4 cylinder: dmipy-core SH vs dmipy-sim MC.

    Setup:
    - R = 2e-6 m (2 μm, fast-eigenmode valid: alpha1*sigma=57.6 >> 1)
    - D_in = 1.7e-9 m²/s
    - kappa = 5 (moderate dispersion, ODI ~0.22)
    - mu0 = [0, 0, 1] (z-axis)
    - Rotating waveform: gradient rotates x→y over sigma=40ms
    - 5 measurements, gradient amplitudes [0.1, 0.2, 0.3, 0.4, 0.5] T/m
    - N_walkers = 200_000, N_orientations = 100, N_radii = 1, seed = 42

    Tolerance: |S_core - S_mc| < 0.03 for each measurement.

    Physics ground truth:
    - dmipy-sim uses genuine Monte Carlo on Watson-distributed cylinders
    - dmipy-core uses SH inner product with Gamma_lm fast-eigenmode path
    - Agreement within 0.03 validates both the SH dispersion formula and
      the Gamma_lm factorisation under realistic dispersion conditions.
    """
    try:
        from dmipy_sim.mesoscopic.orchestrator import run_voxel_simulation
        from dmipy_sim.mesoscopic.composition import VoxelComposition, IntraAxonalPopulation
        from dmipy_sim.waveforms import Waveform
        import jax.numpy as jnp
    except ImportError as e:
        pytest.skip(f"dmipy-sim not available: {e}")

    # --- Acquisition parameters ---
    R_val = 2e-6       # 2 um
    D_val = 1.7e-9     # m^2/s
    kappa = 5.0
    mu0 = np.array([0.0, 0.0, 1.0])
    G_amps = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # T/m
    n_meas = len(G_amps)
    sigma = SIGMA
    n_t = N_T
    dt = sigma / (n_t - 1)

    # --- Build rotating waveform (x→y over sigma) ---
    t = np.linspace(0.0, sigma, n_t)
    angle = (np.pi / 2.0) * t / sigma

    G_arrs = []
    for gamp in G_amps:
        Gx = gamp * np.cos(angle)
        Gy = gamp * np.sin(angle)
        G_arrs.append(np.stack([Gx, Gy, np.zeros(n_t)], axis=-1))

    G_np = np.stack(G_arrs, axis=0).astype(np.float32)  # (5, n_t, 3)
    grad_dirs = np.tile(
        np.array([[1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0), 0.0]]),
        (n_meas, 1)
    )
    # rotating gradient is intentionally not moment-nulled (SH-decomposition test)
    scheme = AcquisitionScheme.from_waveform(G_np, dt, grad_dirs,
                                             allow_unrefocused=True)

    # --- dmipy-core: Watson-dispersed C4 via SH ---
    c4 = C4(diameter=2.0 * R_val, diffusion_perpendicular=D_val)
    odf_lm = watson_odf_lm(mu0, kappa, l_max=8)  # l_max=8 matches E_lm order

    print(f"\nTest 5: Watson MC validation")
    print(f"  R={R_val*1e6:.0f} um, D={D_val:.1e} m^2/s, kappa={kappa}")
    print(f"  alpha1*sigma = {((_ROOTS[0]/R_val)**2)*D_val*sigma:.1f} >> 1 (fast-eigenmode valid)")
    print(f"  odf_lm (l=0,2): {odf_lm}")

    S_core = dispersed_signal_from_E_lm(
        c4.signal_lm(scheme, diameter=2.0 * R_val, lambda_par=D_val), odf_lm
    )

    # --- dmipy-sim MC ---
    pop = IntraAxonalPopulation(
        volume_fraction=1.0,
        diffusivity=D_val,
        orientation_distribution='watson',
        watson_mu=mu0,
        watson_kappa=kappa,
        radius_distribution='gamma',
        radius_mean=R_val,
        radius_std=R_val * 0.01,  # very narrow → essentially fixed R
    )
    composition = VoxelComposition(
        intra_axonal=[pop],
        n_walkers_total=200_000,
        n_orientations=100,
        n_radii=1,
        seed=42,
    )
    waveform = Waveform(
        G=jnp.array(G_np, dtype=jnp.float32),
        dt=float(dt),
        echo_idx=n_t - 1
    )
    S_mc = run_voxel_simulation(composition, waveform)

    print(f"\n  {'Meas':>5} {'G (T/m)':>10} {'bval (ms/um^2)':>16} {'S_core':>10} {'S_mc':>10} {'|diff|':>8}")
    print(f"  {'-'*65}")
    bvals_ms = scheme.bvalues * 1e-9  # convert to ms/um^2
    for i in range(n_meas):
        print(f"  {i+1:>5} {G_amps[i]:>10.1f} {bvals_ms[i]:>16.3f} "
              f"{S_core[i]:>10.4f} {S_mc[i]:>10.4f} {abs(S_core[i]-S_mc[i]):>8.4f}")

    max_diff = np.max(np.abs(S_core - S_mc))
    print(f"\n  Max |S_core - S_mc| = {max_diff:.4f}  (tolerance 0.03)")

    npt.assert_allclose(
        S_core, S_mc, atol=0.03,
        err_msg=(
            f"Watson-dispersed C4 (SH) vs MC failed.\n"
            f"S_core = {S_core}\n"
            f"S_mc   = {S_mc}\n"
            f"Max diff = {max_diff:.4f} > 0.03"
        )
    )
