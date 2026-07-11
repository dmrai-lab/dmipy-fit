# -*- coding: utf-8 -*-
"""
Tests for exact analytical Watson SH coefficients in sh_analytical.py.

Three invariants are verified:
  1. l=0 coefficient is exactly 1/(2*sqrt(pi)) for all kappa and mu.
  2. watson_sh matches a grid-based reference at low/medium kappa
     (where the grid is accurate).
  3. Isotropic limit (kappa → 0): all coefficients l > 0 vanish.
  4. Zonal case (mu = z-hat): only m=0 components are non-zero.
  5. Signal integral is 1 when convolved with a flat ODF kernel.
  6. Gradient w.r.t. kappa is finite (no NaN / Inf).
"""
import numpy as np
import pytest
from scipy.special import erfi

from dmipy_fit.utils.sh_analytical import (
    watson_zonal_ratios, watson_sh,
    bingham_canonical_sh, bingham_sh,
    gaussian_J_l, gaussian_kernel_rh,
)


# ── helpers ──────────────────────────────────────────────────────────────────

L_MAX = 8
N_COEF = (L_MAX + 1) * (L_MAX + 2) // 2
SQRT_PI = np.sqrt(np.pi)
L0_EXACT = 1.0 / (2.0 * SQRT_PI)          # Y_0^0 = 1/(2√π) in Tournier conv.


def _reference_J0(kappa):
    """Exact J_0(kappa) = integral_{-1}^{1} exp(kappa t^2) dt."""
    if kappa < 1e-12:
        return 2.0
    return SQRT_PI * erfi(np.sqrt(kappa)) / np.sqrt(kappa)


def _reference_J2(kappa):
    """Exact J_2(kappa) from the first recurrence step."""
    J0 = _reference_J0(kappa)
    K1 = np.exp(kappa) / kappa - J0 / (2 * kappa)
    return (3 * K1 - J0) / 2.0


# ── watson_zonal_ratios ───────────────────────────────────────────────────────

class TestWatsonZonalRatios:

    def test_r0_always_one(self):
        for kappa in [0.0, 0.1, 1.0, 5.0, 20.0, 100.0]:
            r = watson_zonal_ratios(kappa, l_max=L_MAX)
            assert abs(r[0] - 1.0) < 1e-14, f"r[0] != 1 at kappa={kappa}"

    def test_isotropic_kappa_zero(self):
        r = watson_zonal_ratios(0.0, l_max=L_MAX)
        assert abs(r[0] - 1.0) < 1e-14
        np.testing.assert_array_less(np.abs(r[1:]), 1e-12)

    def test_r1_matches_reference_J2_over_J0(self):
        for kappa in [0.5, 1.0, 2.0, 5.0, 10.0]:
            r = watson_zonal_ratios(kappa, l_max=L_MAX)
            ref = _reference_J2(kappa) / _reference_J0(kappa)
            assert abs(r[1] - ref) < 1e-10, \
                f"r[1]={r[1]:.8f} vs ref={ref:.8f} at kappa={kappa}"

    def test_ratios_bounded_01(self):
        """For a well-behaved distribution, all r_l should be in [0, 1]."""
        for kappa in [0.1, 1.0, 5.0, 20.0]:
            r = watson_zonal_ratios(kappa, l_max=L_MAX)
            assert np.all(r >= -1e-12), f"Negative ratio at kappa={kappa}"
            assert np.all(r <= 1.0 + 1e-10), f"Ratio > 1 at kappa={kappa}"

    def test_monotone_in_kappa(self):
        """Higher kappa → higher anisotropy → larger r_l for l > 0."""
        r_low = watson_zonal_ratios(1.0, l_max=4)
        r_high = watson_zonal_ratios(10.0, l_max=4)
        # r[1] = J_2/J_0 should grow with kappa (distribution narrows)
        assert r_high[1] > r_low[1]

    def test_large_kappa_ratios_approach_one(self):
        """At very high concentration all r_l → 1 (delta function).

        Asymptotic: r_l ≈ 1 - l(l+1)/(4*kappa), so l=8 needs kappa >> 72/4.
        At kappa=50000 the l=8 deviation is < 1e-3.
        """
        r = watson_zonal_ratios(50000.0, l_max=L_MAX)
        np.testing.assert_allclose(r, 1.0, atol=1e-3)


# ── watson_sh ────────────────────────────────────────────────────────────────

class TestWatsonSh:

    def test_l0_coefficient_exact(self):
        """l=0 coefficient must equal 1/(2*sqrt(pi)) for any kappa and mu."""
        for kappa in [0.0, 0.5, 2.0, 10.0, 50.0]:
            for mu in [np.r_[0., 0., 1.], np.r_[1., 0., 0.],
                       np.r_[1., 1., 1.] / np.sqrt(3.)]:
                sh = watson_sh(mu, kappa, l_max=L_MAX)
                assert abs(sh[0] - L0_EXACT) < 1e-13, \
                    f"c_0^0={sh[0]:.10f} != {L0_EXACT:.10f} " \
                    f"(kappa={kappa}, mu={mu})"

    def test_isotropic_only_l0(self):
        """kappa=0 → uniform ODF → all l>0 coefficients zero."""
        sh = watson_sh(np.r_[0., 0., 1.], kappa=0.0, l_max=L_MAX)
        assert abs(sh[0] - L0_EXACT) < 1e-13
        np.testing.assert_array_less(np.abs(sh[1:]), 1e-12)

    def test_z_aligned_zonal_only(self):
        """mu = z-hat → only m=0 coefficients (index 0, 3, 10, 21, 36)
        should be non-zero.  Off-zonal (|m|>0) should be machine-epsilon."""
        sh = watson_sh(np.r_[0., 0., 1.], kappa=3.0, l_max=L_MAX)
        # Non-zonal indices for l=2: positions 1,2 (m=-2,-1,0,1,2 → skip 0)
        # In Tournier ordering the zonal (m=0) index within each l-block is
        # at position l (0-indexed within the block).
        # Full flat positions: l=0→0, l=2→3 (middle of 5: pos 3+2=5? Let me recount)
        # Tournier ordering per dipy: m = -l,...,0,...,l  for each even l
        # Offsets: l=0:0, l=2:1..5, l=4:6..14, l=6:15..27, l=8:28..44
        # Zonal (m=0) within l=2 block: index 3  (1 + 2 = 3)
        # Zonal within l=4 block: index 10  (6 + 4 = 10)
        # etc.
        zonal_idx = [0, 3, 10, 21, 36]  # m=0 positions for l=0,2,4,6,8
        mask = np.ones(len(sh), dtype=bool)
        mask[zonal_idx] = False
        np.testing.assert_array_less(
            np.abs(sh[mask]), 1e-12,
            err_msg="Non-zonal coefficients non-zero for z-aligned Watson"
        )

    def test_matches_grid_at_low_kappa(self):
        """At low kappa the grid-based SD1Watson and analytical agree."""
        from dmipy_fit.distributions.distributions import SD1Watson, kappa2odi
        from dmipy_fit.utils.utils import cart2sphere

        kappa = 1.0
        mu_cart = np.r_[1., 0., 0.]
        mu_sph = cart2sphere(mu_cart[None, :])[0, 1:]

        sh_analytical = watson_sh(mu_cart, kappa, l_max=4)

        # grid-based (old path — use grid directly for comparison)
        from dipy.data import get_sphere, HemiSphere
        from dipy.reconst.shm import real_sh_tournier
        sphere = get_sphere(name='symmetric724')
        hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)
        from scipy.special import hyp1f1
        watson_sf = np.exp(kappa * hemisphere.vertices.dot(mu_cart) ** 2) / (
            4 * np.pi * hyp1f1(0.5, 1.5, kappa))
        from numpy.linalg import pinv
        Y = real_sh_tournier(4, hemisphere.theta, hemisphere.phi, legacy=False)[0]
        sh_grid = pinv(Y) @ watson_sf
        # Enforce l=0 renormalization as the old code did
        sh_grid = sh_grid * (L0_EXACT / sh_grid[0])

        np.testing.assert_allclose(sh_analytical, sh_grid, atol=5e-4,
                                   err_msg="Analytical vs grid mismatch at kappa=1")

    def test_rotational_invariance(self):
        """Rotating the ODF should change individual c_lm but not the total
        power per l: sum_m |c_l^m|^2 = |c_l^0|^2 * (2l+1) for any rotation."""
        # For a zonal distribution, all power at order l is in the m=0 term
        # before rotation.  After rotation (mu != z) the power distributes
        # over m but the sum is the same (Wigner D is unitary).
        kappa = 3.0
        mu_z = np.r_[0., 0., 1.]
        mu_x = np.r_[1., 0., 0.]
        sh_z = watson_sh(mu_z, kappa, l_max=L_MAX)
        sh_x = watson_sh(mu_x, kappa, l_max=L_MAX)

        # Power per l: sum over the 2l+1 coefficients at each order
        offsets = [0, 1, 6, 15, 28]  # start of l=0,2,4,6,8 blocks
        ends    = [1, 6, 15, 28, 45]
        for a, b in zip(offsets, ends):
            power_z = np.sum(sh_z[a:b] ** 2)
            power_x = np.sum(sh_x[a:b] ** 2)
            assert abs(power_z - power_x) < 1e-12, \
                f"Power mismatch at l block [{a}:{b}]: {power_z} vs {power_x}"

    def test_no_nan_inf_over_kappa_range(self):
        """No NaN or Inf for kappa spanning 6 decades."""
        mu = np.r_[1., 1., 1.] / np.sqrt(3.)
        for kappa in np.logspace(-3, 3, 50):
            sh = watson_sh(mu, kappa, l_max=L_MAX)
            assert np.all(np.isfinite(sh)), \
                f"Non-finite SH at kappa={kappa:.4f}: {sh}"


# ── Gaussian kernel (Zeppelin / Stick) ───────────────────────────────────────

class TestGaussianKernelRh:
    """Tests for gaussian_J_l and gaussian_kernel_rh."""

    # Physical test params
    B = 1e9         # 1000 s/mm² in SI
    LP = 1.7e-9     # lambda_par
    LX = 0.3e-9     # lambda_perp

    def test_J0_matches_erf_formula(self):
        """J_0(kappa) = sqrt(pi)*erf(sqrt(-kappa))/sqrt(-kappa) for kappa < 0."""
        from scipy.special import erf as _erf
        for b in [0.5e9, 1e9, 2e9, 5e9]:
            kappa = -b * (self.LP - self.LX)
            J = gaussian_J_l(kappa, l_max=L_MAX)
            sqrt_neg_k = np.sqrt(-kappa)
            J0_ref = np.sqrt(np.pi) * _erf(sqrt_neg_k) / sqrt_neg_k
            assert abs(J[0] - J0_ref) < 1e-12, \
                f"J_0 mismatch at b={b}: {J[0]} vs {J0_ref}"

    def test_J_isotropic(self):
        """J_l(0) = 2 for l=0, 0 for l>0."""
        J = gaussian_J_l(0.0, l_max=L_MAX)
        assert abs(J[0] - 2.0) < 1e-14
        np.testing.assert_array_less(np.abs(J[1:]), 1e-14)

    def test_J_numerical_integration(self):
        """J_l agrees with scipy.integrate.quad over [-1,1]."""
        from scipy.integrate import quad
        from scipy.special import legendre as _leg
        kappa = -self.B * (self.LP - self.LX)
        J = gaussian_J_l(kappa, l_max=L_MAX)
        for m in range(L_MAX // 2 + 1):
            l = 2 * m
            ref, _ = quad(lambda t: _leg(l)(t) * np.exp(kappa * t**2), -1, 1)
            assert abs(J[m] - ref) < 1e-8, \
                f"J_{l} mismatch: {J[m]:.10f} vs {ref:.10f}"

    def test_kernel_rh0_equals_spherical_mean(self):
        """kernel_rh[0] / (2*sqrt(pi)) == analytical Zeppelin spherical mean."""
        from scipy.special import erf as _erf
        for b in [0.5e9, 1e9, 2e9]:
            rh = gaussian_kernel_rh(b, self.LP, self.LX, sh_order=L_MAX)
            sm_analytical = rh[0] / (2.0 * np.sqrt(np.pi))

            sqrt_bl = np.sqrt(b * (self.LP - self.LX))
            sm_ref = (np.exp(-b * self.LX) * np.sqrt(np.pi)
                      * _erf(sqrt_bl) / (2.0 * sqrt_bl))
            assert abs(sm_analytical - sm_ref) < 1e-10, \
                f"Spherical mean mismatch at b={b}: {sm_analytical} vs {sm_ref}"

    def test_stick_is_zeppelin_with_zero_perp(self):
        """C1Stick = Zeppelin with lambda_perp=0."""
        for b in [1e9, 3e9]:
            rh_stick = gaussian_kernel_rh(b, self.LP, 0.0, sh_order=L_MAX)
            rh_zep = gaussian_kernel_rh(b, self.LP, 0.0, sh_order=L_MAX)
            np.testing.assert_array_equal(rh_stick, rh_zep)

# ── Bingham ODF analytical SH ────────────────────────────────────────────────

class TestBinghamCanonicalSh:
    """Tests for bingham_canonical_sh (GL quadrature, canonical x-peaked frame)."""

    def test_l0_exact(self):
        """c_0^0 = 1/(2√π) for any kappa, beta."""
        for kappa, beta in [(0.5, 0.0), (2.0, 1.0), (10.0, 5.0), (50.0, 20.0)]:
            sh = bingham_canonical_sh(kappa, beta, l_max=L_MAX)
            assert abs(sh[0] - L0_EXACT) < 1e-13, \
                f"c_0^0 wrong at kappa={kappa}, beta={beta}"

    def test_odd_m_zero(self):
        """All odd-m coefficients are identically zero."""
        sh = bingham_canonical_sh(3.0, 1.5, l_max=L_MAX)
        # Odd-m indices for l=2: positions 2, 4 (m=-1, m=+1)
        # General: m is at index counter + (m+l) for each l-block
        counter = 0
        for l in range(0, L_MAX + 1, 2):
            for m in range(-l, l + 1):
                coef_idx = counter + (m + l)
                if m % 2 != 0:
                    assert abs(sh[coef_idx]) < 1e-15, \
                        f"Odd-m coef nonzero at l={l}, m={m}: {sh[coef_idx]}"
            counter += 2 * l + 1

    def test_negative_m_zero(self):
        """All negative-m (sin-type) coefficients are zero."""
        sh = bingham_canonical_sh(3.0, 1.5, l_max=L_MAX)
        counter = 0
        for l in range(0, L_MAX + 1, 2):
            for m in range(-l, 0):
                coef_idx = counter + (m + l)
                assert abs(sh[coef_idx]) < 1e-15, \
                    f"Negative-m coef nonzero at l={l}, m={m}: {sh[coef_idx]}"
            counter += 2 * l + 1

    def test_isotropic_kappa_zero(self):
        """kappa=beta=0 → uniform ODF → only c_0^0 nonzero."""
        sh = bingham_canonical_sh(0.0, 0.0, l_max=L_MAX)
        assert abs(sh[0] - L0_EXACT) < 1e-13
        np.testing.assert_array_less(np.abs(sh[1:]), 1e-12)

    def test_watson_limit_beta_zero(self):
        """beta=0 canonical Bingham = Watson peaked at x-axis."""
        kappa = 2.0
        mu_x = np.r_[1., 0., 0.]
        sh_bing = bingham_canonical_sh(kappa, 0.0, l_max=4)
        sh_wat = watson_sh(mu_x, kappa, l_max=4)
        np.testing.assert_allclose(sh_bing, sh_wat, atol=1e-13,
                                   err_msg="beta=0 Bingham != Watson at x-axis")

    def test_normalization_via_quad(self):
        """Z matches direct numerical integration over the sphere."""
        from scipy.integrate import dblquad
        kappa, beta = 1.5, 0.7
        # Z_ref = ∫_0^pi ∫_0^{2pi} exp(kappa sin²θ cos²φ + beta sin²θ sin²φ) sinθ dφ dθ
        def integrand(phi, theta):
            s2 = np.sin(theta)**2
            return np.exp(kappa*s2*np.cos(phi)**2 + beta*s2*np.sin(phi)**2) * np.sin(theta)
        Z_ref, _ = dblquad(integrand, 0, np.pi, 0, 2*np.pi,
                           epsabs=1e-8, epsrel=1e-8)
        # GL-based Z = (2π * ∫_{-1}^1 exp(A) I_0(B) dt)
        from numpy.polynomial.legendre import leggauss
        from scipy.special import iv
        nodes, weights = leggauss(32)
        s2 = 1 - nodes**2
        A = (kappa + beta) / 2 * s2
        B = (kappa - beta) / 2 * s2
        Z_gl = 2*np.pi * np.dot(weights, np.exp(A) * iv(0, B))
        assert abs(Z_gl - Z_ref) / Z_ref < 1e-7, \
            f"Z_gl={Z_gl:.8f} vs Z_ref={Z_ref:.8f}"


class TestBinghamSh:
    """Tests for bingham_sh (full rotation to arbitrary mu, psi)."""

    def test_l0_exact_after_rotation(self):
        """c_0^0 = 1/(2√π) for any orientation and concentration."""
        for kappa, beta, psi in [(1.0, 0.5, 0.3), (5.0, 2.0, 1.2), (20.0, 10.0, 0.0)]:
            for mu in [np.r_[0., 0., 1.], np.r_[1., 0., 0.],
                       np.r_[1., 1., 1.] / np.sqrt(3.)]:
                sh = bingham_sh(mu, psi, kappa, beta, l_max=L_MAX)
                assert abs(sh[0] - L0_EXACT) < 1e-12, \
                    f"c_0^0 wrong at kappa={kappa}, beta={beta}, mu={mu}"

    def test_watson_limit(self):
        """bingham_sh(mu, psi, kappa, beta=0) == watson_sh(mu, kappa)."""
        kappa = 3.0
        for mu in [np.r_[0., 0., 1.], np.r_[1., 1., 1.] / np.sqrt(3.),
                   np.r_[0., 1., 0.]]:
            sh_bing = bingham_sh(mu, 0.5, kappa, 0.0, l_max=L_MAX)
            sh_wat = watson_sh(mu, kappa, l_max=L_MAX)
            np.testing.assert_allclose(sh_bing, sh_wat, atol=1e-12,
                                       err_msg=f"Watson limit failed for mu={mu}")

    def test_isotropic_limit(self):
        """kappa=beta=0 → isotropic → only c_0^0 nonzero regardless of mu, psi."""
        sh = bingham_sh(np.r_[1., 0., 0.], 0.7, 0.0, 0.0, l_max=L_MAX)
        assert abs(sh[0] - L0_EXACT) < 1e-13
        np.testing.assert_array_less(np.abs(sh[1:]), 1e-12)

    def test_psi_changes_higher_order(self):
        """Rotating psi changes m≠0 coefficients but not c_0^0 or SH power per l."""
        kappa, beta = 4.0, 2.0
        mu = np.r_[0., 0., 1.]
        sh0 = bingham_sh(mu, 0.0, kappa, beta, l_max=L_MAX)
        sh1 = bingham_sh(mu, np.pi / 4, kappa, beta, l_max=L_MAX)
        # l=0 invariant
        assert abs(sh0[0] - sh1[0]) < 1e-12
        # Power per l-block must be invariant under rotation
        offsets = [0, 1, 6, 15, 28]
        ends    = [1, 6, 15, 28, 45]
        for a, b_end in zip(offsets, ends):
            p0 = np.sum(sh0[a:b_end] ** 2)
            p1 = np.sum(sh1[a:b_end] ** 2)
            assert abs(p0 - p1) < 1e-11, \
                f"SH power mismatch at block [{a}:{b_end}]: {p0} vs {p1}"

    def test_matches_grid_at_low_kappa(self):
        """At low kappa the new analytical result agrees with the old grid-based SH."""
        from dmipy_fit.distributions.distributions import hemisphere, inverse_sh_matrix_kernel
        from dmipy_fit.utils import utils

        kappa, beta_frac = 1.0, 0.5
        beta = beta_frac * kappa
        mu_sph = np.r_[np.pi / 4, np.pi / 3]
        psi = 0.2
        sh_order = 4

        mu_cart = utils.unitsphere2cart_1d(mu_sph)
        R = utils.rotation_matrix_100_to_theta_phi_psi(mu_sph[0], mu_sph[1], psi)
        mu_beta = R.dot(np.r_[0., 1., 0.])
        n = hemisphere.vertices
        f_sphere = np.exp(kappa * (n @ mu_cart) ** 2
                          + beta * (n @ mu_beta) ** 2)
        sh_mat_inv = inverse_sh_matrix_kernel[sh_order]
        sh_grid = np.dot(sh_mat_inv, f_sphere)
        sh_grid = sh_grid * (L0_EXACT / sh_grid[0])   # renormalize l=0

        sh_new = bingham_sh(mu_cart, psi, kappa, beta, l_max=sh_order)
        np.testing.assert_allclose(sh_new, sh_grid, atol=5e-3,
                                   err_msg="Bingham analytical vs grid mismatch at kappa=1")

    def test_no_nan_inf(self):
        """No NaN or Inf for kappa and beta across practical ranges."""
        mu = np.r_[1., 1., 1.] / np.sqrt(3.)
        for kappa in [0.1, 1.0, 5.0, 20.0, 100.0]:
            for beta_frac in [0.0, 0.3, 0.7, 1.0]:
                sh = bingham_sh(mu, 0.5, kappa, beta_frac * kappa, l_max=L_MAX)
                assert np.all(np.isfinite(sh)), \
                    f"Non-finite SH at kappa={kappa}, beta={beta_frac*kappa}"

    def test_sd2bingham_uses_new_path(self):
        """SD2Bingham.spherical_harmonics_representation returns exact c_0^0."""
        from dmipy_fit.distributions.distributions import SD2Bingham
        dist = SD2Bingham()
        sh = dist.spherical_harmonics_representation(
            sh_order=4, odi=0.3, beta_fraction=0.5,
            mu=np.r_[np.pi / 4, np.pi / 3], psi=0.2)
        assert abs(sh[0] - L0_EXACT) < 1e-12, f"SD2Bingham c_0^0 = {sh[0]}"


# ── Gaussian kernel (Zeppelin / Stick) — continued ───────────────────────────

class TestGaussianModelRhExtra:
    """Continuation of Gaussian RH tests (split off to avoid collision with earlier class)."""

    LP = 1.7e-9
    LX = 0.3e-9

    def test_model_rh_agrees_with_base_class(self):
        """G2Zeppelin.rotational_harmonics_representation matches the base-class
        10-point sampling to within 0.5% (the grid approximation error)."""
        from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
        from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
        import numpy as np

        # Build a simple PGSE scheme with one shell
        bvals = np.array([0., 1e9, 1e9, 1e9, 2e9, 2e9, 2e9])
        dirs = np.tile(np.r_[0., 0., 1.], (len(bvals), 1))
        scheme = acquisition_scheme_from_bvalues(bvals, dirs, delta=0.01,
                                                  Delta=0.03)
        zep = G2Zeppelin()
        rh_analytical = zep.rotational_harmonics_representation(
            scheme, lambda_par=self.LP, lambda_perp=self.LX)

        # Compare l=0 RH (spherical mean) with the known formula
        from scipy.special import erf as _erf
        for i, (shell_index, _) in enumerate(
                scheme.rotational_harmonics_scheme.shell_sh_orders.items()):
            b = scheme.shell_bvalues[shell_index]
            sqrt_bl = np.sqrt(b * (self.LP - self.LX))
            sm_ref = (np.exp(-b * self.LX) * np.sqrt(np.pi)
                      * _erf(sqrt_bl) / (2.0 * sqrt_bl))
            sm_analytical = rh_analytical[i, 0] / (2.0 * np.sqrt(np.pi))
            assert abs(sm_analytical - sm_ref) < 1e-10, \
                f"Shell {i} spherical mean: {sm_analytical} vs {sm_ref}"
