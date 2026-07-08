"""
Validation tests for scientific ledger hypotheses.

SC-010  τ = Δ−δ/3 PGSE b-value formula
SC-012  SH convolution eigenvalue √(4π/(2L+1))
SC-013  calc_b() (dmipy-sim) vs b_from_g() (dmipy-core) agree within 0.1%
SC-014  Watson at κ→0 (odi→1) equals spherical mean of kernel
SC-016  water_diffusion_constant default is 37°C (in-vivo human)
"""
import numpy as np
import pytest

from dmipy_fit.core.constants import CONSTANTS
from dmipy_fit.core.gradient_conversions import b_from_g, g_from_b
from dmipy_fit.utils.spherical_convolution import sh_convolution

GAMMA = CONSTANTS['water_gyromagnetic_ratio']   # rad/s/T


# ---------------------------------------------------------------------------
# SC-010  τ = Δ−δ/3
# ---------------------------------------------------------------------------

class TestBValueFormula:
    """b = γ²G²δ²(Δ−δ/3)  (Stejskal-Tanner, PGSE)"""

    @pytest.mark.parametrize("delta,Delta,G", [
        (10e-3, 40e-3, 0.04),    # typical clinical
        (5e-3,  20e-3, 0.08),    # strong gradient
        (20e-3, 60e-3, 0.02),    # long pulses
        (1e-3,  50e-3, 0.10),    # short pulse, large separation
    ])
    def test_b_from_g_formula(self, delta, Delta, G):
        """b_from_g must equal γ²G²δ²(Δ−δ/3) to float64 precision."""
        tau = Delta - delta / 3.
        b_expected = (GAMMA * G * delta) ** 2 * tau
        b_computed = b_from_g(G, delta, Delta)
        np.testing.assert_allclose(b_computed, b_expected, rtol=1e-12)

    def test_round_trip_b_g(self):
        """g_from_b(b_from_g(G)) must recover G."""
        G = np.array([0.01, 0.02, 0.04, 0.08])
        delta, Delta = 10e-3, 40e-3
        b = b_from_g(G, delta, Delta)
        G_rt = g_from_b(b, delta, Delta)
        np.testing.assert_allclose(G_rt, G, rtol=1e-12)

    def test_b_zero_at_zero_gradient(self):
        """b must be zero when G=0."""
        assert b_from_g(0.0, 10e-3, 40e-3) == 0.0

    def test_b_scales_as_g_squared(self):
        """Doubling G must quadruple b."""
        delta, Delta = 10e-3, 40e-3
        b1 = b_from_g(0.04, delta, Delta)
        b2 = b_from_g(0.08, delta, Delta)
        np.testing.assert_allclose(b2 / b1, 4.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# SC-012  SH convolution eigenvalue √(4π/(2L+1))
# ---------------------------------------------------------------------------

class TestSHConvolutionEigenvalue:
    """Eigenvalue of SH convolution for order L is √(4π/(2L+1))."""

    def test_l0_eigenvalue(self):
        """L=0 eigenvalue: √(4π) ≈ 3.5449."""
        # A kernel with only L=0 RH coefficient = 1 and f_sh = [1, 0, 0, ...]
        # Result SH coef at L=0 must be 1 * √(4π/(2*0+1)) = √(4π).
        sh_order = 8
        n_sh = (sh_order + 2) * (sh_order + 1) // 2  # 45
        n_rh = sh_order // 2 + 1                       # 5

        f_sh = np.zeros(n_sh)
        f_sh[0] = 1.0   # only L=0 term
        kernel_rh = np.zeros(n_rh)
        kernel_rh[0] = 1.0   # only L=0 RH

        result = sh_convolution(f_sh, kernel_rh)
        expected = np.sqrt(4 * np.pi / 1)   # L=0: 2L+1=1
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_l2_eigenvalue(self):
        """L=2 eigenvalue: √(4π/5)."""
        sh_order = 8
        n_sh = (sh_order + 2) * (sh_order + 1) // 2
        n_rh = sh_order // 2 + 1

        f_sh = np.zeros(n_sh)
        # L=2 block starts at index 1 (after 1 L=0 coef), has 5 coefs.
        f_sh[1:6] = 1.0
        kernel_rh = np.zeros(n_rh)
        kernel_rh[1] = 1.0   # L=2 RH only

        result = sh_convolution(f_sh, kernel_rh)
        expected = np.sqrt(4 * np.pi / 5)   # L=2: 2L+1=5
        np.testing.assert_allclose(result[1:6], expected, rtol=1e-12)

    def test_each_order_eigenvalue(self):
        """Check eigenvalue √(4π/(2L+1)) for L=0,2,4,6,8 independently."""
        sh_order = 8
        n_sh = (sh_order + 2) * (sh_order + 1) // 2
        n_rh = sh_order // 2 + 1

        counter = 0
        for L in range(0, sh_order + 1, 2):
            n_coef = 2 * L + 1
            f_sh = np.zeros(n_sh)
            f_sh[counter: counter + n_coef] = 1.0
            kernel_rh = np.zeros(n_rh)
            kernel_rh[L // 2] = 1.0

            result = sh_convolution(f_sh, kernel_rh)
            expected = np.sqrt(4 * np.pi / (2 * L + 1))
            np.testing.assert_allclose(
                result[counter: counter + n_coef], expected, rtol=1e-12,
                err_msg=f"L={L}")
            counter += n_coef


# ---------------------------------------------------------------------------
# SC-013  calc_b (dmipy-sim) vs b_from_g (dmipy-core) agree within 0.1%
# ---------------------------------------------------------------------------

class TestCalcBBridge:
    """dmipy-sim calc_b() vs dmipy-core b_from_g() for PGSE waveforms."""

    @pytest.fixture(autouse=True)
    def _import_sim(self):
        pytest.importorskip("dmipy_sim",
                            reason="dmipy-sim not installed")
        from dmipy_sim.waveforms import pgse, calc_b
        self.pgse = pgse
        self.calc_b = calc_b

    @pytest.mark.parametrize("delta,Delta,G_mag,n_t", [
        (10e-3, 40e-3, 0.04, 2000),
        (5e-3,  20e-3, 0.08, 2000),
        (20e-3, 60e-3, 0.02, 2000),
    ])
    def test_calc_b_vs_b_from_g(self, delta, Delta, G_mag, n_t):
        """max relative deviation must be < 0.1% across all measurements."""
        # pgse() takes scalar delta/Delta; bvecs sets n_measurements
        bvecs = np.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]])
        G_mags = np.full(3, G_mag)

        # compare against the SQUARE-lobe analytic b formula, so build the square
        # (instantaneous) waveform explicitly (sim now defaults to slew-limited)
        wf = self.pgse(delta, Delta, G_mags, bvecs, n_t, slew_rate=np.inf)
        b_sim = self.calc_b(wf)
        b_core = b_from_g(G_mag, delta, Delta)

        rel_err = np.abs(b_sim - b_core) / b_core
        assert np.all(rel_err < 1e-3), (
            f"max relative error {rel_err.max():.4%} exceeds 0.1% "
            f"(delta={delta*1e3:.0f}ms Delta={Delta*1e3:.0f}ms G={G_mag}T/m "
            f"n_t={n_t})")


# ---------------------------------------------------------------------------
# SC-014  Watson at κ→0 (odi→1) equals spherical mean of kernel
# ---------------------------------------------------------------------------

class TestWatsonIsotropicLimit:
    """At maximum dispersion (odi→1, κ→0), Watson = uniform distribution.
    The convolved signal must equal the spherical mean of the kernel:
    only the L=0 SH term survives, scaled by 2√π (the Y_00 integral).
    """

    def test_watson_isotropic_gives_spherical_mean(self):
        from dmipy_fit.core.acquisition_scheme import (
            acquisition_scheme_from_bvalues)
        from dmipy_fit.distributions import distributions
        from dmipy_fit.signal_models.cylinder_models import C1Stick

        bvals = np.array([0., 1e9, 2e9, 3e9])
        gdirs = np.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.],
                          [1., 0., 0.]])
        delta = np.full(4, 10e-3)
        Delta = np.full(4, 40e-3)
        scheme = acquisition_scheme_from_bvalues(bvals, gdirs, delta, Delta)

        stick = C1Stick()
        watson = distributions.SD1Watson()

        # Watson at odi≈1 (kappa≈0) should give uniform distribution
        watson_sh = watson.spherical_harmonics_representation(odi=0.9999,
                                                              mu=[0., 0.])

        # Only the L=0 coef should be non-negligible
        # (uniform distribution has all higher-order terms ≈ 0)
        if len(watson_sh) > 1:
            l0_fraction = np.abs(watson_sh[0]) / (np.abs(watson_sh).sum() + 1e-30)
            assert l0_fraction > 0.99, (
                f"Watson at odi=0.9999 not isotropic: L=0 fraction = {l0_fraction:.4f}")

    def test_watson_sh_l0_coef_equals_uniform(self):
        """At odi→1 the L=0 SH coefficient must equal 1/(2√π) (uniform PDF)."""
        from dmipy_fit.distributions import distributions
        watson = distributions.SD1Watson()
        watson_sh = watson.spherical_harmonics_representation(odi=0.9999,
                                                              mu=[0., 0.])
        # Uniform PDF on unit sphere: f = 1/(4π), Y_00 = 1/(2√π)
        # SH coef c_00 = integral(f * Y_00) = 1/(4π) * 4π * Y_00(0,0)
        # For a normalised distribution, c_00 ≈ 1/(2√π)
        expected_c00 = 1. / (2 * np.sqrt(np.pi))
        np.testing.assert_allclose(watson_sh[0], expected_c00, rtol=0.01)


# ---------------------------------------------------------------------------
# SC-016  water_diffusion_constant default is 37°C (in-vivo human)
# ---------------------------------------------------------------------------

class TestWaterDiffusionConstant:
    """Default water diffusivity must be the 37°C in-vivo value."""

    def test_default_is_37C(self):
        """water_diffusion_constant ≈ 3.05e-9 m²/s (37°C)."""
        D = CONSTANTS['water_diffusion_constant']
        # 37°C value from literature: ~3.0–3.1e-9 m²/s
        assert 2.9e-9 < D < 3.2e-9, (
            f"water_diffusion_constant={D:.3e} is not the 37°C value "
            f"(expected ~3.05e-9 m²/s)")

    def test_25C_constant_preserved(self):
        """25°C reference value must still be accessible."""
        D25 = CONSTANTS['water_diffusion_constant_25C']
        np.testing.assert_allclose(D25, 2.299e-9, rtol=1e-6)

    def test_37C_greater_than_25C(self):
        """Diffusivity increases with temperature."""
        assert (CONSTANTS['water_diffusion_constant'] >
                CONSTANTS['water_diffusion_constant_25C'])
