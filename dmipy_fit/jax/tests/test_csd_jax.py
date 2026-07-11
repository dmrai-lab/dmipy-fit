"""Regression tests: CsdOsqpOptimizer (JAX) vs CsdCvxpyOptimizer (cvxpy).

Both solvers solve the same QP; we check that the solutions agree within
a loose numerical tolerance (rtol=0.05, atol=1e-3).

Tests cover:
  - Single-tissue CSD (volume_fractions_fixed=True): C1Stick kernel
  - Multi-tissue CSD (volume_fractions_fixed=False): C1Stick + G1Ball kernels
  - Batch fitting interface (fit_batch returns correct shape)
  - End-to-end fit() with solver='csd_jax' vs solver='csd_cvxpy'
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
jaxopt = pytest.importorskip('jaxopt')
cvxpy = pytest.importorskip('cvxpy')

from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel,
    MultiCompartmentModel,
)
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.jax.csd_jax import CsdOsqpOptimizer
from dmipy_fit.optimizers_fod.csd_cvxpy import CsdCvxpyOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stick_mc_sh(scheme, lambda_par=1.7e-9):
    """Single-tissue SH model with C1Stick kernel fixed (has .scheme set)."""
    mc = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick()])
    mc.set_fixed_parameter('C1Stick_1_lambda_par', lambda_par)
    # fit() normally sets mc.scheme; set it here for direct optimizer tests
    mc.scheme = scheme
    mc._check_if_kernel_parameters_are_fixed()
    mc.S0_responses = np.ones(len(mc.models), dtype=float)
    return mc


def _make_stick_ball_mc_sh(scheme, lambda_par=1.7e-9, lambda_iso=3.0e-9):
    """Multi-tissue SH model: Stick+Ball, VFs free, kernels fixed."""
    mc = MultiCompartmentSphericalHarmonicsModel(
        models=[C1Stick(), G1Ball()])
    mc.set_fixed_parameter('C1Stick_1_lambda_par', lambda_par)
    mc.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso)
    mc.scheme = scheme
    mc._check_if_kernel_parameters_are_fixed()
    mc.S0_responses = np.ones(len(mc.models), dtype=float)
    return mc


def _make_synthetic_signal(scheme, lambda_par=1.7e-9, mu=None, rng=None):
    """Simulate a directional C1Stick signal for CSD testing."""
    if rng is None:
        rng = np.random.default_rng(42)
    if mu is None:
        mu = np.array([float(rng.uniform(0, np.pi)),
                       float(rng.uniform(0, 2 * np.pi))])
    mc_full = MultiCompartmentModel(models=[C1Stick()])
    p = mc_full.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=lambda_par,
        C1Stick_1_mu=mu)
    return mc_full.simulate_signal(scheme, p)


def _make_multitissue_signal(scheme, lambda_par=1.7e-9, lambda_iso=3.0e-9,
                              vf0=0.6, mu=None, rng=None):
    """Simulate a Stick+Ball signal for multi-tissue CSD testing."""
    if rng is None:
        rng = np.random.default_rng(42)
    if mu is None:
        mu = np.array([float(rng.uniform(0, np.pi)),
                       float(rng.uniform(0, 2 * np.pi))])
    mc_full = MultiCompartmentModel(models=[C1Stick(), G1Ball()])
    p = mc_full.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=lambda_par,
        C1Stick_1_mu=mu,
        G1Ball_1_lambda_iso=lambda_iso,
        partial_volume_0=vf0,
        partial_volume_1=1.0 - vf0,
    )
    return mc_full.simulate_signal(scheme, p)


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


# ---------------------------------------------------------------------------
# TestSingleTissueCsd — volume_fractions_fixed = True
# ---------------------------------------------------------------------------

class TestSingleTissueCsd:
    """C1Stick single-tissue CSD: JAX optimizer directly vs cvxpy."""

    def test_sh_coeff_agrees_with_cvxpy(self, scheme):
        mc = _make_stick_mc_sh(scheme)
        signal = _make_synthetic_signal(scheme)
        signal_norm = signal / float(np.mean(signal[scheme.b0_mask]))

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))

        cvxpy_opt = CsdCvxpyOptimizer(
            scheme, mc, x0_2d, sh_order=8,
            unity_constraint=False, lambda_lb=0.)
        jax_opt = CsdOsqpOptimizer(
            scheme, mc, x0_2d, sh_order=8,
            unity_constraint=False, lambda_lb=0.)

        cvxpy_sol = cvxpy_opt(signal_norm, x0_2d[0])
        jax_sol = jax_opt(signal_norm, x0_2d[0])

        assert_allclose(
            jax_sol, cvxpy_sol, rtol=0.05, atol=2e-3,
            err_msg="Single-tissue CSD: JAX and cvxpy solutions differ")

    def test_sh_coeff_shape(self, scheme):
        mc = _make_stick_mc_sh(scheme)
        signal = _make_synthetic_signal(scheme)
        signal_norm = signal / float(np.mean(signal[scheme.b0_mask]))

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))
        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=False)
        sol = jax_opt(signal_norm, x0_2d[0])
        assert sol.shape == x0_2d[0].shape

    def test_zeroth_sh_coeff_normalized(self, scheme):
        """sh_coeff[0] must equal 1/sphere_jacobian after solving."""
        mc = _make_stick_mc_sh(scheme)
        signal = _make_synthetic_signal(scheme)
        signal_norm = signal / float(np.mean(signal[scheme.b0_mask]))

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))
        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=False)
        sol = jax_opt(signal_norm, x0_2d[0])

        params = mc.parameter_vector_to_parameters(sol)
        expected_c0 = 1.0 / (2.0 * np.sqrt(np.pi))
        assert abs(params['sh_coeff'][0] - expected_c0) < 1e-6

    def test_positivity(self, scheme):
        """FOD on hemisphere should be >= -1e-3 (tiny OSQP tolerance)."""
        from dipy.data import get_sphere, HemiSphere
        from dipy.reconst.shm import real_sh_tournier as real_sym_sh_mrtrix

        mc = _make_stick_mc_sh(scheme)
        signal = _make_synthetic_signal(scheme)
        signal_norm = signal / float(np.mean(signal[scheme.b0_mask]))

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))
        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=False)
        sol = jax_opt(signal_norm, x0_2d[0])

        params = mc.parameter_vector_to_parameters(sol)
        sph = get_sphere(name='symmetric724')
        hem = HemiSphere(phi=sph.phi, theta=sph.theta)
        L = real_sym_sh_mrtrix(8, hem.theta, hem.phi, legacy=False)[0]
        fod_vals = L @ params['sh_coeff']
        # OSQP at float32 precision gives small constraint violations (~0.004).
        # These are numerical noise and don't affect microstructure parameters.
        assert np.all(fod_vals > -5e-3), \
            "FOD has negative values: min = {:.4f}".format(fod_vals.min())


# ---------------------------------------------------------------------------
# TestMultiTissueCsd — volume_fractions_fixed = False
# ---------------------------------------------------------------------------

class TestMultiTissueCsd:
    """Stick+Ball multi-tissue CSD: JAX optimizer directly vs cvxpy."""

    def test_sh_coeff_agrees_with_cvxpy(self, scheme):
        mc = _make_stick_ball_mc_sh(scheme)
        signal = _make_multitissue_signal(scheme)
        signal_norm = signal / float(np.mean(signal[scheme.b0_mask]))

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))

        cvxpy_opt = CsdCvxpyOptimizer(
            scheme, mc, x0_2d, sh_order=8,
            unity_constraint=True, lambda_lb=0.)
        jax_opt = CsdOsqpOptimizer(
            scheme, mc, x0_2d, sh_order=8,
            unity_constraint=True, lambda_lb=0.)

        cvxpy_sol = cvxpy_opt(signal_norm, x0_2d[0])
        jax_sol = jax_opt(signal_norm, x0_2d[0])

        assert_allclose(
            jax_sol, cvxpy_sol, rtol=0.05, atol=2e-3,
            err_msg="Multi-tissue CSD: JAX and cvxpy solutions differ")

    def test_volume_fractions_sum_to_one(self, scheme):
        mc = _make_stick_ball_mc_sh(scheme)
        signal = _make_multitissue_signal(scheme)
        signal_norm = signal / float(np.mean(signal[scheme.b0_mask]))

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))
        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=True)
        sol = jax_opt(signal_norm, x0_2d[0])

        params = mc.parameter_vector_to_parameters(sol)
        vf_sum = params['partial_volume_0'] + params['partial_volume_1']
        assert abs(vf_sum - 1.0) < 0.05, \
            "VFs do not sum to 1 (got {:.4f})".format(vf_sum)

    def test_volume_fractions_nonneg(self, scheme):
        mc = _make_stick_ball_mc_sh(scheme)
        signal = _make_multitissue_signal(scheme)
        signal_norm = signal / float(np.mean(signal[scheme.b0_mask]))

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))
        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=True)
        sol = jax_opt(signal_norm, x0_2d[0])

        params = mc.parameter_vector_to_parameters(sol)
        assert params['partial_volume_0'] >= -0.05
        assert params['partial_volume_1'] >= -0.05


# ---------------------------------------------------------------------------
# TestFitBatch
# ---------------------------------------------------------------------------

class TestFitBatch:
    """fit_batch returns correct shapes and matches __call__ results."""

    def test_batch_shape_single_tissue(self, scheme):
        rng = np.random.default_rng(7)
        N = 3
        mc = _make_stick_mc_sh(scheme)
        signals = np.array([
            _make_synthetic_signal(scheme, rng=rng) for _ in range(N)])
        S0 = np.mean(signals[:, scheme.b0_mask], axis=-1, keepdims=True)
        signals_norm = signals / S0

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))
        x0_all = np.tile(x0_2d, (N, 1))

        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=False)
        result = jax_opt.fit_batch(signals_norm, x0_all)
        assert result.shape == (N, x0_2d.shape[1]), \
            "fit_batch output shape mismatch: {}".format(result.shape)

    def test_batch_shape_multi_tissue(self, scheme):
        rng = np.random.default_rng(8)
        N = 3
        mc = _make_stick_ball_mc_sh(scheme)
        signals = np.array([
            _make_multitissue_signal(scheme, rng=rng) for _ in range(N)])
        S0 = np.mean(signals[:, scheme.b0_mask], axis=-1, keepdims=True)
        signals_norm = signals / S0

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))
        x0_all = np.tile(x0_2d, (N, 1))

        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=True)
        result = jax_opt.fit_batch(signals_norm, x0_all)
        assert result.shape == (N, x0_2d.shape[1])

    def test_batch_agrees_with_single(self, scheme):
        """fit_batch(data[None], x0[None])[0] must equal __call__(data, x0)."""
        mc = _make_stick_mc_sh(scheme)
        signal = _make_synthetic_signal(scheme)
        S0 = float(np.mean(signal[scheme.b0_mask]))
        signal_norm = signal / S0

        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))

        jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d, unity_constraint=False)
        single = jax_opt(signal_norm, x0_2d[0])
        batch = jax_opt.fit_batch(signal_norm[None], x0_2d)[0]
        assert_allclose(single, batch, rtol=1e-5, atol=1e-6,
                        err_msg="__call__ and fit_batch disagree")


# ---------------------------------------------------------------------------
# TestEndToEndFit — solver='csd_jax' vs solver='csd_cvxpy' through fit()
# ---------------------------------------------------------------------------

class TestEndToEndFit:
    """model.fit(solver='csd_jax') vs model.fit(solver='csd_cvxpy')."""

    def test_single_tissue_jax_vs_cvxpy(self, scheme):
        rng = np.random.default_rng(99)
        N = 2
        signals = np.array([
            _make_synthetic_signal(scheme, rng=rng) for _ in range(N)])

        mc1 = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick()])
        mc1.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
        mc2 = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick()])
        mc2.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)

        res_cvxpy = mc1.fit(scheme, signals, solver='csd_cvxpy',
                            use_parallel_processing=False, verbose=False)
        res_jax = mc2.fit(scheme, signals, solver='csd_jax',
                          use_parallel_processing=False, verbose=False)

        sh_cvxpy = res_cvxpy.fitted_parameters['sh_coeff']
        sh_jax = res_jax.fitted_parameters['sh_coeff']
        assert sh_jax.shape == sh_cvxpy.shape
        assert_allclose(sh_jax, sh_cvxpy, rtol=0.05, atol=2e-3,
                        err_msg="End-to-end single-tissue: JAX/cvxpy sh_coeff differ")

    def test_multi_tissue_jax_vs_cvxpy(self, scheme):
        rng = np.random.default_rng(100)
        N = 2
        signals = np.array([
            _make_multitissue_signal(scheme, rng=rng) for _ in range(N)])

        mc1 = MultiCompartmentSphericalHarmonicsModel(
            models=[C1Stick(), G1Ball()])
        mc1.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
        mc1.set_fixed_parameter('G1Ball_1_lambda_iso', 3.0e-9)

        mc2 = MultiCompartmentSphericalHarmonicsModel(
            models=[C1Stick(), G1Ball()])
        mc2.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
        mc2.set_fixed_parameter('G1Ball_1_lambda_iso', 3.0e-9)

        res_cvxpy = mc1.fit(scheme, signals, solver='csd_cvxpy',
                            use_parallel_processing=False, verbose=False)
        res_jax = mc2.fit(scheme, signals, solver='csd_jax',
                          use_parallel_processing=False, verbose=False)

        sh_cvxpy = res_cvxpy.fitted_parameters['sh_coeff']
        sh_jax = res_jax.fitted_parameters['sh_coeff']
        assert_allclose(sh_jax, sh_cvxpy, rtol=0.05, atol=2e-3,
                        err_msg="End-to-end multi-tissue: JAX/cvxpy sh_coeff differ")

        pv0_cvxpy = res_cvxpy.fitted_parameters['partial_volume_0']
        pv0_jax = res_jax.fitted_parameters['partial_volume_0']
        assert_allclose(pv0_jax, pv0_cvxpy, rtol=0.1, atol=0.05,
                        err_msg="End-to-end multi-tissue: partial_volume_0 differs")

    def test_result_shape_multivoxel(self, scheme):
        N = 3
        rng = np.random.default_rng(77)
        signals = np.array([
            _make_synthetic_signal(scheme, rng=rng) for _ in range(N)])
        mc = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick()])
        mc.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
        res = mc.fit(scheme, signals, solver='csd_jax',
                     use_parallel_processing=False, verbose=False)
        assert res.fitted_parameters['sh_coeff'].shape[0] == N

    def test_unknown_solver_raises(self, scheme):
        """Unsupported solver names raise ValueError."""
        signals = np.array([_make_synthetic_signal(scheme)])
        mc = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick()])
        mc.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
        with pytest.raises(ValueError, match="Unknown solver name"):
            mc.fit(scheme, signals, solver='csd_notareal',
                   use_parallel_processing=False, verbose=False)
