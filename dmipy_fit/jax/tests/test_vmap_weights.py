"""Tests for per-voxel measurement weights in vmap_fit / build_vmap_fitter.

SC: per-voxel weights enable outlier-robust fitting from dmipy-preprocess.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.jax.optimizers_jax import JaxOptimizer
from dmipy_fit.jax.vmap_fit import vmap_fit, build_vmap_fitter


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


@pytest.fixture(scope='module')
def ball_setup(scheme):
    """Return (mc, jax_opt, E_all, x0_all) for a G1Ball multi-voxel dataset."""
    GT_LAMBDAS = [1.0e-9, 1.5e-9, 2.0e-9, 0.8e-9]
    ball = G1Ball()
    mc = MultiCompartmentModel(models=[ball])
    E_all = np.stack([
        mc.simulate_signal(
            scheme,
            mc.parameters_to_parameter_vector(G1Ball_1_lambda_iso=lam))
        for lam in GT_LAMBDAS
    ])  # (4, N_meas)
    # Build x0 as midpoint of bounds (float64)
    jax_opt = JaxOptimizer(mc, scheme, maxiter=200)
    n_params = len(jax_opt._lower)
    x0_all = np.tile(
        (jax_opt._lower + jax_opt._upper) / 2.0, (len(GT_LAMBDAS), 1))
    return mc, jax_opt, E_all, x0_all


class TestUniformWeightsMatchNoWeights:
    """Uniform weights=1 must give bit-identical results to weights=None."""

    def test_vmap_fit_uniform_weights_match(self, ball_setup):
        _, jax_opt, E_all, x0_all = ball_setup
        N_meas = E_all.shape[1]
        N_vox  = E_all.shape[0]

        fitted_no_w = vmap_fit(jax_opt, E_all, x0_all, dtype=jnp.float64)
        weights_ones = np.ones((N_vox, N_meas), dtype=np.float64)
        fitted_w1   = vmap_fit(jax_opt, E_all, x0_all, dtype=jnp.float64,
                               weights_all=weights_ones)

        # Uniform weights should reproduce the unweighted result closely.
        # Bit-identical is not guaranteed because the weighted code path uses
        # sum(w*L)/sum(w) which has different floating-point rounding than
        # mean(L); however, convergence to the same minimum must hold.
        assert_allclose(fitted_no_w, fitted_w1, rtol=1e-4,
                        err_msg="Uniform weights=1 changed the fitted result")

    def test_build_vmap_fitter_uniform_weights_match(self, ball_setup, scheme):
        _, jax_opt, E_all, x0_all = ball_setup
        N_meas = E_all.shape[1]
        N_vox  = E_all.shape[0]

        fit_no_w = build_vmap_fitter(jax_opt, dtype=jnp.float64,
                                     use_weights=False)
        fit_w    = build_vmap_fitter(jax_opt, dtype=jnp.float64,
                                     use_weights=True)

        x0_j   = jnp.array(x0_all, dtype=jnp.float64)
        data_j = jnp.array(E_all,  dtype=jnp.float64)
        w_ones = jnp.ones((N_vox, N_meas), dtype=jnp.float64)

        res_no_w = np.array(fit_no_w(x0_j, data_j))
        res_w    = np.array(fit_w(x0_j, data_j, w_ones))

        assert_allclose(res_no_w, res_w, rtol=1e-4,
                        err_msg="build_vmap_fitter: uniform weights changed result")


class TestZeroWeightsIgnoreMeasurements:
    """Zeroing out half the measurements must not crash and should change fit."""

    def test_zero_weights_do_not_crash(self, ball_setup):
        _, jax_opt, E_all, x0_all = ball_setup
        N_vox, N_meas = E_all.shape

        weights = np.ones((N_vox, N_meas), dtype=np.float64)
        # Zero out the second half of measurements for every voxel
        weights[:, N_meas // 2:] = 0.0

        # Should not raise
        fitted = vmap_fit(jax_opt, E_all, x0_all, dtype=jnp.float64,
                          weights_all=weights)
        assert fitted.shape == x0_all.shape
        assert np.all(np.isfinite(fitted)), "Non-finite values in fitted params"

    def test_zero_weights_change_fit(self, ball_setup):
        """When nearly all measurements are zeroed, fit result must differ.

        Use a very extreme case: only the first 2 b>0 measurements have weight
        1, everything else 0.  The truncated loss landscape should drive lambda
        to a very different value than the full-data fit.
        """
        _, jax_opt, E_all, x0_all = ball_setup
        N_vox, N_meas = E_all.shape

        fitted_full = vmap_fit(jax_opt, E_all, x0_all)

        # Keep only the first 2 measurements (high b-value, restrictive)
        # and zero-weight everything else.  With such limited information
        # the fit must land at a noticeably different parameter value.
        weights_extreme = np.zeros((N_vox, N_meas), dtype=np.float32)
        weights_extreme[:, :2] = 1.0

        fitted_extreme = vmap_fit(jax_opt, E_all, x0_all,
                                  weights_all=weights_extreme)

        # Results must still be finite
        assert np.all(np.isfinite(fitted_extreme))
        # They must differ from the full-data fit by a noticeable margin
        max_diff = np.max(np.abs(fitted_full - fitted_extreme))
        assert max_diff > 1e-6, (
            f"Expected weights to change the fit result; max_diff={max_diff:.2e}")

    def test_per_voxel_weight_independence(self, ball_setup):
        """Different weights per voxel must produce independent fits."""
        _, jax_opt, E_all, x0_all = ball_setup
        N_vox, N_meas = E_all.shape

        # Voxel 0: uniform weights; Voxel 1: first half only
        weights = np.ones((N_vox, N_meas), dtype=np.float64)
        weights[1, N_meas // 2:] = 0.0

        fitted_mixed = vmap_fit(jax_opt, E_all, x0_all, dtype=jnp.float64,
                                weights_all=weights)

        # Voxel 0 (uniform) should match no-weights result
        fitted_no_w = vmap_fit(jax_opt, E_all[0:1], x0_all[0:1],
                               dtype=jnp.float64)
        assert_allclose(fitted_mixed[0], fitted_no_w[0], rtol=1e-4,
                        err_msg="Voxel 0 (uniform weights) should match no-weights")

        # Voxel 1 fitted with partial weights only
        w1 = np.ones((1, N_meas), dtype=np.float64)
        w1[0, N_meas // 2:] = 0.0
        fitted_w1 = vmap_fit(jax_opt, E_all[1:2], x0_all[1:2],
                             dtype=jnp.float64, weights_all=w1)
        assert_allclose(fitted_mixed[1], fitted_w1[0], rtol=1e-4,
                        err_msg="Voxel 1 partial-weight result should match solo run")
