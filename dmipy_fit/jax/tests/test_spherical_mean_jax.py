"""Tests for JAX-accelerated MultiCompartmentSphericalMeanModel fitting."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import (
    MultiCompartmentModel,
    MultiCompartmentSphericalMeanModel,
)
from dmipy_fit.jax.multicompartment_jax import build_mc_sm_forward_fn
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


def _make_full_signal(mc_full, scheme, params_dict):
    """Generate full-measurement signal from a regular MultiCompartmentModel."""
    params = mc_full.parameters_to_parameter_vector(**params_dict)
    return mc_full.simulate_signal(scheme, params)


# ---------------------------------------------------------------------------
# build_mc_sm_forward_fn correctness (shell-level output)
# ---------------------------------------------------------------------------

class TestSmForwardFn:
    """Test that build_mc_sm_forward_fn produces correct spherical-mean signals."""

    def test_single_ball_sm(self, scheme):
        ball = G1Ball()
        mc   = MultiCompartmentSphericalMeanModel(models=[ball])
        params = mc.parameters_to_parameter_vector(G1Ball_1_lambda_iso=1.7e-9)
        E_np  = mc.simulate_signal(scheme, params)
        fn    = build_mc_sm_forward_fn(mc, scheme, broadcast=False)
        E_jax = np.array(fn(jnp.array(params)))
        assert E_jax.shape == E_np.shape
        assert_allclose(E_jax, E_np, rtol=1e-5, atol=1e-6,
                        err_msg="JAX SM forward (Ball) deviates from numpy")

    def test_stick_ball_sm(self, scheme):
        stick = C1Stick()
        ball  = G1Ball()
        mc    = MultiCompartmentSphericalMeanModel(models=[stick, ball])
        params = mc.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=1.7e-9,
            G1Ball_1_lambda_iso=3.0e-9,
            partial_volume_0=0.6,
            partial_volume_1=0.4,
        )
        E_np  = mc.simulate_signal(scheme, params)
        fn    = build_mc_sm_forward_fn(mc, scheme, broadcast=False)
        E_jax = np.array(fn(jnp.array(params)))
        assert_allclose(E_jax, E_np, rtol=1e-5, atol=1e-6,
                        err_msg="JAX SM forward (Stick+Ball) deviates from numpy")

    def test_zeppelin_ball_sm(self, scheme):
        zep  = G2Zeppelin()
        ball = G1Ball()
        mc   = MultiCompartmentSphericalMeanModel(models=[zep, ball])
        params = mc.parameters_to_parameter_vector(
            G2Zeppelin_1_lambda_par=1.7e-9,
            G2Zeppelin_1_lambda_perp=0.5e-9,
            G1Ball_1_lambda_iso=3.0e-9,
            partial_volume_0=0.7,
            partial_volume_1=0.3,
        )
        E_np  = mc.simulate_signal(scheme, params)
        fn    = build_mc_sm_forward_fn(mc, scheme, broadcast=False)
        E_jax = np.array(fn(jnp.array(params)))
        assert_allclose(E_jax, E_np, rtol=1e-5, atol=1e-6,
                        err_msg="JAX SM forward (Zeppelin+Ball) deviates from numpy")

    def test_b0_shells_are_one(self, scheme):
        """b0 shells must be exactly 1.0."""
        ball = G1Ball()
        mc   = MultiCompartmentSphericalMeanModel(models=[ball])
        params = mc.parameters_to_parameter_vector(G1Ball_1_lambda_iso=1.7e-9)
        fn = build_mc_sm_forward_fn(mc, scheme, broadcast=False)  # per-shell
        E  = fn(jnp.array(params))
        b0_vals = E[scheme.shell_b0_mask]
        assert_allclose(np.array(b0_vals), np.ones_like(np.array(b0_vals)),
                        atol=1e-6, err_msg="b0 shells should be 1.0")

    def test_jit_compiles(self, scheme):
        stick = C1Stick()
        ball  = G1Ball()
        mc    = MultiCompartmentSphericalMeanModel(models=[stick, ball])
        params = mc.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=1.7e-9,
            G1Ball_1_lambda_iso=3.0e-9,
            partial_volume_0=0.6,
            partial_volume_1=0.4,
        )
        fn = jax.jit(build_mc_sm_forward_fn(mc, scheme))
        E  = fn(jnp.array(params))
        assert jnp.isfinite(E).all()

    def test_grad_finite(self, scheme):
        stick = C1Stick()
        ball  = G1Ball()
        mc    = MultiCompartmentSphericalMeanModel(models=[stick, ball])
        params = mc.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=1.7e-9,
            G1Ball_1_lambda_iso=3.0e-9,
            partial_volume_0=0.6,
            partial_volume_1=0.4,
        )
        fn     = build_mc_sm_forward_fn(mc, scheme)
        target = fn(jnp.array(params))

        def loss(p):
            return jnp.mean((fn(p) - target) ** 2)

        g = jax.grad(loss)(jnp.array(params))
        assert jnp.isfinite(g).all(), "SM forward grad has non-finite values"


# ---------------------------------------------------------------------------
# solver='jax' end-to-end fitting tests
#
# Note: MultiCompartmentSphericalMeanModel.fit() expects FULL measurement
# data (N_meas=288), not shell-level data. It internally calls
# estimate_spherical_mean_multi_shell to produce the shell-level signal for
# fitting.  We therefore generate ground-truth signals from full MC models.
# ---------------------------------------------------------------------------

class TestSmJaxFit:
    """End-to-end tests using MultiCompartmentSphericalMeanModel.fit(solver='jax')."""

    def _ball_setup(self, scheme):
        """Ball model: full MC model + SM model, return (sm_model, full_signal)."""
        ball_mc  = MultiCompartmentModel(models=[G1Ball()])
        ball_sm  = MultiCompartmentSphericalMeanModel(models=[G1Ball()])
        E = _make_full_signal(ball_mc, scheme,
                              {'G1Ball_1_lambda_iso': 1.7e-9})
        return ball_sm, E

    def _stick_ball_setup(self, scheme, lambda_par=1.7e-9, lambda_iso=3.0e-9,
                          vf0=0.6):
        """Stick+Ball: full MC model + SM model, return (sm_model, full_signal)."""
        stick = C1Stick()
        ball  = G1Ball()
        # Full anisotropic model for signal generation (with fixed orientation)
        mc_full = MultiCompartmentModel(models=[C1Stick(), G1Ball()])
        E = _make_full_signal(mc_full, scheme, {
            'C1Stick_1_lambda_par': lambda_par,
            'C1Stick_1_mu': np.array([np.pi / 4, np.pi / 3]),
            'G1Ball_1_lambda_iso': lambda_iso,
            'partial_volume_0': vf0,
            'partial_volume_1': 1.0 - vf0,
        })
        sm_model = MultiCompartmentSphericalMeanModel(models=[stick, ball])
        return sm_model, E

    def test_single_ball_fit(self, scheme):
        sm_model, E = self._ball_setup(scheme)
        result = sm_model.fit(scheme, E[None], solver='jax')
        fitted = float(np.squeeze(
            result.fitted_parameters['G1Ball_1_lambda_iso']))
        assert_allclose(fitted, 1.7e-9, rtol=0.05,
                        err_msg="Ball JAX SM fit: lambda_iso off by >5%")

    def test_stick_ball_fit_lambda_par(self, scheme):
        sm_model, E = self._stick_ball_setup(scheme)
        result = sm_model.fit(scheme, E[None], solver='jax')
        fitted = float(np.squeeze(
            result.fitted_parameters['C1Stick_1_lambda_par']))
        assert_allclose(fitted, 1.7e-9, rtol=0.1,
                        err_msg="Stick+Ball JAX SM fit: lambda_par off by >10%")

    def test_stick_ball_fit_lambda_iso(self, scheme):
        # Use lambda_iso=2.0e-9 (away from the 3.0e-9 boundary) so float32
        # gradients remain informative and the optimizer can converge.
        sm_model, E = self._stick_ball_setup(scheme, lambda_iso=2.0e-9)
        result = sm_model.fit(scheme, E[None], solver='jax')
        fitted = float(np.squeeze(
            result.fitted_parameters['G1Ball_1_lambda_iso']))
        assert_allclose(fitted, 2.0e-9, rtol=0.1,
                        err_msg="Stick+Ball JAX SM fit: lambda_iso off by >10%")

    def test_stick_ball_fit_volume_fraction(self, scheme):
        sm_model, E = self._stick_ball_setup(scheme)
        result = sm_model.fit(scheme, E[None], solver='jax')
        fitted_vf = float(np.squeeze(
            result.fitted_parameters['partial_volume_0']))
        assert abs(fitted_vf - 0.6) < 0.15, \
            "Stick+Ball JAX SM fit: partial_volume_0 off by >0.15"

    def test_multivoxel_sm_fit(self, scheme):
        """Fit 3 identical voxels; checks vmap batch correctness and shape."""
        # Use identical voxels so all share the same x0 → same local minimum.
        # This tests that vmap runs and returns correct shapes/values.
        # Differentiating voxels with distinct parameters requires a better x0
        # initializer (e.g., SM brute grid) which is deferred to future work.
        stick = C1Stick()
        ball  = G1Ball()
        sm_model = MultiCompartmentSphericalMeanModel(models=[stick, ball])
        mc_full  = MultiCompartmentModel(models=[C1Stick(), G1Ball()])

        E = _make_full_signal(mc_full, scheme, {
            'C1Stick_1_lambda_par': 1.7e-9,
            'C1Stick_1_mu': np.array([np.pi / 4, np.pi / 3]),
            'G1Ball_1_lambda_iso': 2.0e-9,
            'partial_volume_0': 0.6,
            'partial_volume_1': 0.4,
        })
        E_all = np.tile(E, (3, 1))     # 3 identical voxels

        result = sm_model.fit(scheme, E_all, solver='jax')

        fitted = result.fitted_parameters['C1Stick_1_lambda_par'].flatten()
        assert_allclose(fitted, 1.7e-9, rtol=0.15,
                        err_msg="Multivoxel SM fit: lambda_par off by >15%")

    def test_result_shape(self, scheme):
        """Result arrays have correct shape for multi-voxel input."""
        sm_model, E = self._ball_setup(scheme)
        E_all = np.tile(E[None], (5, 1))   # (5, N_meas)
        result = sm_model.fit(scheme, E_all, solver='jax')
        assert result.fitted_parameters['G1Ball_1_lambda_iso'].shape == (5,)
