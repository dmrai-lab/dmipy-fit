"""Tests for JAX Watson/Bingham distributed model forward functions."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.distributions.distribute_models import (
    SD1WatsonDistributed,
    SD2BinghamDistributed,
)
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


MU      = np.array([np.pi / 4, np.pi / 3])
ODI     = 0.3
LAMBDA  = 1.7e-9


# ---------------------------------------------------------------------------
# SD1Watson distributed (single inner model)
# ---------------------------------------------------------------------------

class TestWatsonStick:
    def _make(self, scheme):
        stick   = C1Stick()
        wstick  = SD1WatsonDistributed(models=[stick])
        mc      = MultiCompartmentModel(models=[wstick])
        gt_params = mc.parameters_to_parameter_vector(
            SD1WatsonDistributed_1_SD1Watson_1_mu=MU,
            SD1WatsonDistributed_1_SD1Watson_1_odi=ODI,
            SD1WatsonDistributed_1_C1Stick_1_lambda_par=LAMBDA,
        )
        E_np = mc.simulate_signal(scheme, gt_params)
        return mc, gt_params, E_np

    def test_jax_forward_matches_numpy(self, scheme):
        mc, gt_params, E_np = self._make(scheme)
        fn = build_mc_forward_fn(mc, scheme)
        E_jax = np.array(fn(jnp.array(gt_params)))
        assert_allclose(E_jax, E_np, rtol=5e-3, atol=5e-4,
                        err_msg="JAX Watson-Stick signal deviates >0.5% from numpy")

    def test_signal_in_range(self, scheme):
        mc, gt_params, _ = self._make(scheme)
        fn = build_mc_forward_fn(mc, scheme)
        E  = fn(jnp.array(gt_params))
        assert jnp.all(E >= 0.0)
        assert jnp.all(E <= 1.0 + 1e-6)

    def test_jit_compiles(self, scheme):
        mc, gt_params, _ = self._make(scheme)
        fn = jax.jit(build_mc_forward_fn(mc, scheme))
        E  = fn(jnp.array(gt_params))
        assert jnp.isfinite(E).all()

    def test_grad_finite(self, scheme):
        mc, gt_params, E_np = self._make(scheme)
        fn   = build_mc_forward_fn(mc, scheme)
        target = jnp.array(E_np)

        def loss(p):
            return jnp.mean((fn(p) - target) ** 2)

        g = jax.grad(loss)(jnp.array(gt_params))
        assert jnp.isfinite(g).all()

    def test_solver_jax_fits(self, scheme):
        """solver='jax' recovers ODI and lambda_par for Watson-Stick."""
        mc, gt_params, E_np = self._make(scheme)
        result = mc.fit(scheme, E_np[None], solver='jax')
        fitted_odi = float(np.squeeze(
            result.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']))
        fitted_lam = float(np.squeeze(
            result.fitted_parameters['SD1WatsonDistributed_1_C1Stick_1_lambda_par']))
        assert_allclose(fitted_lam, LAMBDA, rtol=0.1,
                        err_msg="Fitted lambda_par off by >10%")
        assert abs(fitted_odi - ODI) < 0.2, "Fitted ODI off by >0.2"


# ---------------------------------------------------------------------------
# SD1Watson: NODDI-style (Stick + Zeppelin, with linked params)
# ---------------------------------------------------------------------------

class TestWatsonNODDI:
    def _make_noddi(self, scheme):
        stick    = C1Stick()
        zeppelin = G2Zeppelin()
        wmodel   = SD1WatsonDistributed(models=[stick, zeppelin])
        # Link Zeppelin lambda_par = Stick lambda_par (standard NODDI constraint)
        wmodel.set_equal_parameter('C1Stick_1_lambda_par',
                                   'G2Zeppelin_1_lambda_par')
        ball = G1Ball()
        mc   = MultiCompartmentModel(models=[wmodel, ball])
        return mc

    def _noddi_params(self, mc):
        return mc.parameters_to_parameter_vector(
            SD1WatsonDistributed_1_SD1Watson_1_mu=MU,
            SD1WatsonDistributed_1_SD1Watson_1_odi=0.3,
            SD1WatsonDistributed_1_C1Stick_1_lambda_par=LAMBDA,
            SD1WatsonDistributed_1_G2Zeppelin_1_lambda_perp=0.7e-9,
            SD1WatsonDistributed_1_partial_volume_0=0.7,  # Stick vf inside Watson
            G1Ball_1_lambda_iso=3.0e-9,
            partial_volume_0=0.7,   # Watson WM vf in MC model
            partial_volume_1=0.3,
        )

    def test_mc_noddi_jax_forward(self, scheme):
        mc     = self._make_noddi(scheme)
        params = self._noddi_params(mc)
        fn = build_mc_forward_fn(mc, scheme)
        E  = fn(jnp.array(params))
        assert E.shape == (scheme.number_of_measurements,)
        assert jnp.all(E >= 0.0)
        assert jnp.isfinite(E).all()

    def test_noddi_grad_finite(self, scheme):
        mc     = self._make_noddi(scheme)
        params = self._noddi_params(mc)
        fn   = build_mc_forward_fn(mc, scheme)
        p    = jnp.array(params)

        def loss(x):
            return jnp.mean(fn(x) ** 2)

        g = jax.grad(loss)(p)
        assert jnp.isfinite(g).all()


# ---------------------------------------------------------------------------
# SD2Bingham distributed
# ---------------------------------------------------------------------------

class TestBinghamStick:
    def _make(self, scheme):
        stick  = C1Stick()
        bstick = SD2BinghamDistributed(models=[stick])
        mc     = MultiCompartmentModel(models=[bstick])
        params = mc.parameters_to_parameter_vector(
            SD2BinghamDistributed_1_SD2Bingham_1_mu=MU,
            SD2BinghamDistributed_1_SD2Bingham_1_odi=0.2,
            SD2BinghamDistributed_1_SD2Bingham_1_psi=np.pi / 6,
            SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction=0.3,
            SD2BinghamDistributed_1_C1Stick_1_lambda_par=LAMBDA,
        )
        E_np = mc.simulate_signal(scheme, params)
        return mc, params, E_np

    def test_jax_forward_matches_numpy(self, scheme):
        mc, params, E_np = self._make(scheme)
        fn    = build_mc_forward_fn(mc, scheme)
        E_jax = np.array(fn(jnp.array(params)))
        assert_allclose(E_jax, E_np, rtol=5e-3, atol=5e-4,
                        err_msg="JAX Bingham-Stick deviates >0.5% from numpy")

    def test_jit_compiles(self, scheme):
        mc, params, _ = self._make(scheme)
        fn = jax.jit(build_mc_forward_fn(mc, scheme))
        E  = fn(jnp.array(params))
        assert jnp.isfinite(E).all()

    def test_grad_finite(self, scheme):
        mc, params, E_np = self._make(scheme)
        fn     = build_mc_forward_fn(mc, scheme)
        target = jnp.array(E_np)

        def loss(p):
            return jnp.mean((fn(p) - target) ** 2)

        g = jax.grad(loss)(jnp.array(params))
        assert jnp.isfinite(g).all()

    def test_solver_jax_fits(self, scheme):
        mc, params, E_np = self._make(scheme)
        result    = mc.fit(scheme, E_np[None], solver='jax')
        fitted_lam = float(np.squeeze(result.fitted_parameters[
            'SD2BinghamDistributed_1_C1Stick_1_lambda_par']))
        assert_allclose(fitted_lam, LAMBDA, rtol=0.15)
