"""Phase 2 tests: JIT-compiled multi-compartment forward factory."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


# ---------------------------------------------------------------------------
# Single-compartment models
# ---------------------------------------------------------------------------

class TestSingleCompartment:
    def test_g1ball_single(self, scheme):
        ball = G1Ball()
        mc = MultiCompartmentModel(models=[ball])
        params = mc.parameters_to_parameter_vector(G1Ball_1_lambda_iso=1.7e-9)

        E_np = mc.simulate_signal(scheme, params)
        forward_fn = build_mc_forward_fn(mc, scheme)
        E_jax = np.array(forward_fn(jnp.array(params)))

        assert_allclose(E_jax, E_np, rtol=1e-6)

    def test_c1stick_single(self, scheme):
        stick = C1Stick()
        mc = MultiCompartmentModel(models=[stick])
        mu = np.array([np.pi / 4, np.pi / 3])
        params = mc.parameters_to_parameter_vector(
            C1Stick_1_mu=mu, C1Stick_1_lambda_par=1.7e-9)

        E_np = mc.simulate_signal(scheme, params)
        forward_fn = build_mc_forward_fn(mc, scheme)
        E_jax = np.array(forward_fn(jnp.array(params)))

        assert_allclose(E_jax, E_np, rtol=1e-6)

    def test_g2zeppelin_single(self, scheme):
        zep = G2Zeppelin()
        mc = MultiCompartmentModel(models=[zep])
        mu = np.array([np.pi / 4, np.pi / 3])
        params = mc.parameters_to_parameter_vector(
            G2Zeppelin_1_mu=mu,
            G2Zeppelin_1_lambda_par=1.7e-9,
            G2Zeppelin_1_lambda_perp=0.5e-9)

        E_np = mc.simulate_signal(scheme, params)
        forward_fn = build_mc_forward_fn(mc, scheme)
        E_jax = np.array(forward_fn(jnp.array(params)))

        assert_allclose(E_jax, E_np, rtol=1e-3)


# ---------------------------------------------------------------------------
# Multi-compartment (ball + stick)
# ---------------------------------------------------------------------------

class TestBallAndStick:
    def _make_model_and_params(self, scheme):
        ball = G1Ball()
        stick = C1Stick()
        mc = MultiCompartmentModel(models=[ball, stick])
        mu = np.array([np.pi / 4, np.pi / 3])
        params = mc.parameters_to_parameter_vector(
            G1Ball_1_lambda_iso=1.7e-9,
            C1Stick_1_mu=mu,
            C1Stick_1_lambda_par=1.7e-9,
            partial_volume_0=0.3,
            partial_volume_1=0.7,
        )
        return mc, params

    def test_ball_stick_matches_numpy(self, scheme):
        mc, params = self._make_model_and_params(scheme)
        E_np = mc.simulate_signal(scheme, params)
        forward_fn = build_mc_forward_fn(mc, scheme)
        E_jax = np.array(forward_fn(jnp.array(params)))
        assert_allclose(E_jax, E_np, rtol=1e-6)

    def test_jit_compiles(self, scheme):
        mc, params = self._make_model_and_params(scheme)
        forward_fn = build_mc_forward_fn(mc, scheme)
        # Second call should use cached compilation
        E1 = forward_fn(jnp.array(params))
        E2 = forward_fn(jnp.array(params))
        assert_allclose(np.array(E1), np.array(E2), rtol=1e-10)

    def test_gradient_flows_through(self, scheme):
        mc, params = self._make_model_and_params(scheme)
        target = jnp.array(mc.simulate_signal(scheme, params))
        forward_fn = build_mc_forward_fn(mc, scheme)

        loss = lambda p: jnp.mean((forward_fn(p) - target) ** 2)
        grad = jax.grad(loss)(jnp.array(params))

        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_vf_sum_respected(self, scheme):
        """Weighted sum with vf=0.3/0.7 must match manual computation."""
        ball = G1Ball()
        stick = C1Stick()
        mu = np.array([np.pi / 4, np.pi / 3])

        mc_ball = MultiCompartmentModel(models=[G1Ball()])
        mc_stick = MultiCompartmentModel(models=[C1Stick()])
        mc_bs = MultiCompartmentModel(models=[ball, stick])

        p_ball = mc_ball.parameters_to_parameter_vector(G1Ball_1_lambda_iso=1.7e-9)
        p_stick = mc_stick.parameters_to_parameter_vector(
            C1Stick_1_mu=mu, C1Stick_1_lambda_par=1.7e-9)

        E_ball = mc_ball.simulate_signal(scheme, p_ball)
        E_stick = mc_stick.simulate_signal(scheme, p_stick)
        E_manual = 0.3 * E_ball + 0.7 * E_stick

        p_bs = mc_bs.parameters_to_parameter_vector(
            G1Ball_1_lambda_iso=1.7e-9,
            C1Stick_1_mu=mu,
            C1Stick_1_lambda_par=1.7e-9,
            partial_volume_0=0.3,
            partial_volume_1=0.7,
        )
        forward_fn = build_mc_forward_fn(mc_bs, scheme)
        E_jax = np.array(forward_fn(jnp.array(p_bs)))
        assert_allclose(E_jax, E_manual, rtol=1e-6)


# ---------------------------------------------------------------------------
# Three-compartment model
# ---------------------------------------------------------------------------

class TestThreeCompartment:
    def test_ball_stick_zeppelin(self, scheme):
        ball = G1Ball()
        stick = C1Stick()
        zep = G2Zeppelin()
        mc = MultiCompartmentModel(models=[ball, stick, zep])
        mu = np.array([np.pi / 4, np.pi / 3])
        params = mc.parameters_to_parameter_vector(
            G1Ball_1_lambda_iso=1.7e-9,
            C1Stick_1_mu=mu,
            C1Stick_1_lambda_par=1.7e-9,
            G2Zeppelin_1_mu=mu,
            G2Zeppelin_1_lambda_par=1.7e-9,
            G2Zeppelin_1_lambda_perp=0.5e-9,
            partial_volume_0=0.2,
            partial_volume_1=0.5,
            partial_volume_2=0.3,
        )
        E_np = mc.simulate_signal(scheme, params)
        forward_fn = build_mc_forward_fn(mc, scheme)
        E_jax = np.array(forward_fn(jnp.array(params)))
        assert_allclose(E_jax, E_np, rtol=1e-3)


# ---------------------------------------------------------------------------
# Error on unsupported model
# ---------------------------------------------------------------------------

def test_unsupported_model_raises(scheme):
    # All shipped models are now JAX-ported, so the dispatch (keyed on the exact
    # model type) is exercised with a subclass whose type is not in the registry:
    # it has a valid signal-model interface but no JAX implementation, so
    # build_mc_forward_fn must raise.
    class _UnportedModel(G1Ball):
        pass
    mc = MultiCompartmentModel(models=[_UnportedModel()])
    with pytest.raises(NotImplementedError, match="No JAX implementation"):
        build_mc_forward_fn(mc, scheme)
