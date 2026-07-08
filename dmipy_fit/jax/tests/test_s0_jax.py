"""Tests for per-compartment S0 scaling in the JAX forward path.

Phase 3b of the S0/PD gap fill: verify that S0_tissue_responses are applied
correctly in the JAX forward function and match the numpy path.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
jnp = jax.numpy

from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


def test_jax_s0_responses_match_numpy(scheme):
    """JAX forward with S0_tissue_responses should match numpy __call__."""
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(
        [stick, ball], S0_tissue_responses=[1.0, 3.0])

    params = {
        'C1Stick_1_lambda_par': 1.7e-9,
        'C1Stick_1_mu': [np.pi / 2, np.pi / 2],
        'G1Ball_1_lambda_iso': 3.0e-9,
        'partial_volume_0': 0.5,
        'partial_volume_1': 0.5,
    }

    signal_np = mcm(scheme, **params)

    forward_fn = build_mc_forward_fn(mcm, scheme)
    params_vec = mcm.parameters_to_parameter_vector(**params)
    signal_jax = np.array(forward_fn(jnp.array(params_vec)))

    assert_allclose(signal_jax, signal_np, rtol=1e-3)


def test_jax_no_s0_responses_is_identity(scheme):
    """Without S0_tissue_responses, JAX forward should match numpy (no scaling)."""
    stick = C1Stick()
    ball = G1Ball()
    mcm_no_s0 = MultiCompartmentModel([stick, ball])
    mcm_ones = MultiCompartmentModel(
        [C1Stick(), G1Ball()], S0_tissue_responses=[1.0, 1.0])

    params = {
        'C1Stick_1_lambda_par': 1.7e-9,
        'C1Stick_1_mu': [np.pi / 2, np.pi / 2],
        'G1Ball_1_lambda_iso': 3.0e-9,
        'partial_volume_0': 0.6,
        'partial_volume_1': 0.4,
    }

    fwd_no_s0 = build_mc_forward_fn(mcm_no_s0, scheme)
    fwd_ones = build_mc_forward_fn(mcm_ones, scheme)

    params_vec_no = mcm_no_s0.parameters_to_parameter_vector(**params)
    params_vec_ones = mcm_ones.parameters_to_parameter_vector(**params)

    sig_no = np.array(fwd_no_s0(jnp.array(params_vec_no)))
    sig_ones = np.array(fwd_ones(jnp.array(params_vec_ones)))

    assert_allclose(sig_no, sig_ones, rtol=1e-6)


def test_jax_s0_responses_effect(scheme):
    """S0 scaling should correctly weight each compartment's contribution."""
    # With S0_tissue_responses=[1, 5], ball signal should dominate
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(
        [stick, ball], S0_tissue_responses=[1.0, 5.0])

    params = {
        'C1Stick_1_lambda_par': 1.7e-9,
        'C1Stick_1_mu': [np.pi / 2, np.pi / 2],
        'G1Ball_1_lambda_iso': 3.0e-9,
        'partial_volume_0': 0.5,
        'partial_volume_1': 0.5,
    }

    forward_fn = build_mc_forward_fn(mcm, scheme)
    params_vec = mcm.parameters_to_parameter_vector(**params)
    signal_jax = np.array(forward_fn(jnp.array(params_vec)))

    # Build reference manually: rho_stick=1/5=0.2, rho_ball=5/5=1.0
    stick_sig = C1Stick()(scheme, lambda_par=1.7e-9,
                           mu=[np.pi / 2, np.pi / 2])
    ball_sig = G1Ball()(scheme, lambda_iso=3.0e-9)
    expected = 0.5 * 0.2 * stick_sig + 0.5 * 1.0 * ball_sig

    assert_allclose(signal_jax, expected, rtol=1e-3)
