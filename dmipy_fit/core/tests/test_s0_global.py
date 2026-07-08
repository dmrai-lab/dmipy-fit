"""Tests for S0_global fittable parameter.

Phase 4 of the S0/PD gap fill: verify that S0_global correctly scales the
signal in both numpy __call__ and JAX forward paths.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme

scheme = wu_minn_hcp_acquisition_scheme()


def test_s0_global_scales_signal():
    """S0_global should multiply the entire signal."""
    ball = G1Ball()
    mcm = MultiCompartmentModel([ball], S0_global=True)

    signal_no_s0 = mcm(scheme, G1Ball_1_lambda_iso=3e-9)
    signal_s0_1 = mcm(scheme, G1Ball_1_lambda_iso=3e-9, S0_global=1.0)
    signal_s0_half = mcm(scheme, G1Ball_1_lambda_iso=3e-9, S0_global=0.5)

    assert_allclose(signal_s0_1, signal_no_s0, rtol=1e-12)
    assert_allclose(signal_s0_half, 0.5 * signal_no_s0, rtol=1e-12)


def test_s0_global_nan_is_identity():
    """S0_global=NaN should not change the signal (identity)."""
    ball = G1Ball()
    mcm = MultiCompartmentModel([ball], S0_global=True)

    signal_ref = mcm(scheme, G1Ball_1_lambda_iso=3e-9)
    signal_nan = mcm(scheme, G1Ball_1_lambda_iso=3e-9, S0_global=np.nan)

    assert_allclose(signal_nan, signal_ref, rtol=1e-12)


def test_s0_global_before_eta():
    """S0_global should be applied BEFORE eta (Rician noise floor)."""
    ball = G1Ball()
    mcm = MultiCompartmentModel([ball], S0_global=True, eta=True)

    # Signal with S0=0.8, eta=0.1:
    # expected = sqrt((0.8 * E)^2 + eta^2)
    E = G1Ball()(scheme, lambda_iso=3e-9)
    S0_val = 0.8
    eta_val = 0.1
    expected = np.sqrt((S0_val * E) ** 2 + eta_val ** 2)

    signal = mcm(scheme, G1Ball_1_lambda_iso=3e-9,
                 S0_global=S0_val, eta=eta_val)
    assert_allclose(signal, expected, rtol=1e-10)


def test_s0_global_parameter_registered():
    """S0_global=True should register the parameter correctly."""
    mcm = MultiCompartmentModel([G1Ball()], S0_global=True)
    assert 'S0_global' in mcm.parameter_ranges
    assert mcm.parameter_ranges['S0_global'] == (0.001, 2.0)
    assert mcm.parameter_cardinality['S0_global'] == 1


def test_s0_global_not_registered_by_default():
    """S0_global=False (default) should NOT register the parameter."""
    mcm = MultiCompartmentModel([G1Ball()])
    assert 'S0_global' not in mcm.parameter_ranges


def _has_jax():
    try:
        import jax
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_jax(), reason="JAX not installed")
def test_s0_global_jax_matches_numpy():
    """JAX forward with S0_global should match numpy __call__."""
    import jax.numpy as jnp
    from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn

    ball = G1Ball()
    stick = C1Stick()
    mcm = MultiCompartmentModel([stick, ball], S0_global=True)

    params = {
        'C1Stick_1_lambda_par': 1.7e-9,
        'C1Stick_1_mu': [np.pi / 2, np.pi / 2],
        'G1Ball_1_lambda_iso': 3.0e-9,
        'partial_volume_0': 0.6,
        'partial_volume_1': 0.4,
        'S0_global': 0.75,
    }

    signal_np = mcm(scheme, **params)

    forward_fn = build_mc_forward_fn(mcm, scheme)
    params_vec = mcm.parameters_to_parameter_vector(**params)
    signal_jax = np.array(forward_fn(jnp.array(params_vec)))

    assert_allclose(signal_jax, signal_np, rtol=1e-3)
