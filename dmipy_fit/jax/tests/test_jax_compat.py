"""Phase 0 tests: JAX compatibility layer."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from dmipy_fit.jax.jax_compat import (
    jax_available,
    scheme_to_jax,
    unitsphere2cart_1d_jax,
)
from dmipy_fit.utils.utils import unitsphere2cart_1d
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme


@pytest.fixture(scope='module')
def hcp_scheme():
    return wu_minn_hcp_acquisition_scheme()


def test_jax_available():
    assert jax_available


def test_scheme_to_jax_keys(hcp_scheme):
    jax_scheme = scheme_to_jax(hcp_scheme)
    assert 'bvalues' in jax_scheme
    assert 'gradient_directions' in jax_scheme


def test_scheme_to_jax_values_match(hcp_scheme):
    jax_scheme = scheme_to_jax(hcp_scheme)
    assert_allclose(np.array(jax_scheme['bvalues']), hcp_scheme.bvalues)
    assert_allclose(
        np.array(jax_scheme['gradient_directions']),
        hcp_scheme.gradient_directions,
    )


def test_scheme_to_jax_arrays_are_jax(hcp_scheme):
    jax_scheme = scheme_to_jax(hcp_scheme)
    for key, val in jax_scheme.items():
        assert isinstance(val, jnp.ndarray), f"{key} is not a JAX array"


def test_unitsphere2cart_1d_jax_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu = rng.uniform([0., -np.pi], [np.pi, np.pi])
        np_result = unitsphere2cart_1d(mu)
        jax_result = np.array(unitsphere2cart_1d_jax(jnp.array(mu)))
        assert_allclose(jax_result, np_result, rtol=1e-5, atol=1e-6)


def test_unitsphere2cart_1d_jax_is_unit_vector():
    mu = jnp.array([np.pi / 4, np.pi / 3])
    v = unitsphere2cart_1d_jax(mu)
    assert_allclose(float(jnp.linalg.norm(v)), 1.0, rtol=1e-6)


def test_unitsphere2cart_1d_jax_is_jittable():
    f = jax.jit(unitsphere2cart_1d_jax)
    mu = jnp.array([0.5, 1.0])
    result = f(mu)
    assert result.shape == (3,)
