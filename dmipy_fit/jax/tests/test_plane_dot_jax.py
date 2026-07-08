"""JAX forwards for the remaining constructor compartments: S1Dot, P2, P3.

Completes JAX coverage of the compartment compendium so any constructor model
can fit with solver='jax'. Each JAX forward is checked against its numpy twin.
"""
import numpy as np
import numpy.testing as npt
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy_fit.signal_models.sphere_models import S1Dot
from dmipy_fit.signal_models.plane_models import (
    P2PlaneStejskalTannerApproximation, P3PlaneCallaghanApproximation)
from dmipy_fit.jax.jax_compat import scheme_to_jax
from dmipy_fit.jax.multicompartment_jax import (
    _s1dot_jax_fn, _p2plane_jax_fn, _make_p3plane_jax_fn)

scheme = wu_minn_hcp_acquisition_scheme()
scheme.TE = 0.08
sj = scheme_to_jax(scheme)


def test_s1dot_jax_matches_numpy():
    npt.assert_allclose(np.asarray(_s1dot_jax_fn(sj, {})),
                        S1Dot()(scheme), atol=1e-12)


@pytest.mark.parametrize("d", [2e-6, 5e-6, 1e-5])
def test_p2plane_jax_matches_numpy(d):
    E_jax = np.asarray(_p2plane_jax_fn(sj, {'diameter': jnp.array(d)}))
    npt.assert_allclose(E_jax, P2PlaneStejskalTannerApproximation()(
        scheme, diameter=d), atol=1e-9)


@pytest.mark.parametrize("d", [2e-6, 5e-6, 1e-5])
def test_p3plane_jax_matches_numpy(d):
    p3 = P3PlaneCallaghanApproximation()
    E_jax = np.asarray(_make_p3plane_jax_fn(p3)(sj, {'diameter': jnp.array(d)}))
    npt.assert_allclose(E_jax, p3(scheme, diameter=d), atol=1e-9)
