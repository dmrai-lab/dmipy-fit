"""JAX S4 sphere on multidimensional b-tensor encoding matches NumPy.

The JAX Gaussian-phase sphere previously projected the waveform onto its
x-component ("proxy"), which is wrong for a non-x-aligned or multidimensional
waveform. It now factorises over the three Cartesian components
(E = prod_i E_1D(G_i)), exactly like the NumPy engine, so b-tensor STE/PTE
schemes are handled correctly.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("dmipy_sim")

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models import sphere_models
from dmipy_fit.jax.jax_compat import scheme_to_jax
from dmipy_fit.jax.multicompartment_jax import _make_s4sphere_ogse_jax_fn

_b = np.repeat([0., 1e9, 2e9, 3e9], 2)
_d = np.random.RandomState(0).randn(len(_b), 3)
_d /= np.linalg.norm(_d, axis=1, keepdims=True)


@pytest.mark.parametrize('scheme_name', ['ste', 'pte'])
def test_jax_s4_matches_numpy_on_btensor(scheme_name):
    if scheme_name == 'ste':
        scheme = AcquisitionScheme.from_btensor_ste(_b, delta=0.02, Delta=0.02)
    else:
        scheme = AcquisitionScheme.from_btensor_pte(
            _b, plane_normal=[0, 0, 1.], delta=0.02, Delta=0.02)
    s4 = sphere_models.S4SphereGaussianPhaseApproximation()
    E_np = np.asarray(s4(scheme, diameter=8e-6))
    fn = _make_s4sphere_ogse_jax_fn(s4, scheme)
    E_jax = np.asarray(fn(scheme_to_jax(scheme), {'diameter': 8e-6}))
    np.testing.assert_allclose(E_jax, E_np, rtol=1e-4, atol=1e-5)
    assert np.all(E_jax > 0) and np.all(E_jax <= 1.0 + 1e-6)
