"""Regression: vmapped anisotropic JAX signal fns must be correct when an inner
parameter is a closed-over constant (i.e. a fixed/linked parameter).

The orientation projection used ``jnp.dot(gradient_directions, mu_cart)`` -- a
matmul with contracting dimension 3. Under ``jax.vmap`` + ``jit`` the XLA GPU
autotuner miscompiles such a small-contracting-dim matmul ("Too small divisible
part of the contracting dimension"), and crucially picks a *different* (wrong)
config when the diffusivity is a baked constant versus a traced argument -- so a
dispersed model with FIXED diffusivities (e.g. the kernel calibration, or any
set_fixed_parameter fit) silently returned garbage, while the all-free fit was
fine. Replacing the matmul with an elementwise ``sum`` over the last axis avoids
the matmul entirely and is correct under vmap.

This pins the invariant: at b=0 the signal is 1 regardless of diffusivity, and
the constant-parameter result equals the traced-parameter result bit-for-bit.
"""
import numpy as np
import numpy.testing as npt
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy_fit.jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
from dmipy_fit.jax.multicompartment_jax import (
    _c1stick_jax_fn, _g2zeppelin_jax_fn)

scheme = wu_minn_hcp_acquisition_scheme()
scheme.TE = 0.08
sj = scheme_to_jax(scheme)
b0 = scheme.b0_mask
MU = jnp.asarray(np.random.RandomState(0).rand(48, 2))   # 48 orientations


def _vmap_traced(fn, **fixed):
    return jax.jit(lambda p: jax.vmap(lambda m: fn(sj, dict(p, mu=m)))(MU))

def _vmap_const(fn, **fixed):
    return jax.jit(lambda: jax.vmap(lambda m: fn(sj, dict(fixed, mu=m)))(MU))


def test_stick_vmap_const_param_matches_traced():
    tr = np.asarray(_vmap_traced(_c1stick_jax_fn)(
        {'lambda_par': jnp.asarray(1.7e-9)}))
    co = np.asarray(_vmap_const(_c1stick_jax_fn, lambda_par=jnp.asarray(1.7e-9))())
    npt.assert_allclose(co, tr, atol=1e-12)
    npt.assert_allclose(co[:, b0], 1.0, atol=1e-12)      # b=0 -> 1, all orientations


def test_zeppelin_vmap_const_param_matches_traced():
    p = {'lambda_par': jnp.asarray(1.7e-9), 'lambda_perp': jnp.asarray(0.45e-9)}
    tr = np.asarray(_vmap_traced(_g2zeppelin_jax_fn)(p))
    co = np.asarray(_vmap_const(_g2zeppelin_jax_fn, **p)())
    npt.assert_allclose(co, tr, atol=1e-12)
    npt.assert_allclose(co[:, b0], 1.0, atol=1e-12)
