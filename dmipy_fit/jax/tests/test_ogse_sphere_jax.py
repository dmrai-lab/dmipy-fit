"""Phase 4 tests: JAX waveform path for S4 sphere model (OGSE).

Tests:
1. test_jax_ogse_matches_numpy — JAX path vs numpy path for cosine OGSE scheme,
   within 0.5%.
2. test_jax_pgse_matches_numpy — JAX path on PGSE waveform matches numpy
   Murday-Cotts path (backward compat).
3. test_jax_ogse_jit_compilable — JAX path compiles without Python control flow.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

pytest.importorskip('jax')

import jax
import jax.numpy as jnp

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.constants import CONSTANTS
from dmipy_fit.signal_models.sphere_models import S4SphereGaussianPhaseApproximation
from dmipy_fit.jax.signal_models_jax import s4sphere_ogse_signal_jax


GAMMA = CONSTANTS['water_gyromagnetic_ratio']


def _make_bvecs(n_m):
    return np.tile(np.r_[1., 0., 0.], (n_m, 1))


# ---------------------------------------------------------------------------
# 1. JAX OGSE matches numpy
# ---------------------------------------------------------------------------

def test_jax_ogse_matches_numpy():
    """JAX waveform path matches numpy path for cosine OGSE within 0.5%."""
    D = 1.7e-9
    diameter = 10e-6
    f = 50.0
    sigma = 0.04
    n_t = 2000

    bvalues = np.array([0, 5e8, 1e9, 2e9])
    bvecs = _make_bvecs(4)
    scheme = AcquisitionScheme.from_ogse(
        bvalues, bvecs,
        oscillation_frequency=f,
        gradient_duration=sigma,
        n_t=n_t)

    s4 = S4SphereGaussianPhaseApproximation(
        diameter=diameter, diffusion_constant=D)

    # Numpy path
    E_numpy = s4(scheme)

    # JAX path via s4sphere_ogse_signal_jax
    roots_jax = jnp.array(s4.SPHERE_TRASCENDENTAL_ROOTS)
    G_waveform = jnp.array(scheme._G)   # (n_m, n_t, 3)
    dt = float(scheme._dt)

    def _single(G_m):
        return s4sphere_ogse_signal_jax(
            G_m, dt, diameter, D, roots_jax, float(GAMMA))

    E_jax = np.array(jax.vmap(_single)(G_waveform))

    # b=0 rows: both should be 1
    assert_allclose(E_jax[0], 1.0, atol=1e-5)
    # Non-zero b: match within 0.5%
    assert_allclose(
        E_jax[1:], E_numpy[1:], rtol=0.005,
        err_msg="JAX OGSE path deviates from numpy path by > 0.5%")


# ---------------------------------------------------------------------------
# 2. JAX PGSE waveform matches numpy Murday-Cotts
# ---------------------------------------------------------------------------

def test_jax_pgse_matches_numpy():
    """JAX waveform path on PGSE waveform matches numpy Murday-Cotts within 1%.

    PGSE AcquisitionScheme stores the waveform via from_pgse; the JAX path
    reads G_waveform directly, so it uses the discretised PGSE waveform.
    Error vs Murday-Cotts is from discretisation (~0.6% at n_t=2000).
    """
    D = 1.7e-9
    diameter = 10e-6
    delta = 0.02
    Delta = 0.04
    n_t = 2000

    bvalues = np.array([0, 5e8, 1e9, 2e9])
    bvecs = _make_bvecs(4)
    scheme = AcquisitionScheme.from_pgse(
        bvalues, bvecs, delta=delta, Delta=Delta, n_t=n_t)

    s4 = S4SphereGaussianPhaseApproximation(
        diameter=diameter, diffusion_constant=D)

    # Numpy Murday-Cotts reference
    E_numpy = s4(scheme)

    # JAX waveform path
    roots_jax = jnp.array(s4.SPHERE_TRASCENDENTAL_ROOTS)
    G_waveform = jnp.array(scheme._G)
    dt = float(scheme._dt)

    def _single(G_m):
        return s4sphere_ogse_signal_jax(
            G_m, dt, diameter, D, roots_jax, float(GAMMA))

    E_jax = np.array(jax.vmap(_single)(G_waveform))

    # b=0: both are 1
    assert_allclose(E_jax[0], 1.0, atol=1e-5)
    # Non-zero b: within 1% (discretisation error is ~0.6%)
    assert_allclose(
        E_jax[1:], E_numpy[1:], rtol=0.01,
        err_msg="JAX PGSE waveform path deviates from Murday-Cotts by > 1%")


# ---------------------------------------------------------------------------
# 3. JIT compilable
# ---------------------------------------------------------------------------

def test_jax_ogse_jit_compilable():
    """s4sphere_ogse_signal_jax is JIT-compilable (no Python control flow)."""
    diameter = 10e-6
    D = 1.7e-9
    f = 50.0
    sigma = 0.04
    n_t = 100   # small for fast JIT

    roots_jax = jnp.array(
        S4SphereGaussianPhaseApproximation.SPHERE_TRASCENDENTAL_ROOTS)

    bvalues = np.array([1e9])
    bvecs = np.atleast_2d(np.r_[1., 0., 0.])
    scheme = AcquisitionScheme.from_ogse(
        bvalues, bvecs,
        oscillation_frequency=f,
        gradient_duration=sigma,
        n_t=n_t)

    G_m = jnp.array(scheme._G[0])  # (n_t, 3)
    dt = float(scheme._dt)

    # JIT compile
    fn_jit = jax.jit(
        lambda G: s4sphere_ogse_signal_jax(
            G, dt, diameter, D, roots_jax, float(GAMMA)))

    E = fn_jit(G_m)
    E_val = float(E)
    assert 0 < E_val <= 1.0 + 1e-6, (
        f"JIT-compiled JAX OGSE E={E_val:.4f} not in (0, 1]")
