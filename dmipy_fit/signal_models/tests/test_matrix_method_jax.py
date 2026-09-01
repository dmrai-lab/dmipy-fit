"""JAX path for the exact matrix-method compartments: parity with the NumPy path, and differentiability
w.r.t. the fitted geometry (the point of a GPU fitting backend). Skipped if JAX is unavailable."""
import numpy as np
import numpy.testing as npt
import pytest

import os   # noqa: E402
os.environ.setdefault("JAX_ENABLE_X64", "1")   # best-effort reference precision (before jax init)
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp   # noqa: E402

# Parity bound covering either precision: dmipy initialises jax x64-OFF at import, so the batch path may
# run complex64 here (max diff ~2e-5 vs the numpy float64 path); production fitting enables x64 globally
# (multicompartment_jax / vmap_fit) and matches to ~1e-7. 5e-5 is robust in both, and 20x under eps=1e-3.
_PARITY_ATOL = 5e-5

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme   # noqa: E402
from dmipy_fit.signal_models.cylinder_models import C5CylinderMatrixMethod   # noqa: E402
from dmipy_fit.signal_models.sphere_models import S5SphereMatrixMethod   # noqa: E402
from dmipy_fit.signal_models.plane_models import P5PlaneMatrixMethod   # noqa: E402

MU = [np.pi / 2, 0.0]


def _scheme(direction):
    b = np.array([0.0, 5e8, 1e9, 2e9, 5e9])
    return AcquisitionScheme.from_pgse(b, np.tile(direction, (len(b), 1)), 10e-3, 40e-3)


def test_jax_matches_numpy_all_shapes():
    "The JAX batch path reproduces the NumPy path to well within the eps=1e-3 reference tolerance."
    cyl_s = _scheme([0, 0, 1.])
    sph_s = _scheme([1., 0, 0])
    for E_np, E_jx in (
            (C5CylinderMatrixMethod(mu=MU, lambda_par=1.7e-9, diameter=8e-6)(cyl_s, use_jax=False),
             C5CylinderMatrixMethod(mu=MU, lambda_par=1.7e-9, diameter=8e-6)(cyl_s, use_jax=True)),
            (S5SphereMatrixMethod(diameter=10e-6)(sph_s, use_jax=False),
             S5SphereMatrixMethod(diameter=10e-6)(sph_s, use_jax=True)),
            (P5PlaneMatrixMethod(diameter=10e-6)(sph_s, use_jax=False),
             P5PlaneMatrixMethod(diameter=10e-6)(sph_s, use_jax=True))):
        npt.assert_allclose(E_jx, E_np, atol=_PARITY_ATOL)


def test_jax_signal_differentiable_in_diameter():
    "d(signal)/d(diameter) is finite and has the physically correct sign (larger pore -> more decay)."
    from dmipy_fit.signal_models._restricted_matrix import _unit_modes
    from dmipy_fit.jax.signal_models_jax import matrix_restricted_signal_jax
    lam, beta, U = _unit_modes("sphere", 16)
    lam, beta, U = jnp.asarray(lam), jnp.asarray(beta), jnp.asarray(U)
    # a single strong PGSE schedule on a uniform grid
    n_t, dt = 400, 1e-4
    g = np.zeros(n_t); g[:50] = 0.3; g[200:250] = -0.3
    g = jnp.asarray(g)
    gamma, D = 267.513e6, 2e-9

    def E_of_R(R):
        return matrix_restricted_signal_jax(g, dt, D, R, gamma, lam, beta, U)

    val, grad = jax.value_and_grad(E_of_R)(5e-6)
    assert np.isfinite(float(val)) and np.isfinite(float(grad))
    assert 0.0 < float(val) <= 1.0 + 1e-9
    assert float(grad) < 0.0            # a larger sphere restricts less -> lower signal -> dE/dR < 0
