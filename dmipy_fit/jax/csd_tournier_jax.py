"""The Tournier 2007 constrained spherical deconvolution, batched over voxels on the JAX device: the iteration of
:class:`~dmipy_fit.optimizers_fod.csd_tournier_batch.CsdTournierBatch` as ``jit(vmap(solve_one))`` with every
matrix a traced argument, so one compiled program per shape serves every response and every scheme of that shape.

Every matrix product in the kernel is at ``Precision.HIGHEST``: a float32 matmul on a CUDA device is TF32
otherwise (a 10-bit mantissa), which put the DiSCo FODs 5e-4 off the float64 reference and 18 % of voxels on a
different active-set path; the iteration itself is not that sensitive (a 1e-7 input perturbation moves 0.01 %
of voxels). Float32 on the device by default; ``dtype=np.float64`` when ``jax_enable_x64`` is on.
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp

from ..optimizers_fod.csd_tournier_batch import CsdTournierBatch, SPHERE_JACOBIAN

__all__ = ['CsdTournierJaxOptimizer']


@functools.lru_cache(maxsize=16)
def _compiled_solve(n_meas, n_coef, n_init, unity_constraint, max_iter, dtype_name):
    """The batched solver for one shape: ``(signals (b, n_meas), A, ATA_reg, L, P_init, lambda_pos, tau) ->
    (f (b, n_coef), iterations (b,))``, jitted, every matrix a traced argument."""
    dtype = jnp.dtype(dtype_name)
    jac = jnp.asarray(SPHERE_JACOBIAN, dtype)
    hi = jax.lax.Precision.HIGHEST

    def solve_one(s, A, ATA_reg, L, P_init, lambda_pos, tau):
        b = jnp.dot(A.T, s, precision=hi)
        f = jnp.zeros(n_coef, dtype).at[:n_init].set(jnp.dot(P_init, s, precision=hi))
        if unity_constraint:
            f = f.at[0].set(jac)
        thr = tau * L[0, 0] * f[0]
        neg = jnp.dot(L, f, precision=hi) < thr

        def cond(c):
            _, _, it, done = c
            return jnp.logical_and(jnp.logical_not(done), it < max_iter)

        def body(c):
            _, neg, it, _ = c
            Q = ATA_reg + lambda_pos * jnp.dot(L.T * neg, L, precision=hi)    # A'A + lambda_lb R + lambda_pos L' diag(neg) L
            f_new = jnp.linalg.solve(Q, b)
            if unity_constraint:
                f_new = f_new.at[0].set(jac)
            neg_new = jnp.dot(L, f_new, precision=hi) < thr
            return f_new, neg_new, it + 1, jnp.all(neg_new == neg)

        f, neg, it, done = jax.lax.while_loop(cond, body, (f, neg, jnp.int32(0), jnp.bool_(False)))
        return f, it

    return jax.jit(jax.vmap(solve_one, in_axes=(0, None, None, None, None, None, None)))


class CsdTournierJaxOptimizer(CsdTournierBatch):
    """The batched Tournier 2007 CSD on the JAX device (``solver='csd_tournier07_jax'``); the constructor and
    ``fit_batch`` are :class:`~dmipy_fit.optimizers_fod.csd_tournier_batch.CsdTournierBatch`'s."""

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8,
                 lambda_pos=1., lambda_lb=5e-4, tau=0.1, max_iter=50,
                 unity_constraint=True, init_sh_order=4, dtype=np.float32):
        if np.dtype(dtype) == np.float64 and not jax.config.jax_enable_x64:
            raise ValueError("dtype=float64 needs jax_enable_x64; the default float32 does not")
        super().__init__(acquisition_scheme, model, x0_vector, sh_order, lambda_pos, lambda_lb, tau, max_iter,
                         unity_constraint, init_sh_order, dtype)
        d = self.dtype
        self._args = tuple(jnp.asarray(a, d) for a in (self.A, self.ATA_reg, self.L_positivity, self.P_init))
        self._scalars = (jnp.asarray(self.lambda_pos, d), jnp.asarray(self.tau, d))
        self._solve = _compiled_solve(self.A.shape[0], self.Ncoef, self.Ncoef_init, self.unity_constraint,
                                      self.max_iter, self.dtype.name)

    def _solve_chunk(self, signals):
        f, it = self._solve(jnp.asarray(signals, self.dtype), *self._args, *self._scalars)
        return np.asarray(f), np.asarray(it)
