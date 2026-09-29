"""The Tournier 2007 constrained spherical deconvolution, batched over voxels on the JAX device.

The iteration of :class:`~dmipy_fit.optimizers_fod.csd_tournier.CsdTournierOptimizer` (and of MRtrix's
``dwi2fod csd``), solved for every voxel of an image at once::

    f_0      = pinv(A[:, :n_init]) s                      the unconstrained low-order fit
    thr      = tau * L[0, 0] * f_0[0]                      the amplitude below which the FOD counts as negative
    repeat:  neg  = L f < thr
             f    = (A'A + lambda_lb R + lambda_pos L' diag(neg) L)^{-1} A' s
    until the set ``neg`` stops changing, or ``max_iter`` solves.

Every matrix product in the kernel is at ``Precision.HIGHEST``: a float32 matmul on a CUDA device is TF32
otherwise (a 10-bit mantissa), which put the DiSCo FODs 5e-4 off the float64 reference and 18 % of voxels on a
different active-set path; the iteration itself is not that sensitive (a 1e-7 input perturbation moves 0.01 %
of voxels).

``A`` (the convolution kernel), ``R`` (the Laplace-Beltrami weights, ``diag(l^2 (l+1)^2)``) and ``L`` (the basis
on the positivity hemisphere) are the same for every voxel of a fixed-kernel model; only the signal ``s`` and the set ``neg``
are the voxel's. One solve of the ``n_coef x n_coef`` system per iteration, a direct method, so a voxel
terminates in a handful of iterations on a criterion that is exact (the active set is stable), where a
first-order QP solver approaches the same optimum along a ``1/k`` tail. With ``unity_constraint`` the first
coefficient is held at ``1 / (2 sqrt(pi))`` and never solved for, as in the reference.

The kernel is ``jit(vmap(solve_one))`` with every matrix a traced argument, so one compiled program per shape
serves every response and every scheme of that shape; the image is fed in chunks of ``DMIPY_CSD_TOURNIER_BATCH``
voxels (default 4096), zero-padded to the chunk. Float32 on the device by default; ``dtype=np.float64`` when
``jax_enable_x64`` is on.
"""
import functools
import os

import numpy as np
import jax
import jax.numpy as jnp

from ..utils.sh_basis import positivity_basis, sh_degrees

__all__ = ['CsdTournierJaxOptimizer']

_SPHERE_JACOBIAN = 1.0 / (2.0 * np.sqrt(np.pi))


@functools.lru_cache(maxsize=16)
def _compiled_solve(n_meas, n_coef, n_init, unity_constraint, max_iter, dtype_name):
    """The batched solver for one shape: ``(signals (b, n_meas), A, ATA_reg, L, P_init, lambda_pos, tau) ->
    (f (b, n_coef), iterations (b,))``, jitted, every matrix a traced argument."""
    dtype = jnp.dtype(dtype_name)
    jac = jnp.asarray(_SPHERE_JACOBIAN, dtype)
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


class CsdTournierJaxOptimizer:
    """The Tournier 2007 CSD of :class:`~dmipy_fit.optimizers_fod.csd_tournier.CsdTournierOptimizer`, batched
    over voxels on the JAX device: the same constructor, the same ``__call__``, plus :meth:`fit_batch`.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme
    model : MultiCompartmentSphericalHarmonicsModel
        With its volume fractions fixed (one convolution kernel); ``S0_responses`` set, as ``fit()`` does.
    x0_vector : array
        The parameter vector the kernel is built from (all-NaN ``sh_coeff`` for a fixed kernel).
    sh_order : int
        Spherical harmonics order of the FOD (default 8, 45 coefficients).
    lambda_pos : float
        Weight of the non-negativity penalty (default 1).
    lambda_lb : float
        Laplace-Beltrami smoothness weight (default 5e-4; ``fit()`` passes its own).
    tau : float
        The negativity threshold as a fraction of the mean FOD amplitude (default 0.1).
    max_iter : int
        Solves per voxel at most (default 50, MRtrix's ``-niter``).
    unity_constraint : bool
        Hold the first coefficient at ``1 / (2 sqrt(pi))`` so the FOD integrates to one.
    init_sh_order : int
        Order of the unconstrained fit the iteration starts from (default 4).
    dtype : numpy dtype
        The device dtype (default float32; float64 needs ``jax_enable_x64``).
    """
    _citations = {
        'definition': [
            {'key': 'tournier2007', 'authors': 'Tournier J-D, Calamante F, Connelly A',
             'title': 'Robust determination of the fibre orientation distribution in diffusion MRI: '
                      'non-negativity constrained super-resolved spherical deconvolution',
             'journal': 'NeuroImage', 'year': 2007, 'doi': '10.1016/j.neuroimage.2007.02.016'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'SH_convergence', 'name': 'SH convergence',
         'condition_human': 'max_order must be sufficient for the kernel bandwidth',
         'severity': 'info', 'source_key': 'tournier2007'},
    ]

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8,
                 lambda_pos=1., lambda_lb=5e-4, tau=0.1, max_iter=50,
                 unity_constraint=True, init_sh_order=4, dtype=np.float32):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = int(sh_order)
        self.Ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.Ncoef_init = int((init_sh_order + 2) * (init_sh_order + 1) // 2)
        self.lambda_pos = float(lambda_pos)
        self.lambda_lb = float(lambda_lb)
        self.tau = float(tau)
        self.max_iter = int(max_iter)
        self.unity_constraint = bool(unity_constraint)
        self.sphere_jacobian = _SPHERE_JACOBIAN
        self.dtype = np.dtype(dtype)
        if self.dtype == np.float64 and not jax.config.jax_enable_x64:
            raise ValueError("dtype=float64 needs jax_enable_x64; the default float32 does not")

        if not hasattr(model, 'volume_fractions_fixed'):
            model._check_if_kernel_parameters_are_fixed()
        if not self.model.volume_fractions_fixed:
            raise ValueError("This CSD optimizer cannot estimate volume fractions.")
        if not hasattr(model, 'S0_responses'):
            raise AttributeError("model.S0_responses must be set before constructing CsdTournierJaxOptimizer; "
                                 "fit() sets it, or set it manually.")

        self.L_positivity = positivity_basis(self.sh_order)
        sh_l = sh_degrees(self.sh_order)
        self.R_smoothness = np.diag(sh_l ** 2 * (sh_l + 1) ** 2)

        x0_single = np.reshape(x0_vector, (-1, np.shape(x0_vector)[-1]))[0]
        if not np.all(np.isnan(x0_single)):
            raise ValueError("the kernel must be fixed (an all-NaN x0 for its parameters); a voxel-varying "
                             "kernel has one system per voxel and is not what this optimizer batches")
        self._x0_single = x0_single
        self.A = self.model._construct_convolution_kernel(**self.model.parameter_vector_to_parameters(x0_single))
        self.ATA_reg = self.A.T @ self.A + self.lambda_lb * self.R_smoothness
        self.P_init = np.linalg.pinv(self.A[:, :self.Ncoef_init])
        self._sh_slice = self._locate_sh_coefficients()

        d = self.dtype
        self._args = tuple(jnp.asarray(a, d) for a in (self.A, self.ATA_reg, self.L_positivity, self.P_init))
        self._scalars = (jnp.asarray(self.lambda_pos, d), jnp.asarray(self.tau, d))
        self._solve = _compiled_solve(self.A.shape[0], self.Ncoef, self.Ncoef_init, self.unity_constraint,
                                      self.max_iter, self.dtype.name)

    def _locate_sh_coefficients(self):
        """The slice of the parameter vector that holds ``sh_coeff``, checked against the model's own assembly."""
        params = self.model.parameter_vector_to_parameters(self._x0_single)
        marker = np.arange(1, self.Ncoef + 1, dtype=float)
        params['sh_coeff'] = marker
        vec = np.asarray(self.model.parameters_to_parameter_vector(**params), float)
        start = int(np.flatnonzero(vec == marker[0])[0])
        sl = slice(start, start + self.Ncoef)
        if not np.array_equal(vec[sl], marker):
            raise RuntimeError("the model does not lay sh_coeff out contiguously in its parameter vector")
        return sl

    # ------------------------------------------------------------------
    def fit_batch(self, data_all, x0_all, eta=None, *, diagnostics=False):
        """Fit every voxel: ``(N_voxels, N_parameters)``, or with ``diagnostics`` also ``{'iter_num': (N_voxels,)}``,
        the solves each voxel took (``max_iter`` where the active set never settled).

        ``data_all`` is ``(N_voxels, N_meas)`` normalised signal; ``x0_all`` ``(N_voxels, N_parameters)``, whose
        non-``sh_coeff`` entries are carried into the result unchanged. ``eta`` applies the Rician bias correction
        ``sqrt(max(s^2 - eta^2, 0))`` before the solve. Voxels are solved in chunks of ``DMIPY_CSD_TOURNIER_BATCH``
        (default 4096), the last chunk zero-padded (a zero signal converges in one iteration to a zero FOD).
        """
        data_all = np.asarray(data_all, float)
        x0_all = np.asarray(x0_all, float)
        n = data_all.shape[0]
        if eta is not None and eta > 0:
            data_all = np.sqrt(np.maximum(data_all ** 2 - eta ** 2, 0.0))
        batch = max(1, min(int(os.environ.get("DMIPY_CSD_TOURNIER_BATCH", "4096")), n))
        f_all = np.zeros((n, self.Ncoef), float)
        iters = np.zeros(n, int)
        for s in range(0, n, batch):
            chunk = data_all[s:s + batch]
            m = chunk.shape[0]
            if m < batch:
                chunk = np.concatenate([chunk, np.zeros((batch - m, chunk.shape[1]))], axis=0)
            f, it = self._solve(jnp.asarray(chunk, self.dtype), *self._args, *self._scalars)
            f_all[s:s + m] = np.asarray(f)[:m]
            iters[s:s + m] = np.asarray(it)[:m]
        out = x0_all.copy()
        out[:, self._sh_slice] = f_all
        if diagnostics:
            return out, {'iter_num': iters}
        return out

    def __call__(self, data, x0_vector):
        """One voxel: the fitted parameter vector, as :class:`CsdTournierOptimizer` returns it."""
        return self.fit_batch(np.asarray(data)[None], np.asarray(x0_vector)[None])[0]
