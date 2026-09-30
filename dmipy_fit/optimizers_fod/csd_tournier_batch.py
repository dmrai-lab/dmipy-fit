"""The Tournier 2007 constrained spherical deconvolution batched over voxels: what the device solvers share.

The iteration of :class:`~dmipy_fit.optimizers_fod.csd_tournier.CsdTournierOptimizer` (and of MRtrix's
``dwi2fod csd``), solved for every voxel of an image at once::

    f_0      = pinv(A[:, :n_init]) s                      the unconstrained low-order fit
    thr      = tau * L[0, 0] * f_0[0]                      the amplitude below which the FOD counts as negative
    repeat:  neg  = L f < thr
             f    = (A'A + lambda_lb R + lambda_pos L' diag(neg) L)^{-1} A' s
    until the set ``neg`` stops changing, or ``max_iter`` solves.

``A`` (the convolution kernel), ``R`` (the Laplace-Beltrami weights, ``diag(l^2 (l+1)^2)``) and ``L`` (the basis
on the positivity hemisphere) are the same for every voxel of a fixed-kernel model; only the signal ``s`` and the
set ``neg`` are the voxel's. One solve of the ``n_coef x n_coef`` system per iteration, a direct method, so a voxel
terminates in a handful of iterations on a criterion that is exact (the active set is stable), where a first-order
QP solver approaches the same optimum along a ``1/k`` tail. With ``unity_constraint`` the first coefficient is held
at ``1 / (2 sqrt(pi))`` and never solved for, as in the reference.

:class:`CsdTournierBatch` builds the matrices, locates ``sh_coeff`` in the parameter vector and feeds the image in
chunks of ``DMIPY_CSD_TOURNIER_BATCH`` voxels (default the solver's ``DEFAULT_BATCH``: 4096, one compiled shape, for
JAX; 16384 for torch), zero-padded to the chunk; a device solver subclasses it
and implements :meth:`_solve_chunk` (the JAX one in :mod:`dmipy_fit.jax.csd_tournier_jax`, the torch one in
:mod:`dmipy_fit.torch.csd_tournier_torch`). Every matrix product in a solver runs at full float32 precision: a
float32 matmul on a CUDA device is TF32 by default (a 10-bit mantissa), which put the DiSCo FODs 5e-4 off the float64
reference and 18 % of voxels on a different active-set path.
"""
import os

import numpy as np

from ..utils.sh_basis import positivity_basis, sh_degrees

__all__ = ['CsdTournierBatch', 'SPHERE_JACOBIAN']

SPHERE_JACOBIAN = 1.0 / (2.0 * np.sqrt(np.pi))


class CsdTournierBatch:
    """The batched Tournier 2007 CSD of :class:`~dmipy_fit.optimizers_fod.csd_tournier.CsdTournierOptimizer`:
    the same constructor, the same ``__call__``, plus :meth:`fit_batch`; the solve itself is the subclass's.

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
        The device dtype (default float32).
    """
    DEFAULT_BATCH = 4096

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
        self.sphere_jacobian = SPHERE_JACOBIAN
        self.dtype = np.dtype(dtype)

        if not hasattr(model, 'volume_fractions_fixed'):
            model._check_if_kernel_parameters_are_fixed()
        if not self.model.volume_fractions_fixed:
            raise ValueError("This CSD optimizer cannot estimate volume fractions.")
        if not hasattr(model, 'S0_responses'):
            raise AttributeError("model.S0_responses must be set before constructing the batched Tournier "
                                 "optimizer; fit() sets it, or set it manually.")

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

    def _solve_chunk(self, signals):
        """``(f (b, n_coef), iterations (b,))`` for one zero-padded chunk of signals ``(b, n_meas)``, numpy in and
        out: the subclass's device solve."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def fit_batch(self, data_all, x0_all, eta=None, *, diagnostics=False):
        """Fit every voxel: ``(N_voxels, N_parameters)``, or with ``diagnostics`` also ``{'iter_num': (N_voxels,)}``,
        the solves each voxel took (``max_iter`` where the active set never settled).

        ``data_all`` is ``(N_voxels, N_meas)`` normalised signal; ``x0_all`` ``(N_voxels, N_parameters)``, whose
        non-``sh_coeff`` entries are carried into the result unchanged. ``eta`` applies the Rician bias correction
        ``sqrt(max(s^2 - eta^2, 0))`` before the solve. Voxels are solved in chunks of ``DMIPY_CSD_TOURNIER_BATCH``
        (default ``DEFAULT_BATCH``), the last chunk zero-padded (a zero signal converges in one iteration to a zero FOD).
        """
        data_all = np.asarray(data_all, float)
        x0_all = np.asarray(x0_all, float)
        n = data_all.shape[0]
        if eta is not None and eta > 0:
            data_all = np.sqrt(np.maximum(data_all ** 2 - eta ** 2, 0.0))
        batch = max(1, min(int(os.environ.get("DMIPY_CSD_TOURNIER_BATCH", self.DEFAULT_BATCH)), n))
        f_all = np.zeros((n, self.Ncoef), float)
        iters = np.zeros(n, int)
        for s in range(0, n, batch):
            chunk = data_all[s:s + batch]
            m = chunk.shape[0]
            if m < batch:
                chunk = np.concatenate([chunk, np.zeros((batch - m, chunk.shape[1]))], axis=0)
            f, it = self._solve_chunk(chunk)
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
