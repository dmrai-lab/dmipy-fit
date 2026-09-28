"""JAX/OSQP-based Constrained Spherical Deconvolution optimizer.

Drop-in replacement for CsdCvxpyOptimizer that uses jaxopt.OSQP and
jax.vmap to solve all voxels in a single compiled kernel instead of a
Python for-loop of cvxpy problems.

QP formulation (same as CsdCvxpyOptimizer)
-------------------------------------------
  minimize   0.5 * x' Q x + c' x
  subject to  G x <= h          (positivity + VF non-negativity)
              A_eq x = b_eq     (unity VF, when unity_constraint=True)

where:
  Q    = 2 * (A_kernel.T @ A_kernel + lambda_lb * R_smoothness)   [static]
  c    = -2 * A_kernel.T @ signal                                  [per-voxel]
  G    = -[L_positivity_padded; vf_selector]                       [static]
  h    = zeros                                                      [static]
  A_eq = vf_unity_row                                              [static]
  b_eq = [1 / sphere_jacobian]                                     [static]

Only 'c' varies per voxel, which makes jax.vmap very efficient: Q, G, h, A_eq
and b_eq are shared across the batch (vmap in_axes=None) while 'c' -- derived
inside the kernel from the per-voxel signal -- is the batched axis. Q, AT, G,
h, A_eq, b_eq are TRACED ARGUMENTS of the compiled kernel, not baked-in
constants: that is what lets the same compiled program (see the process-
lifetime cache below) serve any concrete tissue response on an equal
acquisition scheme, not just the one that happened to be in scope when it
was first traced.
"""

import functools
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from ..utils.sh_basis import positivity_basis, sh_degrees

from jaxopt import OSQP


__all__ = ['CsdOsqpOptimizer', 'warm', 'pin_x64_off', 'X64PinnedError']


# ---------------------------------------------------------------------------
# Process-lifetime compile cache
# ---------------------------------------------------------------------------
#
# One compiled jit(vmap(fit_one)) per (scheme fingerprint, sh_order,
# unity_constraint, lambda_lb, maxiter, tol, dtype, batch) key, bounded LRU.
# The QP matrices (Q, AT, G, h, A_eq, b_eq) are TRACED ARGUMENTS of the
# returned callable, not compile-time constants closed over at trace time --
# that is what makes reuse across CsdOsqpOptimizer instances correct: two
# instances built from an equal scheme (same fingerprint) but a different
# tissue response (a different S0_responses / kernel A) still get the
# numerically right answer from the shared compiled program, because the
# response only changes the VALUES passed in at call time, never the
# structure that was traced. `batch` does not change the trace either (vmap
# is shape-polymorphic and jax's own per-shape cache handles a new batch
# size on the shared callable) -- it is kept in the key only so the cache's
# identity matches the one stated in dmipy-fit#32 and so warm() can report
# one row per declared (scheme, batch).
_KERNEL_CACHE_MAXSIZE = 16


@functools.lru_cache(maxsize=_KERNEL_CACHE_MAXSIZE)
def _compiled_fit_batch(scheme_fingerprint, sh_order, unity_constraint,
                         lambda_lb, maxiter, tol, dtype_name, batch):
    """Build (once per key) the jit+vmap OSQP kernel for one CSD shape.

    Returns a callable ``fit_one_batch(signal, Q, AT, G, h)`` (or, when
    ``unity_constraint``, ``fit_one_batch(signal, Q, AT, G, h, A_eq, b_eq)``)
    vmapped over the leading axis of ``signal`` only -- the QP matrices are
    shared (broadcast) across the batch.  ``scheme_fingerprint`` and ``batch``
    are part of the key (see module docstring above) but not read here;
    ``sh_order`` is likewise not read (it only shapes Q/AT/G/h, which are
    arguments, not captured).  Repeated calls with an equal key return the
    exact same Python callable, so a repeat ``.fit(solver='csd_jax')`` on an
    equal scheme pays no compile.
    """
    dtype = jnp.dtype(dtype_name)
    solver = OSQP(
        maxiter=maxiter,
        tol=tol,
        check_primal_dual_infeasability=False,  # jaxopt typo: "infeasability"
    )

    if unity_constraint:
        def fit_one(signal, Q, AT, G, h, A_eq, b_eq):
            c = jnp.array(-2.0, dtype=dtype) * (AT @ signal.astype(dtype))
            sol = solver.run(
                None,                  # init_params=None -> auto-init
                params_obj=(Q, c),
                params_eq=(A_eq, b_eq),
                params_ineq=(G, h),
            )
            return sol.params.primal  # KKTSolution.primal

        return jax.jit(jax.vmap(fit_one, in_axes=(0, None, None, None, None, None, None)))
    else:
        def fit_one(signal, Q, AT, G, h):
            c = jnp.array(-2.0, dtype=dtype) * (AT @ signal.astype(dtype))
            sol = solver.run(
                None,
                params_obj=(Q, c),
                params_ineq=(G, h),
            )
            return sol.params.primal

        return jax.jit(jax.vmap(fit_one, in_axes=(0, None, None, None, None)))


def compile_cache_info():
    """``functools.lru_cache``-style stats (hits, misses, maxsize, currsize) for the CSD compile cache."""
    return _compiled_fit_batch.cache_info()


def clear_compile_cache():
    """Drop every cached compiled kernel. Mainly for tests; a serving process never needs this."""
    _compiled_fit_batch.cache_clear()


# ---------------------------------------------------------------------------
# x64 pinning
# ---------------------------------------------------------------------------
#
# dmipy enables jax_enable_x64 globally whenever a cylinder fit needs it
# (multicompartment_jax.py, float64 Van Gelderen sums) and never turns it
# back off -- by design, since another cylinder fit may follow in the same
# process. CsdOsqpOptimizer.fit_batch's ordinary behaviour (unchanged, see
# below) is therefore to save the flag, force it off for the float32 OSQP
# solve, and restore it -- correct for a short-lived script, but wrong for a
# long-lived server: two concurrent requests racing that save/restore would
# stomp on each other's global JAX config, and every request still pays the
# save/restore. A serving process instead calls pin_x64_off() once at
# start-up (after warm()); from then on fit_batch REFUSES BY NAME
# (X64PinnedError) instead of silently saving/restoring if it finds x64 back
# on -- that is a bug in the server (something re-enabled a process-wide
# flag the server declared fixed), not a condition to paper over per call.
_x64_pinned_off = False


class X64PinnedError(RuntimeError):
    """Raised by :meth:`CsdOsqpOptimizer.fit_batch` when ``pin_x64_off()`` was called for this process but
    ``jax_enable_x64`` is on anyway -- something (e.g. a preceding cylinder fit) re-enabled a flag the
    process declared fixed. Fix the caller that flipped it; csd_jax will not silently save/restore around it."""


def pin_x64_off():
    """Declare, once, that this process runs CSD (and only float32-safe paths) for its lifetime.

    Turns ``jax_enable_x64`` off now and switches :meth:`CsdOsqpOptimizer.fit_batch` from its default
    save/restore-per-call behaviour to a refusal: any later call that finds ``jax_enable_x64`` back on
    raises :class:`X64PinnedError` by name instead of silently flipping it back off. Call this once at
    server start-up, before (or after) :func:`warm` -- not per request, and not in a process that also
    serves cylinder fits (those need x64 on and are incompatible with the pin).
    """
    global _x64_pinned_off
    jax.config.update("jax_enable_x64", False)
    _x64_pinned_off = True


def x64_pinned():
    """Whether :func:`pin_x64_off` has been called in this process."""
    return _x64_pinned_off


class CsdOsqpOptimizer:
    """JAX/OSQP multi-compartment CSD optimizer.
    """
    _citations = {
        'definition': [
            {'key': 'jeurissen2014', 'authors': 'Jeurissen B, Tournier J-D, Dhollander T, Connelly A, Sijbers J',
             'title': 'Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data',
             'journal': 'NeuroImage',
             'year': 2014, 'doi': '10.1016/j.neuroimage.2014.07.061'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'SH_convergence', 'name': 'SH convergence',
         'condition_human': 'max_order must be sufficient for the kernel bandwidth',
         'severity': 'info',
         'source_key': 'jeurissen2014'},
    ]
    r"""

    Matches the interface of CsdCvxpyOptimizer (same constructor signature
    and __call__ signature) and additionally exposes fit_batch() for
    GPU-parallel fitting of all voxels at once.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme
    model : MultiCompartmentSphericalHarmonicsModel
    x0_vector : array
        Initial parameter vector (used to build the convolution kernel).
    sh_order : int
        Spherical harmonics order (default 8).
    unity_constraint : bool
        Whether to constrain volume fractions to sum to 1.
    lambda_lb : float
        Laplace-Beltrami regularisation weight.
    maxiter : int
        Maximum OSQP iterations per voxel solve (default 4000).
    tol : float
        OSQP primal/dual tolerance (default 1e-4).
    """

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8,
                 unity_constraint=True, lambda_lb=0., maxiter=4000, tol=1e-4):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = sh_order
        self.Ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.Nmodels = len(self.model.models)
        self.lambda_lb = lambda_lb
        self.unity_constraint = unity_constraint
        self.sphere_jacobian = 2 * np.sqrt(np.pi)
        self.maxiter = maxiter
        self.tol = tol

        # Ensure model has volume_fractions_fixed set (normally called by fit())
        if not hasattr(model, 'volume_fractions_fixed'):
            model._check_if_kernel_parameters_are_fixed()

        # S0_responses must be set by the caller (e.g. fit() in
        # spherical_harmonics_framework.py) before constructing this optimizer,
        # because _construct_convolution_kernel multiplies each compartment
        # kernel by S0_responses[i].  A missing attribute indicates a caller bug.
        if not hasattr(model, 'S0_responses'):
            raise AttributeError(
                "model.S0_responses must be set before constructing "
                "CsdOsqpOptimizer. Call fit() or set S0_responses manually."
            )

        # --- positivity basis (same as cvxpy version) -----------------------
        self.L_positivity = positivity_basis(self.sh_order)

        # --- convolution kernel ---------------------------------------------
        x0_single_voxel = np.reshape(
            x0_vector, (-1, x0_vector.shape[-1]))[0]
        if np.all(np.isnan(x0_single_voxel)):
            self.single_convolution_kernel = True
            parameters_dict = self.model.parameter_vector_to_parameters(
                x0_single_voxel)
            self.A = self._construct_kernel(parameters_dict)
        else:
            self.single_convolution_kernel = False
            self.A = None

        # --- layout of the QP variable x ------------------------------------
        self.Ncoef_total = 0
        vf_array = []

        if self.model.volume_fractions_fixed:
            self.sh_start = 0
            self.Ncoef_total = self.Ncoef
            self.vf_indices = np.array([0])
        else:
            for m in self.model.models:
                if 'orientation' in m.parameter_types.values():
                    self.sh_start = self.Ncoef_total
                    sh_model = np.zeros(self.Ncoef)
                    sh_model[0] = 1
                    vf_array.append(sh_model)
                    self.Ncoef_total += self.Ncoef
                else:
                    vf_array.append(1)
                    self.Ncoef_total += 1
            self.vf_indices = np.where(np.hstack(vf_array))[0]

        # --- Laplace-Beltrami smoothness matrix -----------------------------
        sh_l = sh_degrees(sh_order)
        lb_weights = sh_l ** 2 * (sh_l + 1) ** 2
        if self.model.volume_fractions_fixed:
            self.R_smoothness = np.diag(lb_weights)
        else:
            diagonal = np.zeros(self.Ncoef_total)
            diagonal[self.sh_start: self.sh_start + self.Ncoef] = lb_weights
            self.R_smoothness = np.diag(diagonal)

        # --- precompute static QP matrices if kernel is fixed ---------------
        if self.single_convolution_kernel:
            self._build_qp_and_solver(self.A)

    def _construct_kernel(self, parameters_dict):
        """Build the (diffusion) observation matrix / convolution kernel."""
        return self.model._construct_convolution_kernel(
            acquisition_scheme=self.acquisition_scheme, **parameters_dict)

    # ------------------------------------------------------------------
    # QP matrix precomputation
    # ------------------------------------------------------------------

    def _build_qp_and_solver(self, A):
        """Precompute static QP matrices and fetch the compiled vmap'd solver for this shape.

        Called once at init time (or per-voxel for voxel-varying kernels, but that path currently
        falls back to __call__). The compiled kernel itself comes from the process-lifetime cache
        (:func:`_compiled_fit_batch`): rebuilding a ``CsdOsqpOptimizer`` per fit (the
        `spherical_harmonics_framework.py` pattern) stays cheap because only this numpy-only setup
        repeats -- an equal (scheme, sh_order, unity_constraint, lambda_lb, maxiter, tol, dtype,
        batch) key returns the already-compiled callable, so a second `.fit(solver='csd_jax')` on
        an equal scheme pays no compile.
        """
        Ncoef_total = self.Ncoef_total
        sh_start = self.sh_start
        Ncoef = self.Ncoef
        vf_idx = self.vf_indices
        N_hem = self.L_positivity.shape[0]

        # Q = 2 * (A' A + lambda R)
        Q = 2.0 * (A.T @ A + self.lambda_lb * self.R_smoothness)

        # A' (kept for per-voxel q = -2 A' signal)
        AT = A.T  # (Ncoef_total, N_meas)

        # Constraint matrix G (inequality G x <= h, h = 0):
        #   block 1: -L_positivity_padded  (FOD positivity)
        #   block 2: -vf_selector           (VF non-negativity)
        L_pad = np.zeros((N_hem, Ncoef_total))
        L_pad[:, sh_start:sh_start + Ncoef] = self.L_positivity
        G_pos = -L_pad  # (N_hem, Ncoef_total)

        N_vf = len(vf_idx)
        vf_sel = np.zeros((N_vf, Ncoef_total))
        for i, vi in enumerate(vf_idx):
            vf_sel[i, vi] = 1.0
        G_vf = -vf_sel  # (N_vf, Ncoef_total)

        G = np.vstack([G_pos, G_vf])  # (N_hem + N_vf, Ncoef_total)
        h = np.zeros(G.shape[0])

        # Equality constraint A_eq x = b_eq (unity VF, if requested):
        #   sum(x[vf_indices]) == 1 / sphere_jacobian
        if self.unity_constraint:
            A_eq = np.zeros((1, Ncoef_total))
            for vi in vf_idx:
                A_eq[0, vi] = 1.0
            b_eq = np.array([1.0 / self.sphere_jacobian])
        else:
            A_eq = None
            b_eq = None

        # Solve dtype.  dmipy enables jax_enable_x64 globally (float64) for
        # reference correctness, but the CSD QP solve is a GPU production path and
        # the FOD is thresholded/peak-extracted downstream, so it is comfortably
        # within the float32 noise floor.  On non-datacentre GPUs (e.g. L40S)
        # float64 runs at ~1/64 throughput AND has no good XLA matmul configs
        # ("All configs filtered out"), so float64 here is ~30-60x slower for no
        # accuracy benefit.  Default to float32; override with
        # DMIPY_CSD_JAX_DTYPE=float64 for a reference solve.
        _dt = os.environ.get("DMIPY_CSD_JAX_DTYPE", "float32").lower()
        dtype = jnp.float64 if _dt in ("float64", "f64", "64") else jnp.float32
        self._solve_dtype = dtype

        self._Q_jax  = jnp.array(Q,  dtype=dtype)
        self._AT_jax = jnp.array(AT, dtype=dtype)
        self._G_jax  = jnp.array(G,  dtype=dtype)
        self._h_jax  = jnp.array(h,  dtype=dtype)
        if self.unity_constraint:
            self._A_eq_jax = jnp.array(A_eq, dtype=dtype)
            self._b_eq_jax = jnp.array(b_eq, dtype=dtype)
        else:
            self._A_eq_jax = None
            self._b_eq_jax = None

        # Under jax.vmap, OSQP's while_loop runs until EVERY voxel in the batch
        # converges (or hits maxiter), so a few hard voxels drag the whole batch
        # to maxiter.  CSD FODs are thresholded/peak-extracted, so 1e-4 QP
        # precision and 4000 iters are far more than needed; capping iterations
        # bounds the batch cost.  Override with DMIPY_CSD_JAX_MAXITER / _TOL.
        maxiter = int(os.environ.get("DMIPY_CSD_JAX_MAXITER", self.maxiter))
        tol = float(os.environ.get("DMIPY_CSD_JAX_TOL", self.tol))
        batch = int(os.environ.get("DMIPY_CSD_JAX_BATCH", "16384"))

        self._fit_batch_fn = _compiled_fit_batch(
            self.acquisition_scheme.fingerprint(), self.sh_order, self.unity_constraint,
            float(self.lambda_lb), maxiter, tol, jnp.dtype(dtype).name, batch)

    # ------------------------------------------------------------------
    # Batch fitting (GPU-parallel over voxels)
    # ------------------------------------------------------------------

    def fit_batch(self, data_all, x0_all, eta=None):
        """Fit all voxels; float32 (jax_enable_x64=False) for the OSQP solve.

        CSD is a float32 production path: the FOD is thresholded / peak-extracted, so it sits
        within the float32 noise floor, and float64 is far slower on GPU. jaxopt's OSQP mixes
        float32 and float64 internals when the *global* ``jax_enable_x64`` flag is on -- which a
        preceding cylinder-model fit leaves enabled (cylinders need float64 for their Van
        Gelderen sums, whose alpha^6 terms overflow float32). That mismatch raises a dtype
        TypeError in ``lax.cond``.

        Two behaviours, chosen by whether this process called :func:`pin_x64_off`:

        - **Pinned** (a serving process, after ``pin_x64_off()``): x64 must already be off. If it
          is on anyway, refuse by name (:class:`X64PinnedError`) instead of silently saving and
          restoring a global flag underneath a concurrent server -- something upstream re-enabled
          a flag this process declared fixed, and that is what needs fixing.
        - **Unpinned** (the default, ordinary/script use): save the flag, force it off for this
          solve, restore it afterwards -- robust to whatever the ambient global flag was, at the
          cost of a save/restore on every call.
        """
        if _x64_pinned_off:
            if jax.config.jax_enable_x64:
                raise X64PinnedError(
                    "jax_enable_x64 is on, but this process pinned it off via "
                    "dmipy_fit.jax.csd_jax.pin_x64_off(). Something (e.g. a preceding cylinder "
                    "fit) re-enabled it; fix that caller -- csd_jax will not silently flip a "
                    "process-wide flag back off underneath a concurrent server.")
            return self._fit_batch_impl(data_all, x0_all, eta)

        _prev_x64 = jax.config.jax_enable_x64
        if _prev_x64:
            jax.config.update("jax_enable_x64", False)
        try:
            return self._fit_batch_impl(data_all, x0_all, eta)
        finally:
            if _prev_x64:
                jax.config.update("jax_enable_x64", True)

    def _fit_batch_impl(self, data_all, x0_all, eta=None):
        """Fit all voxels in parallel using jaxopt.OSQP + jax.vmap.

        Parameters
        ----------
        data_all : np.array, shape (N_voxels, N_meas)
            Normalised signal attenuation per voxel.
        x0_all : np.array, shape (N_voxels, N_parameters)
            Initial parameter vector per voxel (used to build per-voxel
            convolution kernels when the kernel is voxel-varying; for
            fixed-kernel models x0_all[0] was already used at init time).
        eta : float or None
            Rician noise floor estimate (in normalised signal units).
            When provided, a pre-processing bias correction is applied:
            ``data_corrected = sqrt(max(data^2 - eta^2, 0))``.
            This removes the Rician bias before the QP solve.

        Returns
        -------
        fitted_parameters : np.array, shape (N_voxels, N_parameters)
        """
        N_voxels = data_all.shape[0]

        # Rician bias pre-processing correction
        if eta is not None and eta > 0:
            data_all = np.sqrt(np.maximum(data_all ** 2 - eta ** 2, 0.0))

        if self.single_convolution_kernel:
            # Sub-batch the vmap over voxels at a FIXED batch size: the jitted
            # kernel is compiled (and XLA-autotuned) once on the first batch and
            # reused for the rest -- this bounds GPU memory and amortises the
            # (often very large) compile cost, instead of one monolithic call
            # over all voxels.  Also lets us show a progress bar.  Override the
            # batch size with the env var DMIPY_CSD_JAX_BATCH.
            batch = int(os.environ.get("DMIPY_CSD_JAX_BATCH", "16384"))
            batch = max(1, min(batch, N_voxels))
            n_batches = -(-N_voxels // batch)
            x_solutions = None
            try:
                from tqdm import tqdm
                rng = tqdm(range(0, N_voxels, batch), total=n_batches,
                           desc="CSD JAX vmap", unit="batch")
            except Exception:
                rng = range(0, N_voxels, batch)
            qp_args = (self._Q_jax, self._AT_jax, self._G_jax, self._h_jax)
            if self.unity_constraint:
                qp_args = qp_args + (self._A_eq_jax, self._b_eq_jax)
            for s in rng:
                chunk = data_all[s:s + batch]
                n = chunk.shape[0]
                if n < batch:                      # pad to keep a single shape
                    chunk = np.concatenate(
                        [chunk, np.zeros((batch - n, chunk.shape[1]),
                                         dtype=chunk.dtype)], axis=0)
                out = np.array(self._fit_batch_fn(
                    jnp.array(chunk, dtype=self._solve_dtype), *qp_args))
                if x_solutions is None:
                    x_solutions = np.zeros((N_voxels, out.shape[1]), dtype=float)
                x_solutions[s:s + n] = out[:n]
        else:
            # Voxel-varying kernel: fall back to sequential per-voxel fitting.
            # A future optimisation could batch voxels with the same kernel.
            x_solutions = np.zeros((N_voxels, self.Ncoef_total), dtype=float)
            for i in range(N_voxels):
                params_dict = self.model.parameter_vector_to_parameters(
                    x0_all[i])
                A_i = self.model._construct_convolution_kernel(
                    acquisition_scheme=self.acquisition_scheme, **params_dict)
                x_solutions[i] = self._solve_single(A_i, data_all[i])

        return self._postprocess_batch(x_solutions, x0_all)

    def _solve_single(self, A, signal):
        """Build per-voxel QP matrices and run OSQP (voxel-varying kernel)."""
        Q = 2.0 * (A.T @ A + self.lambda_lb * self.R_smoothness)
        c = -2.0 * (A.T @ signal)
        N_hem = self.L_positivity.shape[0]
        Ncoef_total = self.Ncoef_total
        sh_start = self.sh_start
        Ncoef = self.Ncoef
        vf_idx = self.vf_indices

        L_pad = np.zeros((N_hem, Ncoef_total))
        L_pad[:, sh_start:sh_start + Ncoef] = self.L_positivity
        G_pos = -L_pad
        N_vf = len(vf_idx)
        vf_sel = np.zeros((N_vf, Ncoef_total))
        for i, vi in enumerate(vf_idx):
            vf_sel[i, vi] = 1.0
        G = np.vstack([-L_pad, -vf_sel])
        h = np.zeros(G.shape[0])

        Q_jax = jnp.array(Q)   # dtype follows global JAX config
        c_jax = jnp.array(c)
        G_jax = jnp.array(G)
        h_jax = jnp.array(h)

        solver = OSQP(maxiter=self.maxiter, tol=self.tol,
                      check_primal_dual_infeasability=False)
        if self.unity_constraint:
            A_eq = np.zeros((1, Ncoef_total))
            for vi in vf_idx:
                A_eq[0, vi] = 1.0
            b_eq = np.array([1.0 / self.sphere_jacobian])
            sol = solver.run(None, params_obj=(Q_jax, c_jax),
                             params_eq=(jnp.array(A_eq), jnp.array(b_eq)),
                             params_ineq=(G_jax, h_jax))
        else:
            sol = solver.run(None, params_obj=(Q_jax, c_jax),
                             params_ineq=(G_jax, h_jax))
        return np.array(sol.params.primal)

    # ------------------------------------------------------------------
    # Post-processing: OSQP solution → parameter vector
    # ------------------------------------------------------------------

    def _postprocess_one(self, x_sol, x0_vector):
        """Convert a single OSQP solution array to the model's parameter vector."""
        fitted_params = self.model.parameter_vector_to_parameters(x0_vector)

        sh_fod = np.array(x_sol[self.sh_start:self.sh_start + self.Ncoef])
        sh_fod[0] = 1.0 / self.sphere_jacobian
        fitted_params['sh_coeff'] = sh_fod

        if not self.model.volume_fractions_fixed:
            fractions_array = (np.array(x_sol[self.vf_indices])
                               * 2.0 * np.sqrt(np.pi))
            for i, name in enumerate(self.model.partial_volume_names):
                fitted_params[name] = float(fractions_array[i])

        return self.model.parameters_to_parameter_vector(**fitted_params)

    def _postprocess_batch(self, x_solutions, x0_all):
        """Post-process a batch of OSQP solutions.

        Parameters
        ----------
        x_solutions : np.array, shape (N_voxels, Ncoef_total)
        x0_all      : np.array, shape (N_voxels, N_parameters)

        Returns
        -------
        np.array, shape (N_voxels, N_parameters)
        """
        N_voxels = x_solutions.shape[0]
        N_params = x0_all.shape[1]
        result = np.zeros((N_voxels, N_params), dtype=float)
        for i in range(N_voxels):
            result[i] = self._postprocess_one(x_solutions[i], x0_all[i])
        return result

    # ------------------------------------------------------------------
    # Single-voxel interface (compatible with CsdCvxpyOptimizer)
    # ------------------------------------------------------------------

    def __call__(self, data, x0_vector):
        """Fit a single voxel.

        Compatible with CsdCvxpyOptimizer.__call__ so this can be used as
        a drop-in in the standard per-voxel loop if needed.

        Parameters
        ----------
        data : np.array, shape (N_meas,)
        x0_vector : np.array, shape (N_parameters,)

        Returns
        -------
        np.array, shape (N_parameters,)
        """
        result = self.fit_batch(data[None], x0_vector[None])
        return result[0]


# ---------------------------------------------------------------------------
# Start-up warm-up
# ---------------------------------------------------------------------------

def warm(schemes, *, sh_order=8, unity_constraint=True, lambda_lb=0., maxiter=4000, tol=1e-4,
         batch=(1024, 8192), lambda_par=1.7e-9, lambda_iso=3.0e-9):
    """Compile the CSD kernel for each of ``schemes`` (and each ``batch`` size) ahead of a request.

    For every scheme and batch size, builds the default two-compartment response (``C1Stick`` +
    ``G1Ball``, the same response shape used across dmipy-fit's CSD tests) and calls
    ``CsdOsqpOptimizer.fit_batch`` once on a zero signal at that exact voxel count -- which forces
    jax to trace and compile that (scheme, batch) shape and populates the process-lifetime compile
    cache (:func:`_compiled_fit_batch`). A serving process calls this once at start-up, for the
    acquisition shapes it actually serves, so the first real request never pays a compile.

    Parameters
    ----------
    schemes : iterable of AcquisitionScheme
        The acquisition shapes to warm, e.g. the DiSCo 4-shell and clinical 1-shell schemes a
        server fits against.
    sh_order, unity_constraint, lambda_lb, maxiter, tol : as :class:`CsdOsqpOptimizer`
        Must equal what the server actually fits with -- warming a different key leaves that key
        cold.
    batch : int or tuple of int
        Voxel-batch size(s) to warm (default: 1024 and 8192).
    lambda_par, lambda_iso : float
        Response diffusivities (m^2/s) for the placeholder C1Stick + G1Ball response the warm-up
        model uses; irrelevant to the compiled shape, only present so the kernel construction has
        concrete numbers.

    Returns
    -------
    list of dict
        One entry per (scheme, batch): ``scheme_fingerprint``, ``n_measurements``, ``sh_order``,
        ``batch`` and ``seconds`` (wall-clock time of the warming call, i.e. the compile this
        process just paid so a later request does not).
    """
    from ..signal_models.gaussian_models import G1Ball
    from ..signal_models.cylinder_models import C1Stick
    from ..core.modeling_framework import MultiCompartmentSphericalHarmonicsModel

    if isinstance(batch, int):
        batch = (batch,)

    report = []
    for scheme in schemes:
        mc = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick(), G1Ball()], sh_order=sh_order)
        mc.set_fixed_parameter('C1Stick_1_lambda_par', lambda_par)
        mc.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso)
        mc.scheme = scheme
        mc._check_if_kernel_parameters_are_fixed()
        mc.S0_responses = np.ones(len(mc.models), dtype=float)
        x0 = mc.parameter_initial_guess_to_parameter_vector()
        x0_2d = np.reshape(x0, (1, -1))

        for b in batch:
            _prev_batch_env = os.environ.get("DMIPY_CSD_JAX_BATCH")
            os.environ["DMIPY_CSD_JAX_BATCH"] = str(b)
            try:
                opt = CsdOsqpOptimizer(
                    scheme, mc, x0_2d, sh_order=sh_order, unity_constraint=unity_constraint,
                    lambda_lb=lambda_lb, maxiter=maxiter, tol=tol)
                data = np.zeros((b, scheme.number_of_measurements), dtype=float)
                x0_all = np.tile(x0_2d, (b, 1))
                t0 = time.perf_counter()
                opt.fit_batch(data, x0_all)
                dt = time.perf_counter() - t0
            finally:
                if _prev_batch_env is None:
                    os.environ.pop("DMIPY_CSD_JAX_BATCH", None)
                else:
                    os.environ["DMIPY_CSD_JAX_BATCH"] = _prev_batch_env

            report.append({
                'scheme_fingerprint': scheme.fingerprint(),
                'n_measurements': scheme.number_of_measurements,
                'sh_order': sh_order,
                'batch': b,
                'seconds': dt,
            })
    return report
