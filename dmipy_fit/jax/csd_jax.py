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

Only 'c' varies per voxel, which makes jax.vmap very efficient: Q, G, h,
A_eq and b_eq are compiled as constants, while 'c' is the batched axis.
"""

import numpy as np
import jax
import jax.numpy as jnp
from dipy.data import get_sphere, HemiSphere
from dipy.reconst.shm import real_sh_tournier as real_sym_sh_mrtrix
from dipy.reconst.shm import sph_harm_ind_list

from jaxopt import OSQP


__all__ = ['CsdOsqpOptimizer']


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
        sphere = get_sphere(name='symmetric724')
        hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)
        self.L_positivity = real_sym_sh_mrtrix(
            self.sh_order, hemisphere.theta, hemisphere.phi, legacy=False)[0]

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
        sh_l = sph_harm_ind_list(sh_order)[1]
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
        """Precompute static QP matrices and compile the vmap'd solver.

        Called once at init time (or per-voxel for voxel-varying kernels,
        but that path currently falls back to __call__).
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
        import os
        _dt = os.environ.get("DMIPY_CSD_JAX_DTYPE", "float32").lower()
        dtype = jnp.float64 if _dt in ("float64", "f64", "64") else jnp.float32
        self._solve_dtype = dtype

        Q_jax   = jnp.array(Q,   dtype=dtype)
        AT_jax  = jnp.array(AT,  dtype=dtype)
        G_jax   = jnp.array(G,   dtype=dtype)
        h_jax   = jnp.array(h,   dtype=dtype)
        if self.unity_constraint:
            A_eq_jax = jnp.array(A_eq, dtype=dtype)
            b_eq_jax = jnp.array(b_eq, dtype=dtype)
        else:
            A_eq_jax = None
            b_eq_jax = None

        self._fit_batch_fn = self._make_fit_batch(
            Q_jax, AT_jax, G_jax, h_jax, A_eq_jax, b_eq_jax, dtype)

    def _make_fit_batch(self, Q, AT, G, h, A_eq, b_eq, dtype):
        """Return a jit+vmap'd function: (B, N_meas) → (B, Ncoef_total)."""
        # Under jax.vmap, OSQP's while_loop runs until EVERY voxel in the batch
        # converges (or hits maxiter), so a few hard voxels drag the whole batch
        # to maxiter.  CSD FODs are thresholded/peak-extracted, so 1e-4 QP
        # precision and 4000 iters are far more than needed; capping iterations
        # bounds the batch cost.  Override with DMIPY_CSD_JAX_MAXITER / _TOL.
        import os
        maxiter = int(os.environ.get("DMIPY_CSD_JAX_MAXITER", self.maxiter))
        tol = float(os.environ.get("DMIPY_CSD_JAX_TOL", self.tol))
        solver = OSQP(
            maxiter=maxiter,
            tol=tol,
            check_primal_dual_infeasability=False,  # jaxopt typo: "infeasability"
        )

        if A_eq is not None:
            def fit_one(signal):
                c = jnp.array(-2.0, dtype=dtype) * (AT @ signal.astype(dtype))
                sol = solver.run(
                    None,                  # init_params=None → auto-init
                    params_obj=(Q, c),
                    params_eq=(A_eq, b_eq),
                    params_ineq=(G, h),
                )
                return sol.params.primal  # KKTSolution.primal
        else:
            def fit_one(signal):
                c = jnp.array(-2.0, dtype=dtype) * (AT @ signal.astype(dtype))
                sol = solver.run(
                    None,
                    params_obj=(Q, c),
                    params_ineq=(G, h),
                )
                return sol.params.primal

        return jax.jit(jax.vmap(fit_one))

    # ------------------------------------------------------------------
    # Batch fitting (GPU-parallel over voxels)
    # ------------------------------------------------------------------

    def fit_batch(self, data_all, x0_all, eta=None):
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
            import os
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
            for s in rng:
                chunk = data_all[s:s + batch]
                n = chunk.shape[0]
                if n < batch:                      # pad to keep a single shape
                    chunk = np.concatenate(
                        [chunk, np.zeros((batch - n, chunk.shape[1]),
                                         dtype=chunk.dtype)], axis=0)
                out = np.array(self._fit_batch_fn(
                    jnp.array(chunk, dtype=self._solve_dtype)))
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
