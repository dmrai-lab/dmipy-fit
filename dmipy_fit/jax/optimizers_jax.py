"""JAX-based gradient optimizer for MultiCompartmentModel.

Strategy: JAX computes the forward model and its analytic gradient via
jax.value_and_grad. scipy.optimize.minimize(method='L-BFGS-B') runs the
optimization loop. This avoids compiling the entire multi-step optimizer
as a single XLA kernel while still getting exact gradients from JAX.

After the first call the JIT-compiled value_and_grad function is cached,
so all subsequent voxels use compiled code with no Python overhead in the
forward pass.

Fittable sigma
--------------
When fit_sigma=True is passed to JaxOptimizer, sigma (the Rician noise
standard deviation) is appended as the last element of the parameter
vector and jointly optimized with the diffusion model parameters.

The dispatch between fixed-sigma and fittable-sigma loss paths is done
at __init__ time, so the JIT-compiled function never contains a Python
conditional — no recompilation penalty.
"""

import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp

from .multicompartment_jax import build_mc_forward_fn, build_mc_sm_forward_fn
from .losses_jax import mse_loss, rician_nll_fittable


def nested_to_normalized_fractions_jax(nested):
    """Convert N-1 nested fractions to N normalized fractions (sums to 1).

    JAX-compatible equivalent of brute2fine.nested_to_normalized_fractions.
    Uses jax.lax.scan so the recurrence is JIT-compatible.

    Parameters
    ----------
    nested : jnp.array, shape (N-1,), each in [0, 1]

    Returns
    -------
    jnp.array, shape (N,), sums to 1
    """
    def body(remaining, v):
        frac = remaining * v
        return remaining - frac, frac

    _, fracs = jax.lax.scan(body, jnp.array(1.0), nested)
    last = jnp.array(1.0) - jnp.sum(fracs)
    return jnp.concatenate([fracs, jnp.array([last])])


def normalized_to_nested_fractions_jax(normalized):
    """Convert N normalized fractions to N-1 nested fractions.

    Parameters
    ----------
    normalized : array-like, shape (N,), sums to 1

    Returns
    -------
    jnp.array, shape (N-1,)
    """
    normalized = jnp.asarray(normalized)

    def body(remaining, frac):
        nested_i = frac / jnp.where(remaining > 1e-15, remaining, 1e-15)
        return remaining - frac, nested_i

    _, nested = jax.lax.scan(body, jnp.array(1.0), normalized[:-1])
    return nested


class JaxOptimizer:
    """JAX-gradient optimizer for MultiCompartmentModel.

    Uses JAX's automatic differentiation (via jax.value_and_grad) to compute
    exact gradients of the loss, then passes those to scipy L-BFGS-B.

    The JIT-compiled value_and_grad function is compiled once on the first
    voxel call and reused for all subsequent voxels.

    Parameters
    ----------
    model : MultiCompartmentModel
    acquisition_scheme : DmipyAcquisitionScheme
    maxiter : int
        Maximum L-BFGS-B iterations per voxel.
    Ns : int
        (unused — kept for API compatibility with Brute2FineOptimizer)
    N_sphere_samples : int
        (unused — kept for API compatibility)
    forward_fn_override : callable or None
        Optional override for the JAX forward model function.
    loss_fn : callable or None
        Optional fixed-sigma loss with signature loss_fn(E_model, data) -> scalar.
        Defaults to MSE.  Ignored when fit_sigma=True (Rician NLL is used instead).
    fit_sigma : bool
        If True, sigma is appended as the last element of the parameter vector
        and jointly optimized with the diffusion model parameters.
        The Rician NLL is used as the loss function in this mode.
    sigma_x0 : float or None
        Initial guess for sigma in SI units (= 1/SNR₀).  Required when
        fit_sigma=True; ignored otherwise.
    sigma_range : (float, float)
        (lower, upper) bounds for sigma in SI units.  Default (0.001, 0.5)
        covers SNR 2–1000.
    sigma_scale : float
        Scale factor applied to sigma so it lives at O(1) in the optimizer.
        Default 1.0 (no rescaling) is fine since sigma ~ 0.01–0.1 matches
        the scale of normalized volume fractions.
    """
    _citations = {
        'definition': [
            {'key': 'byrd1995', 'authors': 'Byrd RH, Lu P, Nocedal J',
             'title': 'A limited memory algorithm for bound constrained optimization',
             'journal': 'SIAM Journal on Scientific Computing',
             'year': 1995, 'doi': '10.1137/0916069'},
            {'key': 'gudbjartsson1995', 'authors': 'Gudbjartsson H, Patz S',
             'title': 'The Rician distribution of noisy MRI data',
             'journal': 'Magnetic Resonance in Medicine',
             'year': 1995, 'doi': '10.1002/mrm.1910340618'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = []

    def __init__(self, model, acquisition_scheme,
                 maxiter=200, Ns=5, N_sphere_samples=30,
                 forward_fn_override=None, loss_fn=None,
                 fit_sigma=False, sigma_x0=None,
                 sigma_range=(0.001, 0.5), sigma_scale=1.0,
                 fittable_loss_override=None,
                 warm_start_mu=False):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.maxiter = maxiter
        self._fit_sigma = fit_sigma
        self._sigma_scale = float(sigma_scale)

        if forward_fn_override is not None:
            self._forward_fn = forward_fn_override
        else:
            from ..core.spherical_mean_framework import MultiCompartmentSphericalMeanModel
            if isinstance(model, MultiCompartmentSphericalMeanModel):
                self._forward_fn = build_mc_sm_forward_fn(model, acquisition_scheme)
            else:
                self._forward_fn = build_mc_forward_fn(model, acquisition_scheme)
        self._scales = np.array(model.scales_for_optimization)
        self._is_multi = model.N_models > 1
        self._N_models = model.N_models

        # Loss function for fixed-sigma path: defaults to MSE.
        # Custom losses (e.g. rician_nll(sigma)) must have signature
        #   loss_fn(E_model, data) -> scalar   (pure JAX, JIT-compatible).
        # Ignored when fit_sigma=True (Rician NLL is used instead).
        self._loss_fn = loss_fn if loss_fn is not None else mse_loss()

        # Bounds in nested-VF space (drop last VF if multi-compartment)
        bounds_np = np.array(model.bounds_for_optimization)  # (N_params, 2)
        if self._is_multi:
            bounds_np = bounds_np[:-1]  # drop implicit last VF
        self._bounds = list(map(tuple, bounds_np))
        self._lower = bounds_np[:, 0].copy()
        self._upper = bounds_np[:, 1].copy()

        # ------------------------------------------------------------------
        # Fittable sigma: extend bounds by one element for sigma
        # ------------------------------------------------------------------
        if fit_sigma:
            if sigma_x0 is None:
                raise ValueError(
                    "sigma_x0 must be provided when fit_sigma=True. "
                    "Pass an initial guess for sigma in SI units (= 1/SNR0)."
                )
            sigma_lo_norm = float(sigma_range[0]) / self._sigma_scale
            sigma_hi_norm = float(sigma_range[1]) / self._sigma_scale
            self._bounds.append((sigma_lo_norm, sigma_hi_norm))
            self._lower = np.append(self._lower, sigma_lo_norm)
            self._upper = np.append(self._upper, sigma_hi_norm)
            # Normalize the initial sigma guess
            self._sigma_x0_norm = float(sigma_x0) / self._sigma_scale
            self._sigma_x0_norm = float(
                np.clip(self._sigma_x0_norm, sigma_lo_norm, sigma_hi_norm))
            # Rician NLL with sigma passed explicitly (used in _loss_fn_jax).
            # fittable_loss_override allows callers (e.g. SM framework) to
            # substitute a domain-specific loss, e.g. per-shell sigma_eff.
            if fittable_loss_override is not None:
                self._fittable_loss = fittable_loss_override
            else:
                self._fittable_loss = rician_nll_fittable()

        # Build the JIT-compiled value_and_grad function once at init time.
        # The dispatch between fixed-sigma and fittable-sigma paths is done
        # HERE (Python level), so _loss_fn_jax never contains a Python
        # conditional — no JIT recompilation when sigma changes.
        self._val_and_grad = jax.jit(
            jax.value_and_grad(self._loss_fn_jax)
        )

        # ------------------------------------------------------------------
        # Warm-start mu: GPU DTI fitter for orientation initialisation
        # ------------------------------------------------------------------
        self._warm_start_mu = warm_start_mu
        if warm_start_mu:
            from .dti_jax import detect_mu_indices, build_dti_fitter
            mu_list = detect_mu_indices(model)
            if len(mu_list) == 0:
                raise ValueError(
                    "warm_start_mu=True requires a model with an orientation "
                    "parameter (a parameter whose name ends in '_mu'). "
                    f"This model has none. Parameter names: "
                    f"{list(model.parameter_cardinality.keys())}"
                )
            if len(mu_list) > 1:
                mu_names = [p[0] for p in mu_list]
                raise ValueError(
                    f"warm_start_mu=True requires exactly one orientation "
                    f"compartment. Found {len(mu_list)}: {mu_names}. "
                    f"DTI provides a single principal eigenvector and cannot "
                    f"warm-start multi-bundle models. Set warm_start_mu=False "
                    f"and supply x0 from a multi-orientation initialisation."
                )
            _, i0, i1 = mu_list[0]
            self._mu_slice = slice(i0, i1)          # nested vector indices
            # Auto-select b_max: DTI is only valid at low b-values.
            # If the acquisition has shells above 2000 s/mm² (2e9 s/m²),
            # restrict to b ≤ 1500 s/mm² to avoid fitting non-Gaussian signal.
            bvals_si = np.array(acquisition_scheme.bvalues)
            auto_b_max = None
            if bvals_si.max() > 2e9:
                auto_b_max = 1.5e9   # 1500 s/mm² in s/m²
            self._dti_fitter = build_dti_fitter(acquisition_scheme, b_max=auto_b_max)

    # ------------------------------------------------------------------
    # Forward model helper (shared by both loss paths)
    # ------------------------------------------------------------------

    def _forward_from_params(self, params_model_nested, scales):
        """Run forward model from nested-VF, normalized-units model params.

        Parameters
        ----------
        params_model_nested : jnp.array
            Model parameters only (no sigma), in nested-VF / normalized units.
        scales : jnp.array
            Parameter scales matching params_model_nested, cast to correct dtype.

        Returns
        -------
        jnp.array, shape (N_meas,)
            Predicted signal attenuation.
        """
        if self._is_multi:
            n_non_vf = len(scales) - self._N_models
            non_vf_si = params_model_nested[:n_non_vf] * scales[:n_non_vf]
            nested_vf = params_model_nested[n_non_vf:]
            normalized_vf = nested_to_normalized_fractions_jax(nested_vf)
            params_scaled = jnp.concatenate([non_vf_si, normalized_vf])
        else:
            params_scaled = params_model_nested * scales
        return self._forward_fn(params_scaled)

    # ------------------------------------------------------------------
    # x0 initialisation
    # ------------------------------------------------------------------

    def make_x0(self, data_batch, dtype=None):
        """Build initial parameter vector for a batch of voxels.

        All parameters are initialised to their midpoints.  If
        ``warm_start_mu=True`` was set at construction, the orientation
        indices are overwritten with the DTI principal eigenvector
        estimated from ``data_batch`` on-GPU — no dipy required.

        The returned array is ready to pass directly to any fitter:
        ``build_vmap_fitter``, ``build_gn_fitter``, ``build_mrest_fitter``.

        Parameters
        ----------
        data_batch : jnp.array (N_vox, N_meas)
            Normalised signal S/S0.  Must already be on-device.
        dtype : jnp.dtype or None
            Defaults to float32.

        Returns
        -------
        x0 : jnp.array (N_vox, N_params_nested)
        """
        if dtype is None:
            dtype = jnp.float32

        N_vox  = data_batch.shape[0]
        lower  = jnp.array(self._lower, dtype=dtype)
        upper  = jnp.array(self._upper, dtype=dtype)
        mid    = (lower + upper) * 0.5

        # Midpoint initialisation — all voxels start from centre of bounds.
        x0 = jnp.broadcast_to(mid[None, :], (N_vox, len(mid)))

        if self._warm_start_mu:
            mu_sph, _ = self._dti_fitter(data_batch)   # (N_vox, 2) radians

            # Clip into bounds with 5 % margin (same as warm_start_to_nested).
            margin   = 0.05 * (upper - lower)
            mu_lo    = (lower + margin)[self._mu_slice]
            mu_hi    = (upper - margin)[self._mu_slice]
            mu_clipped = jnp.clip(mu_sph, mu_lo, mu_hi)

            x0 = x0.at[:, self._mu_slice].set(mu_clipped)

        return x0

    # ------------------------------------------------------------------
    # Loss function in JAX (operates in nested-VF, normalized space)
    # ------------------------------------------------------------------

    def _loss_fn_jax(self, params_nested_normalized, data, weights=None):
        """Loss function, fully differentiable via JAX.

        When fit_sigma=False:
            params_nested_normalized has length N_model_params (nested-VF).
            Uses self._loss_fn (default MSE, or fixed-sigma Rician NLL).

        When fit_sigma=True:
            params_nested_normalized has length N_model_params + 1.
            The last element is sigma / sigma_scale (normalized).
            Uses Rician NLL with sigma extracted from params.

        Cast scales to match input dtype so lax.while_loop carry types
        are consistent whether inputs are float32 (GPU) or float64.

        Parameters
        ----------
        params_nested_normalized : jnp.array
        data : jnp.array, shape (N_meas,)
        weights : jnp.array or None, shape (N_meas,)
            Optional per-measurement weights.  When provided, the scalar loss
            is computed as ``sum(w * per_meas_loss) / sum(w)`` instead of the
            unweighted mean.  ``weights=None`` produces bit-identical results
            to the original (no-weight) path.
        """
        scales = jnp.array(self._scales, dtype=params_nested_normalized.dtype)

        if self._fit_sigma:
            # Split off sigma from the end of the parameter vector.
            params_model = params_nested_normalized[:-1]
            sigma_norm   = params_nested_normalized[-1]
            sigma_si     = sigma_norm * jnp.array(
                self._sigma_scale, dtype=params_nested_normalized.dtype)
            E_model = self._forward_from_params(params_model, scales)
            if weights is None:
                return self._fittable_loss(E_model, data, sigma_si)
            else:
                # Compute per-measurement NLL then apply weights.
                loss_scalar = self._fittable_loss(E_model, data, sigma_si)
                # _fittable_loss returns mean(); we need per-meas values.
                # Re-derive per-meas terms inline to avoid refactoring loss API.
                sig    = jnp.maximum(sigma_si,
                                     jnp.array(1e-6, dtype=E_model.dtype))
                sig2   = sig ** 2
                nu     = jnp.clip(E_model, 0.0, None)
                r      = data
                arg    = nu * r / sig2
                log_i0 = (jnp.abs(arg)
                          + jnp.log(jax.scipy.special.i0e(arg) + 1e-38))
                nll_per = (2.0 * jnp.log(sig)
                           + (r ** 2 + nu ** 2) / (2.0 * sig2)
                           - log_i0)
                w = jnp.array(weights, dtype=E_model.dtype)
                return jnp.sum(w * nll_per) / jnp.maximum(jnp.sum(w), 1e-38)
        else:
            E_model = self._forward_from_params(params_nested_normalized, scales)
            if weights is None:
                return self._loss_fn(E_model, data)
            else:
                # Apply weights to per-measurement squared residuals (MSE path).
                # For custom loss_fn we fall back to the weighted-MSE formula
                # so the interface remains generic.
                diff = E_model - data
                w = jnp.array(weights, dtype=E_model.dtype)
                return jnp.sum(w * diff * diff) / jnp.maximum(jnp.sum(w), 1e-38)

    # ------------------------------------------------------------------
    # scipy-compatible objective (returns (value, gradient) as numpy)
    # ------------------------------------------------------------------

    def _scipy_obj(self, params_nested_normalized, data_jax):
        """Return (loss, grad) as numpy float64 arrays for scipy."""
        p = jnp.array(params_nested_normalized)
        val, grad = self._val_and_grad(p, data_jax)
        return float(val), np.array(grad, dtype=np.float64)

    # ------------------------------------------------------------------
    # Per-voxel call (same interface as Brute2FineOptimizer.__call__)
    # ------------------------------------------------------------------

    def __call__(self, data, x0_vector):
        """Fit a single voxel.

        Parameters
        ----------
        data : np.array, shape (N_meas,), normalized signal attenuation
        x0_vector : np.array, shape (N_params,), initial guess in
            normalized space (SI / scales). NaN = no guess.
            When fit_sigma=True this vector covers model params only
            (sigma_x0 was captured at __init__ time).

        Returns
        -------
        x_fine : np.array
            When fit_sigma=False: shape (N_params,), fitted params in
            normalized space (SI / scales). VFs are normalized (sum to 1).
            When fit_sigma=True: shape (N_params + 1,). The last element
            is sigma in SI units (already de-normalized by sigma_scale).
        """
        data_jax = jnp.array(data)
        x0_nested = self._prepare_x0(x0_vector)

        result = minimize(
            self._scipy_obj,
            x0_nested,
            args=(data_jax,),
            method='L-BFGS-B',
            jac=True,
            bounds=self._bounds,
            options={'maxiter': self.maxiter, 'ftol': 1e-15, 'gtol': 1e-8},
        )
        return self._unnest(result.x)

    def _prepare_x0(self, x0_vector):
        """Convert full normalized x0 → nested-VF form; fill NaN → midpoint.

        When fit_sigma=True, appends sigma_x0_norm as the last element.
        """
        if self._is_multi:
            n_non_vf = len(self._scales) - self._N_models
            x0_non_vf = x0_vector[:n_non_vf].copy()
            x0_vf_normalized = x0_vector[n_non_vf:]
            if np.all(np.isnan(x0_vf_normalized)):
                equal_vf = np.ones(self._N_models) / self._N_models
                x0_vf_nested = np.array(
                    normalized_to_nested_fractions_jax(jnp.array(equal_vf)))
            else:
                x0_vf_nested = np.array(
                    normalized_to_nested_fractions_jax(
                        jnp.array(x0_vf_normalized)))
            x0_nested = np.concatenate([x0_non_vf, x0_vf_nested])
        else:
            x0_nested = x0_vector.copy()

        nan_mask = np.isnan(x0_nested)
        # _lower/_upper include sigma bounds when fit_sigma=True, but x0_nested
        # does not yet have sigma appended — use only the model-param slice.
        n_model = len(x0_nested)
        x0_nested[nan_mask] = (
            (self._lower[:n_model] + self._upper[:n_model]) / 2.0)[nan_mask]
        x0_nested = np.clip(x0_nested, self._lower[:n_model],
                            self._upper[:n_model])

        if self._fit_sigma:
            x0_nested = np.append(x0_nested, self._sigma_x0_norm)

        return x0_nested

    def _unnest(self, x_nested):
        """Convert nested-VF result → full normalized VF + optional sigma.

        When fit_sigma=True the last element of x_nested is sigma (normalized).
        It is de-normalized (multiplied by sigma_scale) and kept as the last
        element of the returned array so the caller can extract it easily.
        """
        if self._fit_sigma:
            # Separate sigma from model params
            x_model_nested = x_nested[:-1]
            sigma_norm = x_nested[-1]
            sigma_si = sigma_norm * self._sigma_scale
            # Clip to declared range: L-BFGS-B can overshoot by a float ULP.
            sigma_lo = self._lower[-1] * self._sigma_scale
            sigma_hi = self._upper[-1] * self._sigma_scale
            sigma_si = float(np.clip(sigma_si, sigma_lo, sigma_hi))
            x_model_unnested = self._unnest_model(x_model_nested)
            return np.append(x_model_unnested, sigma_si)
        else:
            return self._unnest_model(x_nested)

    def _unnest_model(self, x_model_nested):
        """Convert nested-VF model params → normalized VF form (no sigma)."""
        if self._is_multi:
            n_non_vf = len(self._scales) - self._N_models
            non_vf = x_model_nested[:n_non_vf]
            nested_vf = x_model_nested[n_non_vf:]
            normalized_vf = np.array(
                nested_to_normalized_fractions_jax(jnp.array(nested_vf)))
            return np.concatenate([non_vf, normalized_vf])
        return x_model_nested
