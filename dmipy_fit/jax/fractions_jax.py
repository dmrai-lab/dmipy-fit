r"""Batched (GPU/vmap) multi-tissue volume-fraction estimation.

When per-compartment ``S0_tissue_responses`` are supplied, the multi-tissue
*geometric* volume fractions are recovered by a separate convex step (after the
non-linear parameters are fit on the normalised, scale-stable signal): a
non-negative least squares

.. math::  \hat f = \arg\min_{f\ge 0} \lVert S - \Phi f\rVert_2^2,

where the basis :math:`\Phi[:,i] = S0_i\,E_i` is each compartment's predicted
signal (full per-measurement signal for a :class:`MultiCompartmentModel`, or the
per-shell spherical mean for a :class:`MultiCompartmentSphericalMeanModel`),
scaled by its raw S0 response. Keeping this linear step separate from the
non-linear fit is deliberate: the non-linear solver's absolute/relative loss
tolerances are sensitive to the (arbitrary, scanner-dependent) S0 scaling, so
S0 is kept out of the non-linear convergence and folded in only here, where the
convex optimum is exact regardless of scale.

This module batches that NNLS across voxels with ``jax.vmap`` + ``jaxopt.OSQP``
(the same machinery as csd_jax), replacing the per-voxel scipy COBYLA loop.
``MultiTissueVolumeFractionOptimizer`` (numpy/COBYLA) remains the CPU fallback.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import OSQP

from .multicompartment_jax import (
    _build_dispatch, _extract_params_jax, _JAX_SM_MODEL_FNS,
    _JAX_MODEL_FNS, _JAX_MODEL_FACTORIES)
from .jax_compat import scheme_to_jax


def _nnls(phi, data, maxiter, tol):
    """Non-negative least squares min_{f>=0} ||data - phi f||^2 via OSQP.

    phi and data are ~S0 magnitude (thousands), so the raw QP (Q ~ 1e8) is
    badly conditioned for OSQP. Scaling both by a common factor leaves the
    argmin unchanged, so normalise to unit scale first."""
    scale = jnp.maximum(jnp.sqrt(jnp.mean(data ** 2)), 1e-12)
    phi = phi / scale
    data = data / scale
    Q = 2.0 * (phi.T @ phi)
    c = -2.0 * (phi.T @ data)
    K = phi.shape[1]
    G = -jnp.eye(K)
    h = jnp.zeros(K)
    sol = OSQP(maxiter=maxiter, tol=tol,
               check_primal_dual_infeasability=False).run(
        params_obj=(Q, c), params_ineq=(G, h))
    return jnp.maximum(sol.params.primal, 0.0)


def _sm_compartment_basis(model, acquisition_scheme):
    """Builder: param_vector -> (N_shells, N_models) per-compartment spherical
    mean (b0 row = 1), using the JAX spherical-mean dispatch."""
    dispatch = _build_dispatch(model)
    b0_mask = acquisition_scheme.shell_b0_mask
    non_b0_idx = jnp.asarray(np.where(~b0_mask)[0])
    shell_bvals_non_b0 = jnp.asarray(acquisition_scheme.shell_bvalues[~b0_mask])
    N_shells = acquisition_scheme.N_shells

    def basis(params_scaled):
        cols = []
        for entry in dispatch:
            mp = _extract_params_jax(params_scaled, entry['param_slices'])
            sm_non_b0 = _JAX_SM_MODEL_FNS[entry['model_type']](
                shell_bvals_non_b0, mp)
            col = jnp.ones(N_shells).at[non_b0_idx].set(sm_non_b0)
            cols.append(col)
        return jnp.stack(cols, axis=1)              # (N_shells, N_models)
    return basis


def _full_compartment_basis(model, acquisition_scheme):
    """Builder: param_vector -> (N_meas, N_models) per-compartment full signal,
    using the JAX full-signal dispatch."""
    dispatch = _build_dispatch(model, acquisition_scheme)
    scheme_jax = scheme_to_jax(acquisition_scheme)

    def basis(params_scaled):
        cols = []
        for entry in dispatch:
            mp = _extract_params_jax(params_scaled, entry['param_slices'])
            cols.append(entry['jax_fn'](scheme_jax, mp))
        return jnp.stack(cols, axis=1)              # (N_meas, N_models)
    return basis


def supports_jax_fraction_fit(model, spherical_mean):
    """True iff every compartment has the JAX forward needed for the basis."""
    if spherical_mean:
        return all(type(m) in _JAX_SM_MODEL_FNS for m in model.models)
    reg = set(_JAX_MODEL_FNS) | set(_JAX_MODEL_FACTORIES)
    return all(type(m) in reg for m in model.models)


def fit_multi_tissue_fractions_jax(model, acquisition_scheme, param_vectors,
                                   data, S0_tissue_responses,
                                   spherical_mean, maxiter=4000, tol=1e-6):
    """Batched geometric multi-tissue volume fractions for all voxels.

    Parameters
    ----------
    param_vectors : (N_vox, N_params) fitted parameter vectors in SI units.
    data : (N_vox, M) signal the fractions are fit against -- raw per-shell
        spherical mean (spherical_mean=True) or per-measurement signal.
    S0_tissue_responses : (N_models,) raw per-compartment S0 responses.
    spherical_mean : bool, choose the per-compartment basis.

    Returns
    -------
    fractions : (N_vox, N_models) non-negative volume fractions.
    """
    basis = (_sm_compartment_basis if spherical_mean
             else _full_compartment_basis)(model, acquisition_scheme)
    S0 = jnp.asarray(np.asarray(S0_tissue_responses, dtype=float))

    def fit_one(params_scaled, voxel_data):
        phi = basis(params_scaled) * S0[None, :]    # S0-weighted basis
        return _nnls(phi, voxel_data, maxiter, tol)

    fitter = jax.jit(jax.vmap(fit_one))
    return np.asarray(fitter(jnp.asarray(param_vectors), jnp.asarray(data)))
