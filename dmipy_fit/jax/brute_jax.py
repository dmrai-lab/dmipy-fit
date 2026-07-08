"""JAX-accelerated brute-force global search for MC model fitting.

Mirrors GlobalBruteOptimizer but moves both the signal grid computation
and the per-voxel MSE matching onto the GPU:

  1. Build parameter grid (CPU, once) — same logic as brute2fine, no signal eval.
  2. Evaluate forward model for all grid points (GPU, vmap, once).
  3. At search time: (N_voxels × N_grid) MSE via distance-matrix matmul,
     argmin over grid dim gives the best starting point per voxel.

The result feeds directly into vmap_fit as x0_nested_all.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

_SPHERES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'spheres')


def _build_param_grid(model, Ns, N_sphere_samples):
    """Build the brute-force parameter grid in SI units.

    Replicates GlobalBruteOptimizer.precompute_signal_grid() grid-construction
    logic but does NOT evaluate the signal (that happens on GPU via JAX).

    Returns
    -------
    parameter_grid : np.ndarray, shape (N_grid, N_params_full), SI units
    """
    from ..utils.utils import cart2mu
    from ..optimizers.brute2fine import nested_to_normalized_fractions

    sphere_vertices = np.loadtxt(
        os.path.join(_SPHERES_PATH, "01-shells-{}.txt".format(N_sphere_samples)),
        skiprows=1)[:, 1:]
    mu = cart2mu(sphere_vertices)           # (N_sphere, 2) — (theta, phi)

    parameter_cardinality_items = list(model.parameter_cardinality.items())
    N_model_fracts = 0
    if model.N_models > 1:
        N_model_fracts = model.N_models
        parameter_cardinality_items = parameter_cardinality_items[:-N_model_fracts]

    max_cardinality = int(np.max(list(model.parameter_cardinality.values())))
    grids_per_mu = []
    for card_counter in range(max_cardinality):
        per_parameter_vectors = []
        counter = 0
        for name, card in parameter_cardinality_items:
            par_range = model.parameter_ranges[name]
            opt_flag = model.parameter_optimization_flags[name]
            if card == 1:
                if not opt_flag:
                    # passive parameter: use midpoint, don't grid-search
                    midpoint = ((par_range[0] + par_range[1]) / 2.
                                * model.parameter_scales[name])
                    per_parameter_vectors.append(np.array([midpoint]))
                else:
                    per_parameter_vectors.append(
                        np.linspace(par_range[0], par_range[1], Ns)
                        * model.parameter_scales[name])
                counter += 1
            if card == 2:
                per_parameter_vectors.append(
                    mu[:, card_counter] * model.parameter_scales[name][0])
                # Keep nan as a 1-element array to preserve counter alignment
                # with brute2fine's extraction loop (same indexing convention).
                per_parameter_vectors.append(np.array([np.nan]))
                counter += 2
        if model.N_models > 1:
            for _ in range(N_model_fracts - 1):
                per_parameter_vectors.append(np.linspace(0., 1., Ns))

        # Default indexing='xy' matches brute2fine — do not change.
        grids_per_mu.append(np.meshgrid(*per_parameter_vectors))

    counter = 0
    param_dict = {}
    for name, card in parameter_cardinality_items:
        if card == 1:
            param_dict[name] = grids_per_mu[0][counter].reshape(-1)
            counter += 1
        if card == 2:
            param_dict[name] = np.concatenate(
                [grids_per_mu[0][counter][..., None],
                 grids_per_mu[1][counter][..., None]], axis=-1).reshape(-1, 2)
            counter += 2

    if model.N_models > 1:
        nested_fractions = grids_per_mu[0][-(N_model_fracts - 1):]
        lin_nested = [f.reshape(-1) for f in nested_fractions]
        N_pts = len(lin_nested[0])
        lin_fractions = np.empty((N_pts, N_model_fracts))
        for i in range(N_pts):
            lin_fractions[i] = nested_to_normalized_fractions(
                np.array([f[i] for f in lin_nested]))
        counter = 0
        for name, _ in list(model.parameter_cardinality.items())[-N_model_fracts:]:
            param_dict[name] = lin_fractions[:, counter]
            counter += 1

    return model.parameters_to_parameter_vector(**param_dict)  # (N_grid, N_params_full)


def build_jax_brute_fn(model, acquisition_scheme, forward_fn, optimizer,
                       Ns=5, N_sphere_samples=30):
    """Build a JIT-compiled brute-force grid search for a MC model.

    Parameters
    ----------
    model : MultiCompartmentModel
    acquisition_scheme : DmipyAcquisitionScheme
    forward_fn : callable
        JIT-compiled forward function returned by build_mc_forward_fn.
        Signature: forward_fn(params_si) -> jnp.array (N_meas,), SI units.
    optimizer : JaxOptimizer
        Used to convert grid points → nested-VF normalized x0 vectors.
    Ns : int
        Grid steps per scalar parameter (default 5, same as brute2fine).
    N_sphere_samples : int
        Orientation grid points on the hemisphere (default 30).

    Returns
    -------
    brute_search : callable (JIT-compiled)
        Signature: brute_search(data_jax) -> x0_nested (N_voxels, N_params_nested)
        data_jax : jnp.array (N_voxels, N_meas), normalised signal attenuation.

    Notes
    -----
    Setup: build param grid (CPU, fast) + evaluate grid signals on GPU (seconds).
    Per-call: one (N_voxels, N_grid) distance-matrix matmul on GPU.
    Memory: avoids (N_voxels, N_grid, N_meas) intermediate via distance-matrix
    identity: MSE[i,j] = (||d_i||² + ||g_j||² - 2 d_i·g_j) / N_meas.
    """
    # ── 1. Build parameter grid on CPU (no signal computation) ───────────────
    param_grid_si = _build_param_grid(model, Ns, N_sphere_samples)
    N_grid  = len(param_grid_si)
    scales  = np.array(model.scales_for_optimization)

    # ── 2. Evaluate forward model for all grid points on GPU ─────────────────
    # Chunked vmap to bound peak GPU memory and avoid OOM on large grids.
    # Each chunk compiles independently (fixed shape → no recompilation within
    # the run), and results are concatenated on CPU.
    chunk_size = 256
    param_grid_jax = jnp.array(param_grid_si, dtype=jnp.float32)
    chunks = []
    for start in range(0, N_grid, chunk_size):
        end   = min(start + chunk_size, N_grid)
        chunk = param_grid_jax[start:end]
        # Pad the last (possibly smaller) chunk so every call has the same
        # shape → avoids a second JIT compilation for the tail.
        if end - start < chunk_size:
            pad   = chunk_size - (end - start)
            chunk = jnp.concatenate(
                [chunk, jnp.zeros((pad, chunk.shape[1]), dtype=jnp.float32)])
            out   = jax.vmap(forward_fn)(chunk)[: end - start]
        else:
            out   = jax.vmap(forward_fn)(chunk)
        chunks.append(jax.block_until_ready(out))
    grid_signals = jnp.concatenate(chunks, axis=0)          # (N_grid, N_meas)
    N_meas = grid_signals.shape[1]

    # ── 3. Preconvert grid → nested-VF normalized x0 (CPU, pure numpy) ──────
    # Do NOT call optimizer._prepare_x0 in a loop — it triggers jax.lax.scan
    # (GPU) 3750 times, exhausting GPU memory.  Convert the whole batch in
    # numpy: divide by scales, then convert VF params to nested fractions.
    p_norm_all = param_grid_si / scales[None, :]          # (N_grid, N_params)
    if optimizer._is_multi:
        N_models   = optimizer._N_models
        n_non_vf   = p_norm_all.shape[1] - N_models
        non_vf_all = p_norm_all[:, :n_non_vf]            # (N_grid, n_non_vf)
        vf_norm    = p_norm_all[:, n_non_vf:]            # (N_grid, N_models)
        # Numpy version of normalized → nested fractions (avoids JAX)
        nested_vf  = np.empty((N_grid, N_models - 1))
        for i in range(N_grid):
            remaining = 1.0
            for j in range(N_models - 1):
                nested_vf[i, j] = vf_norm[i, j] / max(remaining, 1e-15)
                remaining       -= vf_norm[i, j]
        x0_candidates = np.concatenate([non_vf_all, nested_vf], axis=1)
    else:
        x0_candidates = p_norm_all

    # When fit_sigma=True, optimizer._lower/_upper have an extra sigma element
    # at the end.  The brute grid covers model params only, so slice bounds to
    # match x0_candidates width.
    n_model_nested = x0_candidates.shape[1]
    lower_model = np.array(optimizer._lower[:n_model_nested])
    upper_model = np.array(optimizer._upper[:n_model_nested])

    # Fill NaN (orientation placeholder slots) with midpoint of bounds
    mid      = 0.5 * (lower_model + upper_model)
    nan_mask = np.isnan(x0_candidates)
    x0_candidates[nan_mask] = np.broadcast_to(mid, x0_candidates.shape)[nan_mask]

    # Clip 5% inside bounds so LBFGS-B isn't launched from an exact boundary,
    # which can stall the float32 line-search.
    margin = 0.05 * (upper_model - lower_model)
    x0_candidates = np.clip(x0_candidates,
                            lower_model + margin, upper_model - margin)
    x0_candidates_jax = jnp.array(x0_candidates, dtype=jnp.float32)

    # Precompute per-grid sum-of-squares for distance-matrix MSE
    grid_sq = jnp.sum(grid_signals ** 2, axis=-1)          # (N_grid,)

    # ── 4. Return JIT-compiled search function ───────────────────────────────
    @jax.jit
    def brute_search(data_jax):
        """Find best-fit grid starting point per voxel.

        Parameters
        ----------
        data_jax : jnp.array (N_voxels, N_meas)

        Returns
        -------
        jnp.array (N_voxels, N_params_nested)
        """
        data_sq = jnp.sum(data_jax ** 2, axis=-1)          # (N_voxels,)
        cross   = data_jax @ grid_signals.T                 # (N_voxels, N_grid)
        mse = (data_sq[:, None] + grid_sq[None, :] - 2.0 * cross) / N_meas
        best_idx = jnp.argmin(mse, axis=1)                  # (N_voxels,)
        return x0_candidates_jax[best_idx]                  # (N_voxels, N_params_nested)

    return brute_search
