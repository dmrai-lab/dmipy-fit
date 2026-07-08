"""vmap-based voxel-parallel fitting via jaxopt.LBFGSB.

Architecture
------------
build_vmap_fitter(jax_optimizer)
    Compiles a jax.jit + jax.vmap kernel that runs jaxopt.LBFGSB
    independently for every voxel in a batch.  All voxels execute in
    parallel — on GPU the forward pass and gradient for the entire batch
    are evaluated in a single XLA kernel at each L-BFGS-B step.

vmap_fit(jax_optimizer, data_all, x0_all, batch_size)
    Splits voxels into batches, calls the compiled kernel per batch,
    and returns a (N_voxels, N_params_nested) result array.

How it differs from the old scipy-per-voxel loop
-------------------------------------------------
Old:  for each voxel → scipy.minimize(jax_grad_fn) → sequential, Python overhead
New:  jax.vmap(jaxopt.LBFGSB.run)(x0_all, data_all) → one compiled kernel,
      all voxels step together, GPU-parallel forward/grad evaluation.

Each voxel still runs its own L-BFGS-B trajectory and can converge in a
different number of steps.  jaxopt handles this by masking converged
voxels so they perform no-ops until the slowest voxel finishes.

Batching
--------
batch_size controls how many voxels are compiled into one kernel call.
Larger batches amortise Python overhead and fill GPU SIMD units more
efficiently.  On CPU the benefit is smaller but compilation is only
triggered once per unique batch shape.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import LBFGSB


def _default_tol():
    """Return a convergence tolerance appropriate for the current JAX dtype.

    jaxopt.LBFGSB defaults to float32 on GPU.  float32 machine eps is ~1.2e-7,
    so requesting tol=1e-8 (tighter than eps) causes the solver to exhaust all
    maxiter iterations without ever converging — wasting compute.  Using 1e-5
    is comfortably achievable in float32 while still being tight enough for
    scientific accuracy.  float64 retains the original tight default.
    """
    if jnp.ones(1).dtype == jnp.float32:
        return 1e-5
    return 1e-8


def build_vmap_fitter(jax_optimizer, tol=None, dtype=None, use_weights=False):
    """Return a jit+vmap'd function that fits a batch of voxels at once.

    Parameters
    ----------
    jax_optimizer : JaxOptimizer
    tol : float or None
        Gradient-norm convergence tolerance passed to jaxopt.LBFGSB.
        If None, a dtype-aware default is used (1e-5 for float32, 1e-8 for
        float64).
    dtype : jnp.dtype or None
        Dtype for bounds and initial parameters. Defaults to jnp.float32.
        Pass jnp.float64 (with jax_enable_x64=True) for reference runs.
    use_weights : bool
        If True, the returned ``fit_batch`` accepts a third argument
        ``weights_batch`` of shape ``(B, N_meas)`` and passes it through
        to the loss function as per-measurement weights.
        If False (default), ``fit_batch`` has the original two-argument
        signature and weights are not used — bit-identical to prior behaviour.

    Returns
    -------
    fit_batch : callable
        When use_weights=False (default):
            fit_batch(x0_batch, data_batch) -> fitted_params_batch
        When use_weights=True:
            fit_batch(x0_batch, data_batch, weights_batch) -> fitted_params_batch
        x0_batch      : jnp.array (B, N_params_nested)
        data_batch    : jnp.array (B, N_meas)
        weights_batch : jnp.array (B, N_meas)  [only when use_weights=True]
        returns       : jnp.array (B, N_params_nested)
    """
    if tol is None:
        tol = _default_tol()
    if dtype is None:
        dtype = jnp.float32

    # Bounds must match x0 dtype so lax.while_loop carry types are consistent.
    lower  = jnp.array(jax_optimizer._lower, dtype=dtype)
    upper  = jnp.array(jax_optimizer._upper, dtype=dtype)
    bounds = (lower, upper)

    solver = LBFGSB(
        fun=jax_optimizer._loss_fn_jax,
        maxiter=jax_optimizer.maxiter,
        tol=tol,
        implicit_diff=True,   # uses lax.while_loop — compiles once, fast at runtime
        # implicit_diff=False would unroll the entire loop at compile time (slow)
    )

    if use_weights:
        @jax.jit
        def fit_batch(x0_batch, data_batch, weights_batch):
            """Fit a batch of B voxels in parallel with per-voxel weights.

            x0_batch      : (B, N_params_nested)
            data_batch    : (B, N_meas)
            weights_batch : (B, N_meas)
            returns       : (B, N_params_nested)
            """
            results = jax.vmap(
                lambda x0, data, weights: solver.run(
                    x0, bounds=bounds, data=data, weights=weights)
            )(x0_batch, data_batch, weights_batch)
            return results.params
    else:
        @jax.jit
        def fit_batch(x0_batch, data_batch):
            """Fit a batch of B voxels in parallel.

            x0_batch   : (B, N_params_nested)
            data_batch : (B, N_meas)
            returns    : (B, N_params_nested)
            """
            results = jax.vmap(
                lambda x0, data: solver.run(x0, bounds=bounds, data=data)
            )(x0_batch, data_batch)
            return results.params

    return fit_batch


def vmap_fit(jax_optimizer, data_all, x0_all, batch_size=None, tol=None,
             dtype=None, weights_all=None):
    """Fit all voxels using jaxopt.LBFGSB vectorised over voxels via jax.vmap.

    Parameters
    ----------
    jax_optimizer : JaxOptimizer
    data_all : np.array, shape (N_voxels, N_meas)
        Normalised signal attenuation per voxel.
    x0_all : np.array, shape (N_voxels, N_params_nested)
        Initial guess per voxel (nested-VF, normalised space).
    batch_size : int or None
        Voxels per compiled kernel call.  None = all at once.
        On GPU: use a large value (512–4096) to saturate SIMD.
        On CPU: smaller values (32–256) avoid memory pressure.
    tol : float or None
        Convergence tolerance forwarded to jaxopt.LBFGSB.
        If None, uses 1e-5 for float32 or 1e-8 for float64.
    dtype : jnp.dtype or None
        Computation dtype. Defaults to jnp.float32 (GPU production).
        Pass jnp.float64 with jax_enable_x64=True for reference validation.
    weights_all : np.array or None, shape (N_voxels, N_meas)
        Optional per-voxel, per-measurement weights.  Each weight multiplies
        the corresponding measurement's contribution to the loss before
        summing, so outlier volumes can be down-weighted per voxel.
        ``weights_all=None`` (default) produces bit-identical results to
        unweighted fitting — no overhead on the existing code path.

    Returns
    -------
    fitted_params : np.array, shape (N_voxels, N_params_nested)
        Fitted parameters in nested-VF, normalised space.
    """
    N_voxels = data_all.shape[0]
    if batch_size is None:
        # A monolithic vmap over ALL voxels compiles one enormous
        # jaxopt.LBFGSB while_loop kernel (slow compile, large GPU memory, and
        # the batch runs until the slowest voxel converges).  Sub-batch at a
        # fixed size so the kernel compiles ONCE and is reused per chunk
        # (bounded memory, progress feedback).  Tune with DMIPY_JAX_BATCH.
        import os
        batch_size = min(N_voxels, int(os.environ.get("DMIPY_JAX_BATCH", "8192")))
    if dtype is None:
        # Infer from x0_all: float32 when x0 comes from GPU brute-force search
        # (production path); float64 when x0 comes from numpy _prepare_x0
        # (reference / SM path).  This preserves existing precision behaviour
        # instead of silently downgrading float64 inputs.
        arr_dtype = np.asarray(x0_all).dtype
        dtype = jnp.float32 if arr_dtype == np.float32 else jnp.float64

    use_weights = weights_all is not None
    fit_batch = build_vmap_fitter(
        jax_optimizer, tol=tol, dtype=dtype, use_weights=use_weights)
    all_results = np.empty_like(x0_all)

    try:
        from jaxlib.xla_extension import XlaRuntimeError
    except Exception:
        XlaRuntimeError = RuntimeError

    def _run_all(bs):
        """Fit every voxel in chunks of ``bs``. Overwrites all_results in place."""
        n_batches = -(-N_voxels // bs)
        try:
            from tqdm import tqdm
            _rng = tqdm(range(0, N_voxels, bs), total=n_batches,
                        desc="JAX LBFGS-B vmap", unit="batch")
        except Exception:
            _rng = range(0, N_voxels, bs)
        for start in _rng:
            end = min(start + bs, N_voxels)
            B   = end - start

            x0_b   = jnp.array(x0_all[start:end], dtype=dtype)
            data_b = jnp.array(data_all[start:end], dtype=dtype)

            # Pad last batch to bs so the compiled kernel shape stays constant
            # across calls — avoids recompilation for the tail.
            if B < bs:
                pad    = bs - B
                x0_b   = jnp.concatenate([x0_b,   jnp.zeros((pad, x0_b.shape[1]),   dtype=dtype)])
                data_b = jnp.concatenate([data_b, jnp.zeros((pad, data_b.shape[1]), dtype=dtype)])

            if use_weights:
                w_b = jnp.array(weights_all[start:end], dtype=dtype)
                if B < bs:
                    pad = bs - B
                    # Pad weight rows with ones so padded voxels don't cause NaN
                    # from zero-sum weights (those results are discarded anyway).
                    w_b = jnp.concatenate(
                        [w_b, jnp.ones((pad, w_b.shape[1]), dtype=dtype)])
                fitted = np.array(fit_batch(x0_b, data_b, w_b))
            else:
                fitted = np.array(fit_batch(x0_b, data_b))

            all_results[start:end] = fitted[:B]

    # Adaptive batch: on GPU OOM, halve the batch and retry (down to 1) rather
    # than dying — lets the same code run on a small GPU. Pin with DMIPY_JAX_BATCH.
    bs = batch_size
    while True:
        try:
            _run_all(bs)
            return all_results
        except (XlaRuntimeError, RuntimeError) as exc:
            msg = str(exc)
            is_oom = 'RESOURCE_EXHAUSTED' in msg or 'out of memory' in msg.lower()
            if not is_oom or bs <= 1:
                raise
            new_bs = max(1, bs // 2)
            import warnings
            warnings.warn(
                "JAX fit hit GPU OOM at batch_size={}; retrying at {}. "
                "Set DMIPY_JAX_BATCH to pin a size.".format(bs, new_bs),
                RuntimeWarning, stacklevel=2)
            bs = new_bs
