"""Loss / likelihood functions for JAX-based fitting.

All functions are factories that return a callable with signature:
    loss_fn(E_model, data) -> scalar

where E_model and data are 1-D JAX arrays of shape (N_measurements,),
and the returned scalar is the quantity to *minimise*.

Conventions:
- Signals are normalised to S₀ = 1.  E_model ∈ [0, 1], data ∈ [0, ∞).
- sigma is the noise standard deviation in the same normalised units;
  sigma = 1 / SNR₀ where SNR₀ is the SNR at b = 0.
- All functions are pure JAX (no Python control flow), JIT-compatible,
  and produce finite, well-behaved gradients for E_model ≥ 0.

Usage
-----
    from dmipy_fit.jax.losses_jax import mse_loss, rician_nll

    # Default (reproduces existing behaviour exactly)
    model.fit(scheme, data, solver='jax')

    # Rician MLE with known sigma
    model.fit(scheme, data, solver='jax', loss_fn=rician_nll(sigma=0.05))
"""

import jax
import jax.numpy as jnp


def _log_ive_n(n, x):
    """log(ive_n(x)) = log(exp(-x) · I_n(x)) for Python-int n ≥ 0, JAX array x ≥ 0.

    Uses the Miller–Wallis backward ratio recurrence:

        r_{k-1}(x) = 1 / (2k/x + r_k(x)),   r_N = 0

    where r_k = I_{k+1}(x) / I_k(x).  All r_k ∈ (0, 1), so there is no
    overflow for any x > 0, regardless of n or N.

    Then:
        log(I_n / I_0) = Σ_{k=0}^{n-1} log(r_k)
        log(ive_n(x))  = log(I_n/I_0) + log(i0e(x))

    At x = 0: all r_k → 0, so log(ive_n(0)) → −∞ for n ≥ 1, which is
    the correct limit (I_n(0) = 0 for n ≥ 1).
    """
    if n == 0:
        return jnp.log(jax.scipy.special.i0e(x) + 1e-38)
    if n == 1:
        return jnp.log(jax.scipy.special.i1e(x) + 1e-38)

    N_EXTRA = 40          # extra steps beyond order n; convergence is geometric
    N = n + N_EXTRA       # total backward steps (compile-time constant)

    # At x = 0: 2k / x_safe ≫ r_k, so r_{k-1} ≈ 0 for all k.
    # This correctly yields log(ive_n(0)) ≈ n · log(1e-38) ≪ 0.
    x_safe = jnp.maximum(x, 1e-30)

    def step(r_k, k):
        """k is a float element of k_vals = [N, N-1, ..., 1]."""
        r_km1 = 1.0 / (2.0 * k / x_safe + r_k)
        return r_km1, r_km1   # (new carry, output)

    k_vals = jnp.arange(N, 0, -1, dtype=x.dtype)   # shape (N,): N, N-1, …, 1
    _, r_vals = jax.lax.scan(step, jnp.zeros_like(x), k_vals)
    # r_vals[j] = r_{N-j-1}  (ratio I_{N-j}/I_{N-j-1} after convergence)
    # We need log(r_0) + log(r_1) + … + log(r_{n-1}) = log(I_n / I_0).
    # r_0 = r_vals[N-1], …, r_{n-1} = r_vals[N-n]  →  slice r_vals[N-n : N].

    log_ratio_n_0 = jnp.sum(jnp.log(r_vals[N - n:N] + 1e-38))
    log_ive_0 = jnp.log(jax.scipy.special.i0e(x) + 1e-38)
    return log_ratio_n_0 + log_ive_0


def mse_loss():
    """Mean squared error loss.  Exactly reproduces the prior default.

    Returns
    -------
    loss_fn : callable (E_model, data) -> scalar
        Mean of squared residuals.
    """
    def loss(E_model, data):
        diff = E_model - data
        return jnp.dot(diff, diff) / data.shape[0]
    return loss


def rician_nll(sigma):
    """Rician negative log-likelihood (NLL) loss factory.

    For magnitude MRI with single-coil or pre-combined reconstruction,
    the observed signal r follows Rice(ν, σ) where ν = E_model.

    PDF:  p(r | ν, σ) = (r/σ²) · exp(-(r² + ν²)/(2σ²)) · I₀(rν/σ²)

    NLL per measurement (dropping r-only constants for gradient):
        L(ν) = (r² + ν²)/(2σ²) - log(I₀(rν/σ²))

    Numerically stable via scaled Bessel:
        I₀(x) = exp(|x|) · i0e(x)  →  log I₀(x) = |x| + log(i0e(x))

    So:
        L(ν) = (r² + ν²)/(2σ²) - |rν/σ²| - log(i0e(rν/σ²))

    At high SNR (ν >> σ), this reduces to the Gaussian MSE form.
    At ν = 0, reduces to Rayleigh NLL (gradient well-behaved).

    Parameters
    ----------
    sigma : float
        Noise standard deviation in normalised signal units (= 1/SNR₀).
        Must be positive.

    Returns
    -------
    loss_fn : callable (E_model, data) -> scalar
        Mean Rician NLL across measurements.

    Raises
    ------
    ValueError
        If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    _sigma = float(sigma)
    _sigma2 = _sigma ** 2

    def loss(E_model, data):
        sig2  = jnp.array(_sigma2, dtype=E_model.dtype)
        nu    = jnp.clip(E_model, 0.0, None)   # model predicts ≥ 0
        r     = data
        arg   = nu * r / sig2                   # (N_meas,)
        # log I₀(x) = |x| + log(i0e(x)); i0e avoids overflow for large |x|
        log_i0 = jnp.abs(arg) + jnp.log(jax.scipy.special.i0e(arg) + 1e-38)
        # Include 2*log(sigma) to match rician_nll_fittable (same constant
        # for fixed sigma; does not affect gradient w.r.t. model params).
        log_sig = jnp.array(2.0 * jnp.log(_sigma), dtype=E_model.dtype)
        nll_per = log_sig + (r ** 2 + nu ** 2) / (2.0 * sig2) - log_i0
        return jnp.mean(nll_per)

    return loss


def rician_nll_fittable():
    """Rician NLL loss where sigma is passed explicitly as a third argument.

    Unlike rician_nll(sigma), sigma is NOT captured in a closure here —
    it is passed at call time.  This allows the optimizer to treat sigma
    as a fittable parameter by appending it to the parameter vector.

    Returns
    -------
    loss_fn : callable (E_model, data, sigma) -> scalar
        Mean Rician NLL.  sigma is the noise standard deviation in the
        same normalised signal units as E_model and data (= 1/SNR₀).
        sigma is clamped to >= 1e-6 internally so the loss is finite
        everywhere the optimizer might explore.

    Notes
    -----
    Numerically identical to rician_nll(sigma) when sigma is constant.
    The guard jnp.maximum(sigma, 1e-6) prevents NaN/Inf for near-zero
    sigma (e.g. zero-padded voxels in vmap batches).
    """
    def loss(E_model, data, sigma):
        sig    = jnp.maximum(sigma, jnp.array(1e-6, dtype=E_model.dtype))
        sig2   = sig ** 2
        nu     = jnp.clip(E_model, 0.0, None)      # model predicts >= 0
        r      = data
        arg    = nu * r / sig2                      # (N_meas,)
        log_i0 = jnp.abs(arg) + jnp.log(jax.scipy.special.i0e(arg) + 1e-38)
        # Include 2*log(sigma) term: required when sigma is fittable.
        # Without it, dL/dsigma < 0 everywhere and sigma drifts to its bound.
        nll_per = 2.0 * jnp.log(sig) + (r ** 2 + nu ** 2) / (2.0 * sig2) - log_i0
        return jnp.mean(nll_per)

    return loss


def rician_nll_sm_fittable(w_norms):
    """Rician NLL for spherical mean data with per-shell effective sigma.

    The spherical mean y_shell = w_shell^T · r_shell is a linear combination
    of N_dirs Rician-distributed measurements.  By the linear-transform
    property of Rician noise:

        y_shell ~ Rice(ν_sm, σ_eff_shell)
        σ_eff_shell = σ · ‖w_shell‖₂

    where σ is the noise standard deviation on the raw directional data and
    ‖w_shell‖₂ is the L2 norm of the L=0 zonal harmonic weight row for the
    shell (precomputed from the pseudoinverse of the SH design matrix).

    Parameters
    ----------
    w_norms : array-like, shape (N_shells,)
        Per-shell L2 norm of the L=0 zonal harmonic weight vector.
        Compute with ``compute_sm_w_norms(acquisition_scheme)``.

    Returns
    -------
    loss_fn : callable (E_model, data_sm, sigma) -> scalar
        Mean Rician NLL across shells.  sigma is the noise std on the raw
        directional signal (= 1/SNR₀); per-shell effective sigma is derived
        internally.  sigma is clamped to >= 1e-6.
    """
    import numpy as _np
    _w = _np.asarray(w_norms, dtype=_np.float64)

    def loss(E_model, data_sm, sigma):
        sig_eff = (jnp.maximum(sigma, jnp.array(1e-6, dtype=E_model.dtype))
                   * jnp.array(_w, dtype=E_model.dtype))
        sig2   = sig_eff ** 2
        nu     = jnp.clip(E_model, 0.0, None)
        r      = data_sm
        arg    = nu * r / sig2
        log_i0 = jnp.abs(arg) + jnp.log(jax.scipy.special.i0e(arg) + 1e-38)
        nll_per = 2.0 * jnp.log(sig_eff) + (r ** 2 + nu ** 2) / (2.0 * sig2) - log_i0
        return jnp.mean(nll_per)

    return loss


def nc_chi_nll(sigma, n_coils):
    """Noncentral-chi NLL loss for multi-coil SOS magnitude reconstruction.

    For L coils with equal sensitivity, the SOS magnitude follows a
    noncentral chi distribution with 2L degrees of freedom.
    The PDF involves the modified Bessel function I_{L-1}.

    This is an exact implementation for integer L ≥ 1.
    For L = 1 it reduces to the Rician NLL.

    Parameters
    ----------
    sigma : float
        Per-coil noise standard deviation in normalised units.
    n_coils : int
        Number of coils L ≥ 1.

    Returns
    -------
    loss_fn : callable (E_model, data) -> scalar
        Mean NC-chi NLL across measurements.

    Notes
    -----
    PDF: p(r | ν, σ, L) = (r/σ²)(r/ν)^(L-1) exp(-(r²+ν²)/(2σ²)) I_{L-1}(rν/σ²)
    NLL: (r²+ν²)/(2σ²) - (L-1)log(r/ν) - log(I_{L-1}(rν/σ²))

    For large arguments, use the scaled Bessel: iv(L-1, x) = exp(-x)*ive(L-1, x),
    so log(I_{L-1}(x)) = x + log(ive(L-1, x)).

    For L=1: I₀(x) = i0(x), ive(0, x) = i0e(x) → reduces to rician_nll.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n_coils < 1:
        raise ValueError(f"n_coils must be >= 1, got {n_coils}")

    _sigma2  = float(sigma) ** 2
    _L       = int(n_coils)
    _n_order = _L - 1          # Bessel order (Python int; used to build the scan)

    def loss(E_model, data):
        sig2  = jnp.array(_sigma2, dtype=E_model.dtype)
        L     = jnp.array(_L,      dtype=E_model.dtype)
        nu    = jnp.clip(E_model, 1e-10, None)   # guard log(ν) in ratio term
        r     = data
        arg   = nu * r / sig2                     # (N_meas,)

        # log I_{L-1}(arg) = |arg| + log(ive_{L-1}(arg))
        # _log_ive_n uses Miller–Wallis ratios: no overflow for any arg ≥ 0.
        log_iLm1 = jnp.abs(arg) + _log_ive_n(_n_order, arg)

        nll_per = ((r ** 2 + nu ** 2) / (2.0 * sig2)
                   - (L - 1.0) * jnp.log(jnp.where(r > 0, r / nu, 1.0))
                   - log_iLm1)
        return jnp.mean(nll_per)

    return loss
