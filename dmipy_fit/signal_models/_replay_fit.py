"""Fast, exact forward evaluation of a Monte-Carlo *replay pack* for a FIXED acquisition scheme.

A replay pack stores each walker's trajectory as truncated DCT-II coefficients ``dct_coeffs`` (N_w, K, 3)
and a spin weight. The diffusion-weighted signal is

    E = < w_i exp(i phi_i) > / < w_i > ,   phi_i(m) = gamma * dt * sum_t G_m(t) . r_i(t)

with r_i(t) the (idct of the) stored trajectory. Because the position enters the phase *linearly* and the
DCT-II is orthonormal, Parseval gives

    phi_i(m) = sum_{k,c} C_{i,k,c} * Ghat_{m,k,c},   Ghat = gamma * dt * DCT(G_m)[:K]

so the waveform's projection onto the pack's temporal basis, ``Ghat``, is **independent of the walkers and
of the fitted parameters** and is compiled ONCE per acquisition scheme (:func:`compile_scheme`). Each
forward evaluation is then a single dense matmul ``C @ W`` followed by a weighted complex mean
(:func:`replay_complex`) — mathematically identical to the engine's ``pack.replay`` (verified to ~1e-14),
but the per-call cost collapses from a trajectory reconstruction to one BLAS-3 call. This is what makes
fitting (same scheme reused across every voxel and optimizer iteration) tractable.

Surface relaxivity is handled *exactly* here, not as an analytic ``exp(-TE rho S/V)`` tag-on: it is a
per-walker reweighting by the stored boundary local time (``blt_dct``), optionally coherence-gated by a
transverse-occupancy schedule ``chi(t)`` — see :func:`surface_logweight`.
"""
import numpy as np

__all__ = ["compile_scheme", "replay_complex", "surface_logweight",
           "replay_complex_jax", "replay_batch_jax"]


def compile_scheme(G_pack, dt, K, gyromagnetic_ratio):
    """Compile a fixed acquisition scheme into its temporal-basis projection ``W`` (3K, n_meas).

    Parameters
    ----------
    G_pack : (n_meas, n_t, 3) array   waveform resampled onto the pack save grid [T/m]
    dt : float                        pack save interval [s]
    K : int                           number of DCT modes the pack stores
    gyromagnetic_ratio : float        [rad/s/T]

    Returns
    -------
    (3K, n_meas) float64 array — reuse across all packs on this grid and all fit iterations.
    """
    from scipy.fft import dct
    G_pack = np.asarray(G_pack, np.float64)
    Ghat = dct(G_pack, type=2, norm="ortho", axis=1)[:, :K, :]      # (n_meas, K, 3)
    n_meas = Ghat.shape[0]
    return (gyromagnetic_ratio * dt * Ghat).reshape(n_meas, K * 3).T   # (3K, n_meas)


def surface_logweight(blt_dct, rho_over_D, n_t, chi_hat=None):
    """Per-walker surface log-weight (rho/D) * sum_t chi(t) l_i(t) from the boundary-local-time DCT
    coefficients ``blt_dct`` (N_w, Kb). Un-gated (chi=None): the exact total contact is sqrt(n_t)*beta0
    (the DC coefficient). Coherence-gated: contract with the DCT of chi(t) (Parseval). Mirrors
    ``dmipy_sim.compression.surface_logweight_dct``."""
    blt = np.asarray(blt_dct, np.float64)
    if chi_hat is None:
        s = np.sqrt(n_t) * blt[:, 0]
    else:
        chi_hat = np.asarray(chi_hat, np.float64)[: blt.shape[1]]
        s = blt[:, : chi_hat.shape[0]] @ chi_hat
    return rho_over_D * s


def replay_complex(coeffs, spin_weights, W, *, blt_dct=None, rho_over_D=0.0, n_t=None, chi_hat=None):
    """Complex diffusion-weighted signal ``< w exp(i phi) > / < w >`` for compiled scheme ``W``.

    ``coeffs`` is the pack's ``dct_coeffs`` (N_w, K, 3); ``W`` is from :func:`compile_scheme`. When
    ``blt_dct`` + ``rho_over_D`` are given, the walker weights are multiplied by ``exp(surface_logweight)``
    — the exact coherence-gated surface-relaxivity replay (``chi_hat`` = DCT of the occupancy schedule).
    Returns a complex array (n_meas,); take ``abs`` for magnitude (kept complex for diameter interpolation).
    """
    coeffs = np.asarray(coeffs, np.float64)
    N_w, K, _ = coeffs.shape
    phi = coeffs.reshape(N_w, K * 3) @ W                            # (N_w, n_meas)
    w0 = np.asarray(spin_weights, np.float64)                       # original weights (normalization)
    w_eff = w0
    if blt_dct is not None and rho_over_D:
        # surface relaxivity is a per-walker signal LOSS: it decays the whole signal (incl. b=0), so it
        # weights the numerator but the denominator stays the original <w>. E(b0) = <w e^s>/<w> < 1.
        w_eff = w0 * np.exp(surface_logweight(blt_dct, rho_over_D, n_t, chi_hat))
    num = (w_eff[:, None] * np.exp(1j * phi)).sum(0)
    return num / w0.sum()


# ------------------------------- JAX / GPU forward (Tier 2) -------------------------------
# The compiled-scheme forward `E = <w_eff exp(i C@W)> / <w0>` is a dense matmul + reduction, so it maps
# directly onto the GPU and is differentiable. `replay_batch_jax` vmaps it over a batch of compiled
# schemes `W` (e.g. one per candidate orientation the optimizer evaluates), turning a whole grid of
# forward evaluations into a single GPU call. Requires the `[jax]` extra. Reference precision: enable
# jax_enable_x64 (the fit backend does globally); otherwise complex64 (~1e-5 vs the NumPy path).

def _weff(spin_weights, blt_dct, rho_over_D, n_t, chi_hat, jnp):
    w0 = jnp.asarray(spin_weights)
    if blt_dct is None or not rho_over_D:
        return w0, w0
    blt = jnp.asarray(blt_dct)
    if chi_hat is None:
        s = jnp.sqrt(n_t) * blt[:, 0]
    else:
        ch = jnp.asarray(chi_hat)[: blt.shape[1]]
        s = blt[:, : ch.shape[0]] @ ch
    return w0, w0 * jnp.exp(rho_over_D * s)


def replay_complex_jax(coeffs, spin_weights, W, *, blt_dct=None, rho_over_D=0.0, n_t=None, chi_hat=None):
    """JAX twin of :func:`replay_complex` — jittable/differentiable single-scheme forward.
    ``coeffs`` (N_w, K, 3), ``W`` (3K, n_meas). Returns a complex (n_meas,) array."""
    import jax.numpy as jnp
    C = jnp.asarray(coeffs)
    N_w, K, _ = C.shape
    phi = C.reshape(N_w, K * 3) @ jnp.asarray(W)                    # (N_w, n_meas)
    w0, w_eff = _weff(spin_weights, blt_dct, rho_over_D, n_t, chi_hat, jnp)
    num = (w_eff[:, None] * jnp.exp(1j * phi)).sum(0)
    return num / w0.sum()


def replay_batch_jax(coeffs, spin_weights, W_batch, *, blt_dct=None, rho_over_D=0.0, n_t=None, chi_hat=None):
    """Vmap :func:`replay_complex_jax` over a batch of compiled schemes ``W_batch`` (n_batch, 3K, n_meas)
    — one GPU call for a whole grid of forward evaluations (e.g. candidate orientations in a fit).
    Returns magnitudes (n_batch, n_meas)."""
    import jax
    import jax.numpy as jnp
    fn = lambda W: replay_complex_jax(coeffs, spin_weights, W, blt_dct=blt_dct,
                                      rho_over_D=rho_over_D, n_t=n_t, chi_hat=chi_hat)
    return jnp.abs(jax.vmap(fn)(jnp.asarray(W_batch)))
