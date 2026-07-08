"""Pure JAX forward functions for dmipy signal models.

Each function is a module-level, side-effect-free function that can be
wrapped with jax.jit directly. All orientation inputs are in Cartesian
coordinates (3-vector); callers must run unitsphere2cart_1d_jax first.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# G1Ball
# ---------------------------------------------------------------------------

def g1ball_signal(bvalues, lambda_iso):
    """Signal attenuation for the Ball (isotropic Gaussian) model.

    Parameters
    ----------
    bvalues : jnp.array, shape (N,)
    lambda_iso : scalar, isotropic diffusivity in m^2/s

    Returns
    -------
    jnp.array, shape (N,)
    """
    return jnp.exp(-bvalues * lambda_iso)


# ---------------------------------------------------------------------------
# C1Stick
# ---------------------------------------------------------------------------

def c1stick_signal(bvalues, gradient_directions, mu_cart, lambda_par):
    """Signal attenuation for the Stick (zero-radius cylinder) model.

    Parameters
    ----------
    bvalues : jnp.array, shape (N,)
    gradient_directions : jnp.array, shape (N, 3)
    mu_cart : jnp.array, shape (3,), Cartesian unit vector
    lambda_par : scalar, parallel diffusivity in m^2/s

    Returns
    -------
    jnp.array, shape (N,)
    """
    cos_theta = jnp.sum(gradient_directions * mu_cart, axis=-1)  # (N,)
    return jnp.exp(-bvalues * lambda_par * cos_theta ** 2)


def c1stick_spherical_mean(bvals, lambda_par):
    """Analytical spherical mean of the Stick model.

    Parameters
    ----------
    bvals : jnp.array, shape (N,) — shell b-values (non-zero)
    lambda_par : scalar

    Returns
    -------
    jnp.array, shape (N,)
    """
    bl = bvals * lambda_par
    sqrt_bl = jnp.sqrt(bl)
    return jnp.where(
        bl > 1e-7,
        jnp.sqrt(jnp.pi) * jax.scipy.special.erf(sqrt_bl) / (2.0 * sqrt_bl),
        jnp.ones_like(bvals),
    )


# ---------------------------------------------------------------------------
# G2Zeppelin
# ---------------------------------------------------------------------------

def g2zeppelin_signal(bvalues, gradient_directions, mu_cart, lambda_par,
                      lambda_perp):
    """Signal attenuation for the Zeppelin (axially symmetric tensor) model.

    Parameters
    ----------
    bvalues : jnp.array, shape (N,)
    gradient_directions : jnp.array, shape (N, 3)
    mu_cart : jnp.array, shape (3,), Cartesian unit vector
    lambda_par : scalar, parallel diffusivity in m^2/s
    lambda_perp : scalar, perpendicular diffusivity in m^2/s

    Returns
    -------
    jnp.array, shape (N,)

    Notes
    -----
    Uses the identity  |g_perp|^2 = |g|^2 - (g·mu)^2  to avoid constructing
    the outer-product perpendicular projector.  This prevents an XLA batched-
    matmul failure ("Too small divisible part") when vmap batch size is small
    (e.g. 1) and the contracting dimension k=3 is not divisible by the cuBLAS
    tile size.
    """
    mag_par     = jnp.sum(gradient_directions * mu_cart, axis=-1)           # (N,)
    mag_sq      = jnp.sum(gradient_directions ** 2, axis=-1)      # (N,)
    mag_perp_sq = jnp.maximum(0.0, mag_sq - mag_par ** 2)        # (N,)
    return jnp.exp(
        -bvalues * (lambda_par * mag_par ** 2 + lambda_perp * mag_perp_sq)
    )


def g2zeppelin_spherical_mean(bvals, lambda_par, lambda_perp):
    """Analytical spherical mean of the Zeppelin model (Kaden et al. 2016).

    Valid when lambda_par > lambda_perp. For the degenerate case this
    falls back to the ball result.

    Parameters
    ----------
    bvals : jnp.array, shape (N,) — shell b-values (non-zero)
    lambda_par : scalar
    lambda_perp : scalar

    Returns
    -------
    jnp.array, shape (N,)
    """
    exp_bl = jnp.exp(-bvals * lambda_perp)
    sqrt_bl = jnp.sqrt(bvals * jnp.abs(lambda_par - lambda_perp))
    # Guard against division by zero when lambda_par == lambda_perp.
    # 1e-30 would be flushed to zero by GPU flush-to-zero on subnormals.
    # Use ones as fallback — the outer jnp.where selects E_iso in that case.
    safe_sqrt_bl = jnp.where(sqrt_bl > 0, sqrt_bl, jnp.ones_like(sqrt_bl))
    E_aniso = (exp_bl * jnp.sqrt(jnp.pi)
               * jax.scipy.special.erf(safe_sqrt_bl) / (2.0 * safe_sqrt_bl))
    # Degenerate case: lambda_par == lambda_perp → isotropic
    E_iso = jnp.exp(-bvals * lambda_par)
    return jnp.where(lambda_par > lambda_perp, E_aniso, E_iso)


# ---------------------------------------------------------------------------
# S2SphereStejskalTannerApproximation
# ---------------------------------------------------------------------------

def s2sphere_signal(qvalues, diameter):
    """Signal attenuation for the Stejskal-Tanner sphere model (Balinov 1993).

    Parameters
    ----------
    qvalues : jnp.array, shape (N,)
    diameter : scalar, sphere diameter in m

    Returns
    -------
    jnp.array, shape (N,)
    """
    radius = diameter / 2.0
    factor = 2.0 * jnp.pi * qvalues * radius
    # limit factor→0: (3/factor^2 * (sin(factor)/factor - cos(factor)))^2 → 1
    safe_factor = jnp.where(factor > 1e-7, factor, jnp.ones_like(factor))
    E_nonzero = (3.0 / safe_factor ** 2 *
                 (jnp.sin(safe_factor) / safe_factor - jnp.cos(safe_factor))) ** 2
    return jnp.where(factor > 1e-7, E_nonzero, jnp.ones_like(qvalues))


# ---------------------------------------------------------------------------
# C2CylinderStejskalTannerApproximation (Soderman)
# ---------------------------------------------------------------------------

def c2cylinder_signal(bvalues, gradient_directions, qvalues,
                      mu_cart, lambda_par, diameter):
    """Signal attenuation for the Soderman cylinder model (SGP + long-time limit).

    Parameters
    ----------
    bvalues : jnp.array, shape (N,)
    gradient_directions : jnp.array, shape (N, 3)
    qvalues : jnp.array, shape (N,)
    mu_cart : jnp.array, shape (3,), Cartesian unit vector
    lambda_par : scalar
    diameter : scalar, cylinder diameter in m

    Returns
    -------
    jnp.array, shape (N,)
    """
    # Parallel attenuation
    cos_theta = jnp.sum(gradient_directions * mu_cart, axis=-1)   # (N,)
    E_parallel = jnp.exp(-bvalues * lambda_par * cos_theta ** 2)

    # Perpendicular q-component magnitude
    mu_outer = jnp.outer(mu_cart, mu_cart)
    perp_plane = jnp.eye(3) - mu_outer
    proj = jnp.dot(perp_plane, gradient_directions.T)   # (3, N)
    q_perp_mag = jnp.sqrt(jnp.sum(proj ** 2, axis=0))  # (N,)
    q_perp = qvalues * q_perp_mag                        # (N,)

    radius = diameter / 2.0
    q_arg = 2.0 * jnp.pi * q_perp * radius              # (N,)

    # E_perp = (2 * J1(q_arg) / q_arg)^2
    # Safe computation: avoid 0/0 when q_arg = 0 (limit → 1)
    safe_q_arg = jnp.where(q_arg > 1e-10, q_arg, jnp.ones_like(q_arg))
    # bessel_jn(z, v=1) returns [J0(z), J1(z)], shape (2, N)
    j1_vals = jax.scipy.special.bessel_jn(safe_q_arg, v=1)[-1]   # (N,)
    E_perp_nonzero = (2.0 * j1_vals / safe_q_arg) ** 2
    E_perpendicular = jnp.where(q_arg > 1e-10, E_perp_nonzero, jnp.ones_like(q_arg))

    return E_parallel * E_perpendicular


# ---------------------------------------------------------------------------
# C4CylinderGaussianPhaseApproximation (Van Gelderen)
# ---------------------------------------------------------------------------

def c4cylinder_signal(bvalues, gradient_directions, gradient_strengths,
                      delta, Delta,
                      mu_cart, lambda_par, diameter,
                      diffusion_perpendicular, gyromagnetic_ratio, roots_jax):
    """Signal attenuation for the Gaussian Phase cylinder model (Van Gelderen 1994).

    Parameters
    ----------
    bvalues : jnp.array, shape (N,)
    gradient_directions : jnp.array, shape (N, 3)
    gradient_strengths : jnp.array, shape (N,)
    delta : jnp.array, shape (N,), pulse length in s
    Delta : jnp.array, shape (N,), pulse separation in s
    mu_cart : jnp.array, shape (3,), Cartesian unit vector
    lambda_par : scalar
    diameter : scalar, cylinder diameter in m
    diffusion_perpendicular : scalar (D), perpendicular diffusivity m^2/s
    gyromagnetic_ratio : scalar (gamma)
    roots_jax : jnp.array, shape (R,), static transcendental roots (J1' zeros)

    Returns
    -------
    jnp.array, shape (N,)
    """
    # Parallel attenuation
    cos_theta = jnp.sum(gradient_directions * mu_cart, axis=-1)
    E_parallel = jnp.exp(-bvalues * lambda_par * cos_theta ** 2)

    # Perpendicular gradient magnitude
    mu_outer = jnp.outer(mu_cart, mu_cart)
    perp_plane = jnp.eye(3) - mu_outer
    proj = jnp.dot(perp_plane, gradient_directions.T)      # (3, N)
    g_perp_mag = jnp.sqrt(jnp.sum(proj ** 2, axis=0))     # (N,)
    g_perp = gradient_strengths * g_perp_mag               # (N,)

    # Van Gelderen summation over transcendental roots
    D = diffusion_perpendicular
    radius = diameter / 2.0
    alpha = roots_jax / radius          # (R,)
    alpha2 = alpha ** 2                 # (R,)
    alpha2D = alpha2 * D                # (R,)

    # Broadcasting: roots axis → (R, 1), measurements axis → (1, N)
    a2D = alpha2D[:, None]             # (R, 1)
    d = delta[None, :]                 # (1, N)
    D_ = Delta[None, :]                # (1, N)

    numer = (
        2.0 * a2D * d - 2.0 +
        2.0 * jnp.exp(-a2D * d) +
        2.0 * jnp.exp(-a2D * D_) -
        jnp.exp(-a2D * (D_ - d)) -
        jnp.exp(-a2D * (D_ + d))
    )                                  # (R, N)
    denom = D ** 2 * alpha2[:, None] ** 3 * (radius ** 2 * alpha2[:, None] - 1.0)
    sum_over_roots = jnp.sum(numer / denom, axis=0)   # (N,)

    first_factor = -2.0 * (g_perp * gyromagnetic_ratio) ** 2
    E_perpendicular = jnp.exp(first_factor * sum_over_roots)

    return E_parallel * E_perpendicular


def s4sphere_pgse_signal_jax(gradient_strength, delta, Delta,
                             diameter, diffusion_constant, roots_jax,
                             gyromagnetic_ratio):
    """JAX-compilable GPA sphere signal for PGSE (Balinov/Stepisnik formula).

    Requires only gradient_strength, delta, Delta — works with schemes built
    from bvalues (no full G(t) waveform needed).

    Parameters
    ----------
    gradient_strength : scalar, T/m
    delta             : scalar, s — gradient pulse duration
    Delta             : scalar, s — pulse separation
    diameter          : scalar, m
    diffusion_constant: scalar, m²/s
    roots_jax         : jnp.array (n_roots,) SPHERE_TRASCENDENTAL_ROOTS
    gyromagnetic_ratio: scalar, rad/(s·T)

    Returns
    -------
    E : scalar, signal attenuation ∈ (0, 1]
    """
    R = diameter / 2.0
    D = diffusion_constant

    alpha    = roots_jax / R                 # (n_roots,)
    alpha2   = alpha ** 2
    alpha2D  = alpha2 * D                    # (n_roots,)

    first_factor = -2.0 * (gyromagnetic_ratio * gradient_strength) ** 2 / D

    summands = (
        alpha ** (-4) / (alpha2 * R ** 2 - 2.0) *
        (
            2.0 * delta - (
                2.0
                + jnp.exp(-alpha2D * (Delta - delta))
                - 2.0 * jnp.exp(-alpha2D * delta)
                - 2.0 * jnp.exp(-alpha2D * Delta)
                + jnp.exp(-alpha2D * (Delta + delta))
            ) / alpha2D
        )
    )
    return jnp.exp(first_factor * jnp.sum(summands))


# ---------------------------------------------------------------------------
# C3CylinderCallaghanApproximation (Callaghan)
# ---------------------------------------------------------------------------

def s4sphere_ogse_signal_jax(G_waveform, dt, diameter, diffusion_constant,
                             roots_jax, gyromagnetic_ratio):
    """JAX-compilable GPA sphere signal from stored G(t) waveform.

    Uses the Stepisnik formulation:
        φ = (γ²/2) Σ_k B_k ∫∫ G(t₁)G(t₂) exp(-a_k D|t₁-t₂|) dt₁ dt₂
        E = exp(-φ)

    Works for PGSE, cosine OGSE, trapezoidal OGSE — any waveform stored in
    the AcquisitionScheme. This is the universal path: no PGSE/OGSE dispatch
    needed in JAX — always numerical.

    Parameters
    ----------
    G_waveform : jnp.array, shape (n_t, 3), T/m — gradient waveform for one
        measurement (projected onto isotropic sphere; uses x-component as proxy).
    dt : float, timestep in seconds
    diameter : float, sphere diameter in m
    diffusion_constant : float, D in m²/s
    roots_jax : jnp.array, shape (n_roots,), SPHERE_TRASCENDENTAL_ROOTS
    gyromagnetic_ratio : float, γ in rad/(s·T)

    Returns
    -------
    E : scalar jnp.float, signal attenuation ∈ (0, 1]

    Notes
    -----
    The causal IIR recursion:
        H_n = H_{n-1} * exp(-a_k D dt) + G(t_n) * dt
    is equivalent to the exact integral and is numerically stable for any
    eigenmode decay rate a_k D. This is JIT-compilable using jax.lax.scan.
    """
    R = diameter / 2.0
    D = diffusion_constant

    # Scalar projection along gradient direction (isotropic sphere: any dir)
    # Use x-component of the waveform as the signed scalar gradient.
    G_t = G_waveform[:, 0]  # (n_t,)

    mu_k = roots_jax                        # (n_roots,)
    lam_k = (mu_k / R) ** 2               # (n_roots,)
    B_k = 2.0 * (R / mu_k) ** 2 / (mu_k ** 2 - 2.0)  # (n_roots,)

    # For each eigenmode k: compute I_k via causal IIR scan.
    # H_n = Σ_{j≤n} G(t_j) exp(-lkD*(t_n - t_j)) dt  (running causal integral)
    # I_k = 2 * Σ_n G(t_n) * H_n * dt
    # We use vmap over roots_jax.

    def compute_I_k(lkd):
        """Compute I_k for a single eigenmode with decay lkd = λ_k D."""
        decay_step = jnp.exp(-lkd * dt)

        def scan_fn(H_prev, G_n):
            H_n = H_prev * decay_step + G_n * dt
            return H_n, H_n

        _, H = jax.lax.scan(scan_fn, jnp.zeros(()), G_t)  # H: (n_t,)
        # Causal sum 2*Σ_{n≥j} overcounts the diagonal (n=j) by 1×; subtract.
        I_k = 2.0 * jnp.dot(G_t * dt, H) - jnp.dot(G_t, G_t) * dt ** 2
        return I_k

    lkD = lam_k * D                       # (n_roots,)
    I_k_all = jax.vmap(compute_I_k)(lkD)  # (n_roots,)

    phi = 0.5 * gyromagnetic_ratio ** 2 * jnp.sum(B_k * I_k_all)
    return jnp.exp(-phi)


def build_c3cylinder_jax_fn(alpha_table, diffusion_perpendicular):
    """Factory: returns a JAX function for the Callaghan cylinder model.

    The alpha table (roots × functions) is static; it is frozen into the
    returned closure so JAX can treat it as a compile-time constant.

    Parameters
    ----------
    alpha_table : np.array, shape (n_roots, n_functions)
        Precomputed Bessel roots from scipy.special.jnp_zeros.
    diffusion_perpendicular : float

    Returns
    -------
    fn : callable (bvalues, gradient_directions, qvalues, tau,
                   mu_cart, lambda_par, diameter) -> jnp.array (N,)
    """
    alpha_jax = jnp.array(alpha_table)           # (n_roots, n_func)
    n_roots, n_func = alpha_table.shape
    D = float(diffusion_perpendicular)
    # Maximum Bessel order needed: J'_{n_func-1}(x) = (J_{n_func-2} - J_{n_func})/2
    max_order = n_func + 1

    def c3cylinder_signal(bvalues, gradient_directions, qvalues, tau,
                          mu_cart, lambda_par, diameter):
        """Callaghan cylinder signal.

        Parameters
        ----------
        bvalues : jnp.array (N,)
        gradient_directions : jnp.array (N, 3)
        qvalues : jnp.array (N,)
        tau : jnp.array (N,)
        mu_cart : jnp.array (3,)
        lambda_par : scalar
        diameter : scalar
        """
        cos_theta = jnp.sum(gradient_directions * mu_cart, axis=-1)
        E_parallel = jnp.exp(-bvalues * lambda_par * cos_theta ** 2)

        mu_outer = jnp.outer(mu_cart, mu_cart)
        perp_plane = jnp.eye(3) - mu_outer
        proj = jnp.dot(perp_plane, gradient_directions.T)
        q_perp_mag = jnp.sqrt(jnp.sum(proj ** 2, axis=0))
        q_perp = qvalues * q_perp_mag

        radius = diameter / 2.0
        q_arg = 2.0 * jnp.pi * q_perp * radius    # (N,)
        q2 = q_arg ** 2                             # (N,)

        # Compute all Bessel orders needed in one call.
        # Use safe_q_arg to avoid issues at q_arg=0; we handle edge cases below.
        safe_q_arg = jnp.where(q_arg > 1e-10, q_arg, jnp.ones_like(q_arg))
        # J_all: shape (max_order+1, N) = (n_func+2, N)
        J_all = jax.scipy.special.bessel_jn(safe_q_arg, v=max_order)

        # --- m = 0 contribution ---
        # res_0 = sum_k [ 4 * exp(-alpha2[k,0] * D * tau / r^2)
        #                 * q2 * J1^2 / (q2 - alpha2[k,0])^2 ]
        alpha2_0 = alpha_jax[:, 0] ** 2           # (n_roots,)
        J1 = J_all[1]                              # (N,)
        # tau: (N,), alpha2_0: (n_roots,) → broadcast to (n_roots, N)
        exp_0 = jnp.exp(-alpha2_0[:, None] * D * tau[None, :] / radius ** 2)
        numer_0 = 4.0 * exp_0 * q2[None, :] * J1[None, :] ** 2
        denom_0_raw = (q2[None, :] - alpha2_0[:, None]) ** 2
        # Safe: avoid 0/0 when q2==0 and alpha2==0 (k=0 row where alpha[0,0]=0)
        denom_0 = jnp.where(denom_0_raw > 1e-60, denom_0_raw, jnp.ones_like(denom_0_raw))
        # numer is also 0 when q2=0, so 0/1 = 0 ✓
        res = jnp.sum(numer_0 / denom_0, axis=0)  # (N,)

        # --- m > 0 contributions ---
        # For each m: J'_m(x) = (J_{m-1}(x) - J_{m+1}(x)) / 2
        for m in range(1, n_func):
            alpha2_m = alpha_jax[:, m] ** 2        # (n_roots,)
            Jderiv_m = (J_all[m - 1] - J_all[m + 1]) / 2.0   # (N,)
            q_Jd_sq = (q_arg * Jderiv_m) ** 2     # (N,)

            exp_m = jnp.exp(
                -alpha2_m[:, None] * D * tau[None, :] / radius ** 2
            )
            # 8 * exp * alpha2 / (alpha2 - m^2) * (q * J'_m)^2 / (q2 - alpha2)^2
            coeff = 8.0 * alpha2_m[:, None] / (alpha2_m[:, None] - float(m) ** 2)
            numer_m = coeff * exp_m * q_Jd_sq[None, :]
            denom_m_raw = (q2[None, :] - alpha2_m[:, None]) ** 2
            denom_m = jnp.where(denom_m_raw > 1e-60, denom_m_raw,
                                jnp.ones_like(denom_m_raw))
            res = res + jnp.sum(numer_m / denom_m, axis=0)

        # q_perp = 0 → E_perp = 1 (res should be 0 there but guard anyway)
        E_perpendicular = jnp.where(q_perp > 1e-10, res, jnp.ones_like(q_perp))
        return E_parallel * E_perpendicular

    return c3cylinder_signal
