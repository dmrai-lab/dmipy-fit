"""JAX forward functions for two-compartment Karger/NEXI exchange models.

All functions are pure, side-effect-free, and JIT-compilable.
No Python-level branching on array values; all conditionals use jnp.where.

jax.scipy.linalg.expm is available in JAX >= 0.4.1 and is used for the
matrix-exponential Kärger propagators (_karger_matrix_se_jax,
_karger_matrix_ste_jax).  These are used when relaxation or finite-RF
parameters are provided.  The scalar eigenvalue formula (karger_signal) is
retained for the no-relaxation, instantaneous-RF fast path.

References
----------
Jelescu IO, et al. (2022). NeuroImage 256, 119277.
doi:10.1016/j.neuroimage.2022.119277
"""

import jax.numpy as jnp
import jax.scipy.linalg as jsl


def karger_signal(bvalues, delta, Delta, gradient_directions,
                  mu_cart, Di, De_par, De_perp, f, kappa):
    """Signal attenuation for the Karger two-compartment exchange model.

    Parameters
    ----------
    bvalues : jnp.array, shape (N,), s/m^2
    delta : jnp.array, shape (N,), pulse length in s
    Delta : jnp.array, shape (N,), pulse separation in s
    gradient_directions : jnp.array, shape (N, 3)
    mu_cart : jnp.array, shape (3,), Cartesian unit vector
    Di : scalar, intra-neurite parallel diffusivity in m^2/s
    De_par : scalar, extra-neurite parallel diffusivity in m^2/s
    De_perp : scalar, extra-neurite perpendicular diffusivity in m^2/s
    f : scalar, intra-neurite volume fraction
    kappa : scalar, exchange rate in s^-1

    Returns
    -------
    jnp.array, shape (N,)
    """
    # Effective diffusion time
    t_d = Delta - delta / 3.0          # (N,)

    # Angular projections
    cos_theta = jnp.sum(gradient_directions * mu_cart, axis=-1)   # (N,)
    cos2 = cos_theta ** 2
    sin2 = 1.0 - cos2

    # Projected diffusivities
    Di_n = Di * cos2                          # (N,)
    De_n = De_par * cos2 + De_perp * sin2    # (N,)

    # Decay exponents
    Ri = bvalues * Di_n       # (N,)
    Re = bvalues * De_n       # (N,)

    return karger_from_Ri_Re(t_d, Ri, Re, f, kappa)


def karger_from_Ri_Re(t_d, Ri, Re, f, kappa):
    """Shared Kärger eigenvalue solver given per-measurement decay exponents.

    Parameters
    ----------
    t_d : jnp.array, shape (N,), effective diffusion time (Delta - delta/3)
    Ri  : jnp.array, shape (N,), b * Di_effective per measurement
    Re  : jnp.array, shape (N,), b * De_effective per measurement
    f   : scalar, intra-compartment volume fraction
    kappa : scalar, exchange rate in s^-1
    """
    kee = kappa * f / (1.0 - f)
    kt   = kappa * t_d
    keet = kee   * t_d

    Tr     = Ri + Re + kt + keet
    Det    = Ri * Re + Ri * keet + Re * kt
    disc_sq = jnp.maximum(Tr ** 2 - 4.0 * Det, 0.0)
    disc    = jnp.sqrt(disc_sq)

    lam_plus  = (Tr + disc) / 2.0
    lam_minus = (Tr - disc) / 2.0

    EPS = 1e-10
    safe_disc = jnp.where(disc > EPS, disc, jnp.ones_like(disc))
    safe_keet = jnp.where(keet > EPS, keet, jnp.ones_like(keet))

    sigma_plus_full = (
        f * (Ri - lam_minus) * (lam_minus - Re) / (safe_keet * safe_disc)
    )
    degenerate = (disc <= EPS) | (keet <= EPS)
    sigma_plus  = jnp.where(degenerate, f, sigma_plus_full)
    sigma_minus = 1.0 - sigma_plus

    return sigma_plus * jnp.exp(-lam_plus) + sigma_minus * jnp.exp(-lam_minus)


def karger_isotropic_signal(bvalues, delta, Delta, D1, D2, f, kappa):
    """Signal attenuation for isotropic two-compartment Kärger exchange.

    Both compartments are isotropic (G1Ball × G1Ball). No orientation needed.

    Parameters
    ----------
    bvalues : jnp.array, shape (N,), s/m^2
    delta   : jnp.array, shape (N,), pulse length in s
    Delta   : jnp.array, shape (N,), pulse separation in s
    D1      : scalar, intra-compartment diffusivity in m^2/s
    D2      : scalar, extra-compartment diffusivity in m^2/s
    f       : scalar, intra-compartment volume fraction
    kappa   : scalar, exchange rate in s^-1
    """
    t_d = Delta - delta / 3.0
    Ri  = bvalues * D1
    Re  = bvalues * D2
    return karger_from_Ri_Re(t_d, Ri, Re, f, kappa)


def nexi_signal(bvalues, delta, Delta, gradient_directions,
                mu_cart, Di, De_par, f, kappa):
    """Signal attenuation for the NEXI model (tortuosity constraint).

    Enforces De_perp = (1 - f) * De_par internally.

    Parameters
    ----------
    bvalues : jnp.array, shape (N,), s/m^2
    delta : jnp.array, shape (N,), pulse length in s
    Delta : jnp.array, shape (N,), pulse separation in s
    gradient_directions : jnp.array, shape (N, 3)
    mu_cart : jnp.array, shape (3,), Cartesian unit vector
    Di : scalar, intra-neurite parallel diffusivity in m^2/s
    De_par : scalar, extra-neurite parallel diffusivity in m^2/s
    f : scalar, intra-neurite volume fraction
    kappa : scalar, exchange rate in s^-1

    Returns
    -------
    jnp.array, shape (N,)
    """
    De_perp = (1.0 - f) * De_par
    return karger_signal(bvalues, delta, Delta, gradient_directions,
                         mu_cart, Di, De_par, De_perp, f, kappa)


# ---------------------------------------------------------------------------
# JAX matrix-exponential Kärger propagators (eq:karger_se_finite, eq:karger_ste_finite)
# jax.scipy.linalg.expm is available in JAX >= 0.4.1.
# ---------------------------------------------------------------------------

def _build_K_jax(kappa, f):
    """2×2 exchange-rate matrix (JAX)."""
    kee = kappa * f / (1.0 - f)
    return jnp.array([[-kappa, kee],
                      [kappa, -kee]])


def _karger_matrix_se_jax(D1, D2, T2_1, T2_2, T1_1, T1_2, kappa, f,
                           b, dt1, dt2, tau_exc, tau_180):
    """JAX finite-RF SE Kärger propagator for a single measurement.

    Parameters
    ----------
    D1, D2 : scalar — effective diffusion coefficients (m²/s)
    T2_1, T2_2, T1_1, T1_2 : scalar — relaxation times (s)
    kappa : scalar — exchange rate (s⁻¹)
    f : scalar — intra volume fraction
    b : scalar — total b-value (s/m²); split equally B1=B2=b/2
    dt1, dt2 : scalar — free-precession interval durations (s)
    tau_exc, tau_180 : scalar — RF pulse durations (s)

    Returns
    -------
    E : scalar — signal attenuation = sum(M_TE) / sum(M0)
    """
    K   = _build_K_jax(kappa, f)
    RT2 = jnp.diag(jnp.array([1.0 / T2_1, 1.0 / T2_2]))
    RT1 = jnp.diag(jnp.array([1.0 / T1_1, 1.0 / T1_2]))
    RD  = jnp.diag(jnp.array([D1, D2]))
    R12 = (2.0 / jnp.pi) * (RT2 + RT1)
    M0  = jnp.array([f, 1.0 - f])
    B = b / 2.0

    P_exc  = jsl.expm((K - R12) * tau_exc)
    P_fp1  = jsl.expm((K - RT2) * dt1 - B * RD)
    P_180  = jsl.expm((K - R12) * tau_180)
    P_fp2  = jsl.expm((K - RT2) * dt2 - B * RD)

    M_TE = P_fp2 @ P_180 @ P_fp1 @ P_exc @ M0
    return jnp.sum(M_TE)


def _karger_matrix_ste_jax(D1, D2, T2_1, T2_2, T1_1, T1_2, kappa, f,
                            b, delta, TM, tau_90, dt6):
    """JAX finite-RF STE Kärger propagator for a single measurement.

    Parameters
    ----------
    D1, D2 : scalar — effective diffusion coefficients (m²/s)
    T2_1, T2_2, T1_1, T1_2 : scalar — relaxation times (s)
    kappa : scalar — exchange rate (s⁻¹)
    f : scalar — intra volume fraction
    b : scalar — total b-value (s/m²); split equally B1=B2=b/2
    delta : scalar — encoding duration (s)
    TM : scalar — mixing time (s)
    tau_90 : scalar — 90° RF pulse duration (s)
    dt6 : scalar — second encoding + echo tail duration (s)

    Returns
    -------
    E : scalar — signal attenuation = sum(M_TE) / sum(M0); includes 0.5 factor
    """
    K   = _build_K_jax(kappa, f)
    RT2 = jnp.diag(jnp.array([1.0 / T2_1, 1.0 / T2_2]))
    RT1 = jnp.diag(jnp.array([1.0 / T1_1, 1.0 / T1_2]))
    RD  = jnp.diag(jnp.array([D1, D2]))
    R12 = (2.0 / jnp.pi) * (RT2 + RT1)
    M0  = jnp.array([f, 1.0 - f])
    B = b / 2.0

    P_exc   = jsl.expm((K - R12) * tau_90)
    P_enc1  = jsl.expm((K - RT2) * delta - B * RD)
    P_store = jsl.expm((K - R12) * tau_90)
    P_mix   = jsl.expm((K - RT1) * TM)
    P_rec   = jsl.expm((K - R12) * tau_90)
    P_enc2  = jsl.expm((K - RT2) * dt6 - B * RD)

    M_TE = 0.5 * P_enc2 @ P_rec @ P_mix @ P_store @ P_enc1 @ P_exc @ M0
    return jnp.sum(M_TE)
