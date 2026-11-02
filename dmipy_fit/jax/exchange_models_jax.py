"""JAX forward functions of the two-compartment Karger exchange models for Gaussian compartments.

The Karger closed form, the exchange generator and the multi-interval propagators are
:mod:`dmipy_analytic.jax.exchange`; the functions here form the dephasing exponents of a Stick, a Zeppelin or a Ball
under a pulsed gradient and pass them to the closed form. All are pure, side-effect-free and JIT-compilable, with
no Python-level branching on array values.
"""

import jax.numpy as jnp
from dmipy_analytic.jax import exchange as jexchange


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
    t_d = jexchange.diffusion_time(delta, Delta)          # (N,)

    cos_theta = jnp.sum(gradient_directions * mu_cart, axis=-1)   # (N,)
    cos2 = cos_theta ** 2
    sin2 = 1.0 - cos2

    Di_n = Di * cos2                          # (N,)
    De_n = De_par * cos2 + De_perp * sin2    # (N,)

    Ri = bvalues * Di_n       # (N,)
    Re = bvalues * De_n       # (N,)

    return jexchange.karger(Ri, Re, kappa, f, t_d)


def karger_isotropic_signal(bvalues, delta, Delta, D1, D2, f, kappa):
    """Signal attenuation for isotropic two-compartment Karger exchange.

    Both compartments are isotropic (G1Ball x G1Ball). No orientation needed.

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
    t_d = jexchange.diffusion_time(delta, Delta)
    Ri = bvalues * D1
    Re = bvalues * D2
    return jexchange.karger(Ri, Re, kappa, f, t_d)
