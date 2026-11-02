"""The orientation-composed JAX forward functions of the cylinder models.

A cylinder's signal is the free diffusion along its axis times the restricted signal across it, and the gradient
across the axis is the part of the gradient direction perpendicular to it. The forms of the two factors are
:mod:`dmipy_analytic.jax.gaussian` and :mod:`dmipy_analytic.jax.restricted.cylinder`; the functions here are the
projection of a gradient onto the plane across the axis and the product, for a Cartesian axis (callers run
``unitsphere2cart_1d_jax`` first). Each is a side-effect-free function that can be wrapped with ``jax.jit``.
"""

import jax.numpy as jnp
from dmipy_analytic.jax import gaussian as jgaussian
from dmipy_analytic.jax.restricted import cylinder as jcylinder


def _perpendicular_magnitude(gradient_directions, mu_cart):
    """The length of the part of each gradient direction perpendicular to the axis ``mu_cart``: ``(N,)``."""
    mu_outer = jnp.outer(mu_cart, mu_cart)
    perp_plane = jnp.eye(3) - mu_outer
    proj = jnp.dot(perp_plane, gradient_directions.T)       # (3, N)
    return jnp.sqrt(jnp.sum(proj ** 2, axis=0))             # (N,)


def c2cylinder_signal(bvalues, gradient_directions, qvalues,
                      mu_cart, lambda_par, diameter):
    """Signal attenuation of the Soderman cylinder (short-pulse, long-time limit) with free diffusion along the axis.

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
    E_parallel = jgaussian.stick(bvalues, gradient_directions, mu_cart, lambda_par)
    q_perp = qvalues * _perpendicular_magnitude(gradient_directions, mu_cart)
    return E_parallel * jcylinder.sgp_long_time(q_perp, diameter)


def c4cylinder_signal(bvalues, gradient_directions, gradient_strengths,
                      delta, Delta,
                      mu_cart, lambda_par, diameter,
                      diffusion_perpendicular, gyromagnetic_ratio):
    """Signal attenuation of the Van Gelderen Gaussian-phase cylinder with free diffusion along the axis.

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

    Returns
    -------
    jnp.array, shape (N,)
    """
    E_parallel = jgaussian.stick(bvalues, gradient_directions, mu_cart, lambda_par)
    g_perp = gradient_strengths * _perpendicular_magnitude(gradient_directions, mu_cart)
    return E_parallel * jcylinder.gaussian_phase(g_perp, delta, Delta, diameter, diffusion_perpendicular,
                                                 gamma=gyromagnetic_ratio)


def build_c3cylinder_jax_fn(n_roots, n_functions, diffusion_perpendicular):
    """Factory: returns a JAX function for the Callaghan cylinder with free diffusion along the axis.

    Parameters
    ----------
    n_roots, n_functions : int
        The radial and angular modes of the series (static).
    diffusion_perpendicular : float

    Returns
    -------
    fn : callable (bvalues, gradient_directions, qvalues, tau,
                   mu_cart, lambda_par, diameter) -> jnp.array (N,)
    """
    D = float(diffusion_perpendicular)

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
        E_parallel = jgaussian.stick(bvalues, gradient_directions, mu_cart, lambda_par)
        q_perp = qvalues * _perpendicular_magnitude(gradient_directions, mu_cart)
        return E_parallel * jcylinder.sgp(q_perp, tau, diameter, D, n_roots, n_functions)

    return c3cylinder_signal
