"""JAX forward functions of the two-compartment Karger exchange models for Gaussian compartments.

The Karger closed form, the exchange generator and the multi-interval propagators are
:mod:`dmipy_analytic.jax.exchange`; the functions here form the dephasing exponents of a Stick, a Zeppelin or a Ball
under a pulsed gradient and pass them to the closed form. All are pure, side-effect-free and JIT-compilable, with
no Python-level branching on array values.
"""

import jax.numpy as jnp
from dmipy_analytic.jax import exchange as jexchange


def nexi_spherical_mean(bvalues, delta, Delta, di, de, f, t_ex,
                        cos_theta, weights):
    """Published NEXI twin: exchange each Stick+Ball kernel, then average.

    bvalues are s/m², diffusivities m²/s and t_ex seconds. The quadrature
    nodes and weights come from X1NEXI. The PGSE effective time and the
    narrow-pulse kernel match NumPy; Gaussian exponents never need a signal floor.
    """
    ri = bvalues[:, None] * di * cos_theta**2
    re = bvalues[:, None] * de
    time = jexchange.diffusion_time(delta, Delta)[:, None]
    kernels = jexchange.karger(ri, re, (1. - f) / t_ex, f, time)
    return kernels @ weights
