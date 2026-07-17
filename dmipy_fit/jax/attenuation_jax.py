r"""JAX forwards for the coherence-pathway attenuation factors.

Each builder takes an :class:`~dmipy_fit.signal_models.attenuation.AttenuationFactor`
instance and returns a pure-JAX function ``factor_fn(scheme_jax, mu_cart, params)``
returning the multiplicative attenuation (a per-measurement array or the scalar
``1.0`` when inactive), mirroring the numpy ``AttenuationFactor.factor`` methods.

These let :class:`OccupancyGatedModel` compartments fit through ``solver='jax'``
(GPU): the gated compartment's signal is ``base_jax_fn x prod(factor_fn)``. Each
factor (transverse relaxation, surface relaxivity) is a b-independent per-compartment
weight; ``mu_cart`` is passed so a factor may couple to orientation, applied per
micro-orientation inside a dispersed bundle automatically.

Requires ``scheme_jax`` to carry ``tau_perp`` -- added by
:func:`dmipy_fit.jax.jax_compat.scheme_to_jax`.
"""
import jax.numpy as jnp

from ..signal_models.attenuation import (
    TransverseRelaxation, LongitudinalRelaxation,
    SurfaceRelaxivity, ExteriorSurfaceRelaxivity, IntraPoreSurfaceRelaxivity)

__all__ = ['build_jax_factor', 'JAX_FACTOR_BUILDERS']


def _build_transverse_relaxation(factor):
    def fn(scheme_jax, mu_cart, params):
        T2 = params.get('T2')
        tau_perp = scheme_jax.get('tau_perp')
        if T2 is None or tau_perp is None:
            return 1.0
        return jnp.where(jnp.isfinite(T2), jnp.exp(-tau_perp / T2), 1.0)
    return fn


def _build_longitudinal_relaxation(factor):
    def fn(scheme_jax, mu_cart, params):
        T1 = params.get('T1')
        tau_par = scheme_jax.get('tau_par')
        if T1 is None or tau_par is None:
            return 1.0
        return jnp.where(jnp.isfinite(T1), jnp.exp(-tau_par / T1), 1.0)
    return fn


def _build_exterior_surface_relaxivity(factor):
    S_over_V = float(factor.S_ext_over_V)

    def fn(scheme_jax, mu_cart, params):
        rho = params.get('surface_relaxivity')
        tau_perp = scheme_jax.get('tau_perp')
        if rho is None or tau_perp is None:
            return 1.0
        return jnp.where(jnp.isfinite(rho),
                         jnp.exp(-rho * S_over_V * tau_perp), 1.0)
    return fn


def _build_surface_relaxivity(factor):
    sv_fixed = factor.surface_to_volume

    def fn(scheme_jax, mu_cart, params):
        rho = params.get('surface_relaxivity')
        tau_perp = scheme_jax.get('tau_perp')
        if rho is None or tau_perp is None:
            return 1.0
        if sv_fixed is not None:
            sv = float(sv_fixed)
        else:
            d = params.get('diameter')
            if d is None:
                return 1.0
            sv = 4.0 / d
        return jnp.where(jnp.isfinite(rho), jnp.exp(-rho * sv * tau_perp), 1.0)
    return fn


def _build_intrapore_surface_relaxivity(factor, n_nodes=64):
    r"""Gamma-averaged intra-pore surface attenuation B = E_d[exp(-4 rho tau/d)]
    for a lumen diameter d ~ Gamma(shape a, rate beta).

    The numpy form (white_matter.surface.b_hat_ia) is the Bessel-K closed form
    2 (beta c)^{a/2}/Gamma(a) K_a(2 sqrt(beta c)), c = 4 rho tau. JAX has no
    Bessel-K, so we evaluate the equivalent integral by generalized
    Gauss-Laguerre quadrature (weight x^{a-1} e^{-x}, sum w_k = Gamma(a)):

        B = (1/Gamma(a)) sum_k w_k exp(-c beta / x_k).

    The shape ``a`` is an instance constant, so the nodes/weights are
    precomputed in numpy once and baked into the JAX closure; only ``c`` (via
    the fitted surface_relaxivity) and ``beta`` (via g_ratio) are traced.
    """
    from scipy.special import roots_genlaguerre, gamma as _gamma
    a_eff = factor.gamma_shape + (2.0 if factor.volume_weighted else 0.0)
    _x, _w = roots_genlaguerre(n_nodes, a_eff - 1.0)
    x = jnp.asarray(_x)                  # (n_nodes,)
    w = jnp.asarray(_w)
    inv_gamma_a = 1.0 / float(_gamma(a_eff))
    outer_scale = factor.gamma_scale_outer_diameter

    def fn(scheme_jax, mu_cart, params):
        rho = params.get('surface_relaxivity')
        tau_perp = scheme_jax.get('tau_perp')
        if rho is None or tau_perp is None:
            return 1.0
        g = params.get('g_ratio')
        g = 1.0 if g is None else g                 # inner = g * outer
        beta = 1.0 / (g * outer_scale)              # rate (1/m)
        c = jnp.asarray(4.0 * rho * tau_perp)        # scalar or (N,), m
        # quadrature over the node axis (leading), broadcasting against c
        xr = x.reshape((-1,) + (1,) * c.ndim)        # (n_nodes, 1...)
        wr = w.reshape((-1,) + (1,) * c.ndim)
        arg = -(c[None, ...] * beta) / xr            # (n_nodes, *c.shape)
        B = (wr * jnp.exp(arg)).sum(0) * inv_gamma_a
        B = jnp.where(c <= 0, 1.0, B)
        return jnp.where(jnp.isfinite(rho), B, 1.0)
    return fn


JAX_FACTOR_BUILDERS = {
    TransverseRelaxation: _build_transverse_relaxation,
    LongitudinalRelaxation: _build_longitudinal_relaxation,
    ExteriorSurfaceRelaxivity: _build_exterior_surface_relaxivity,
    SurfaceRelaxivity: _build_surface_relaxivity,
    IntraPoreSurfaceRelaxivity: _build_intrapore_surface_relaxivity,
}


def build_jax_factor(factor):
    """Return a JAX ``factor_fn(scheme_jax, mu_cart, params)`` for ``factor``."""
    builder = JAX_FACTOR_BUILDERS.get(type(factor))
    if builder is None:
        raise NotImplementedError(
            "No JAX forward for attenuation factor {}".format(
                type(factor).__name__))
    return builder(factor)
