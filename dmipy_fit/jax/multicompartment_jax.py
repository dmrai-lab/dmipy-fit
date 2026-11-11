"""JAX-compiled multi-compartment forward model factory.

Usage
-----
    forward_fn = build_mc_forward_fn(model, acquisition_scheme)
    signal = forward_fn(params_scaled)   # params_scaled in SI units

The factory runs at Python level once per (model, scheme) pair and returns a
jax.jit-compiled function with no Python overhead at evaluation time.
"""

import numpy as np
import jax
import jax.numpy as jnp

from dmipy_analytic.jax import exchange as jexchange
from dmipy_analytic.jax import gaussian as jgaussian
from dmipy_analytic.jax import relaxation as jrelaxation
from dmipy_analytic.jax.restricted import plane as jplane
from dmipy_analytic.jax.restricted import sphere as jsphere

from .jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
from .signal_models_jax import (
    c2cylinder_signal,
    c4cylinder_signal,
    build_c3cylinder_jax_fn,
)
from .exchange_models_jax import nexi_spherical_mean
from ..signal_models.gaussian_models import G1Ball, G2Zeppelin, G3TemporalZeppelin
from ..signal_models.exchange_models import X0GeneralizedKarger, X1NEXI
# Backward-compat alias: X1KargerModel was renamed to X0GeneralizedKarger
X1KargerModel = X0GeneralizedKarger
from ..signal_models.cylinder_models import (
    C1Stick,
    C2CylinderStejskalTannerApproximation,
    C3CylinderCallaghanApproximation,
    C4CylinderGaussianPhaseApproximation,
)
from ..signal_models.sphere_models import (
    S1Dot,
    S2SphereStejskalTannerApproximation,
    S4SphereGaussianPhaseApproximation,
)
from ..signal_models.plane_models import (
    P2PlaneStejskalTannerApproximation,
    P3PlaneCallaghanApproximation,
)
from ..distributions.distribute_models import (
    SD1WatsonDistributed,
    SD2BinghamDistributed,
    DD1GammaDistributed,
    DD2PoissonDistributed,
)
from .distributed_models_jax import (
    build_watson_distributed_jax_fn,
    build_bingham_distributed_jax_fn,
    build_watson_sh_jax_fn,
    build_bingham_sh_jax_fn,
    _all_inner_models_have_rh,
)
from ..signal_models.attenuation import OccupancyGatedModel
from .attenuation_jax import build_jax_factor


# ---------------------------------------------------------------------------
# Dispatch table: model class → JAX evaluation function
#
# Each entry is a callable:
#   fn(scheme_jax, params_scaled_slice) -> jnp.array signal attenuation
#
# The params_scaled_slice dict contains the model's own parameters already
# in SI units (orientation as 2-element array [theta, phi]).
# ---------------------------------------------------------------------------

def _apply_t2_weighting(E, scheme_jax, params):
    """Apply exp(-TE/T2) if T2 is present and finite in params."""
    T2 = params.get('T2')
    TE = scheme_jax.get('TE')
    if T2 is not None and TE is not None:
        t2_factor = jnp.where(jnp.isfinite(T2), jrelaxation.transverse_decay(TE, T2), 1.0)
        return E * t2_factor
    return E


def _g1ball_jax_fn(scheme_jax, params):
    E = jgaussian.ball(scheme_jax['bvalues'], params['lambda_iso'])
    return _apply_t2_weighting(E, scheme_jax, params)


def _c1stick_jax_fn(scheme_jax, params):
    mu_cart = unitsphere2cart_1d_jax(params['mu'])
    E = jgaussian.stick(
        scheme_jax['bvalues'],
        scheme_jax['gradient_directions'],
        mu_cart,
        params['lambda_par'],
    )
    return _apply_t2_weighting(E, scheme_jax, params)


def _g2zeppelin_jax_fn(scheme_jax, params):
    mu_cart = unitsphere2cart_1d_jax(params['mu'])
    E = jgaussian.zeppelin(
        scheme_jax['bvalues'],
        scheme_jax['gradient_directions'],
        mu_cart,
        params['lambda_par'],
        params['lambda_perp'],
    )
    return _apply_t2_weighting(E, scheme_jax, params)


def _g3temporal_zeppelin_jax_fn(scheme_jax, params):
    """G3TemporalZeppelin: Zeppelin with D_perp(δ,Δ) = λ_inf + A*(ln(Δ/δ)+3/2)/(Δ-δ/3).

    Requires 'delta' and 'Delta' in scheme_jax (PGSE timing).
    Signal: E = exp(-b * (D_perp + (λ_par - D_perp) * (n·μ)²))
    """
    mu_cart = unitsphere2cart_1d_jax(params['mu'])
    E = jgaussian.temporal_zeppelin(
        scheme_jax['bvalues'], scheme_jax['gradient_directions'], mu_cart,
        params['lambda_par'], params['lambda_inf'], params['A'],
        scheme_jax['delta'], scheme_jax['Delta'])
    return _apply_t2_weighting(E, scheme_jax, params)


def _s2sphere_jax_fn(scheme_jax, params):
    E = jsphere.sgp_long_time(scheme_jax['qvalues'], params['diameter'])
    return _apply_t2_weighting(E, scheme_jax, params)


def _c2cylinder_jax_fn(scheme_jax, params):
    mu_cart = unitsphere2cart_1d_jax(params['mu'])
    E = c2cylinder_signal(
        scheme_jax['bvalues'],
        scheme_jax['gradient_directions'],
        scheme_jax['qvalues'],
        mu_cart,
        params['lambda_par'],
        params['diameter'],
    )
    return _apply_t2_weighting(E, scheme_jax, params)


def _make_c4cylinder_jax_fn(model_obj, acquisition_scheme=None):
    """Factory: creates a C4 dispatch fn closed over this instance's constants."""
    D = float(model_obj.diffusion_perpendicular)
    gamma = float(model_obj.gyromagnetic_ratio)

    def _c4cylinder_jax_fn(scheme_jax, params):
        mu_cart = unitsphere2cart_1d_jax(params['mu'])
        E = c4cylinder_signal(
            scheme_jax['bvalues'],
            scheme_jax['gradient_directions'],
            scheme_jax['gradient_strengths'],
            scheme_jax['delta'],
            scheme_jax['Delta'],
            mu_cart,
            params['lambda_par'],
            params['diameter'],
            D,
            gamma,
        )
        return _apply_t2_weighting(E, scheme_jax, params)
    return _c4cylinder_jax_fn


def _make_s4sphere_ogse_jax_fn(model_obj, acquisition_scheme=None):
    """Factory: creates an S4 dispatch fn for JAX GPA sphere signal.

    Dispatch logic (chosen once at factory time):
      - If the scheme has a stored G(t) waveform (_G is not None): use the
        numerical Stepisnik / causal-IIR path (OGSE-compatible).
      - Otherwise: use the closed-form PGSE formula (Balinov/Stepisnik) from
        gradient_strengths + delta + Delta.  This is the common path for
        schemes built with acquisition_scheme_from_bvalues().

    Parameters
    ----------
    model_obj : S4SphereGaussianPhaseApproximation instance
    acquisition_scheme : AcquisitionScheme (optional waveform)

    Returns
    -------
    fn : callable (scheme_jax, params) -> jnp.array (n_m,)
    """
    D     = float(model_obj.diffusion_constant)
    gamma = float(model_obj.gyromagnetic_ratio)

    # Decide path at factory time (Python-level, not inside JIT)
    has_waveform = (acquisition_scheme is not None
                    and hasattr(acquisition_scheme, '_G')
                    and acquisition_scheme._G is not None)

    if has_waveform:
        # Numerical waveform path (OGSE or any G(t))
        def _s4sphere_ogse_jax_fn(scheme_jax, params):
            diameter   = params['diameter']
            G_waveform = scheme_jax['G_waveform']  # (n_m, n_t, 3)
            dt         = scheme_jax['dt']

            def _single_measurement(G_m):
                return jsphere.gaussian_phase_waveform(
                    G_m, dt, diameter, D, gamma=gamma)

            E = jax.vmap(_single_measurement)(G_waveform)  # (n_m,)
            return _apply_t2_weighting(E, scheme_jax, params)

        return _s4sphere_ogse_jax_fn

    else:
        # Closed-form PGSE path (gradient_strengths + delta + Delta)
        def _s4sphere_pgse_jax_fn(scheme_jax, params):
            diameter          = params['diameter']
            gradient_strengths = scheme_jax['gradient_strengths']  # (n_m,)
            delta              = scheme_jax['delta']                # (n_m,)
            Delta              = scheme_jax['Delta']                # (n_m,)

            def _single_meas(g, d, D_big):
                return jsphere.gaussian_phase(
                    g, d, D_big, diameter, D, gamma=gamma)

            E = jax.vmap(_single_meas)(gradient_strengths, delta, Delta)
            return _apply_t2_weighting(E, scheme_jax, params)

        return _s4sphere_pgse_jax_fn


def _make_c3cylinder_jax_fn(model_obj, acquisition_scheme=None):
    """Factory: creates a C3 dispatch fn closed over this instance's alpha table."""
    inner_fn = build_c3cylinder_jax_fn(
        model_obj.number_of_roots,
        model_obj.number_of_functions,
        model_obj.diffusion_perpendicular,
    )

    def _c3cylinder_jax_fn(scheme_jax, params):
        mu_cart = unitsphere2cart_1d_jax(params['mu'])
        E = inner_fn(
            scheme_jax['bvalues'],
            scheme_jax['gradient_directions'],
            scheme_jax['qvalues'],
            scheme_jax['tau'],
            mu_cart,
            params['lambda_par'],
            params['diameter'],
        )
        return _apply_t2_weighting(E, scheme_jax, params)
    return _c3cylinder_jax_fn


def _make_watson_jax_fn(model_obj, acquisition_scheme=None):
    """Factory for SD1WatsonDistributed.

    Uses SH convolution when acquisition_scheme is provided and all inner
    models have JAX RH functions (G1Ball, C1Stick, G2Zeppelin).
    Falls back to numerical hemisphere integration otherwise.
    """
    if acquisition_scheme is not None and _all_inner_models_have_rh(model_obj):
        return build_watson_sh_jax_fn(model_obj, acquisition_scheme)

    # Numerical fallback
    inner_jax_fns_dict = {}
    for inner_model in model_obj.models:
        model_type = type(inner_model)
        if model_type in _JAX_MODEL_FACTORIES:
            inner_jax_fns_dict[inner_model] = _JAX_MODEL_FACTORIES[model_type](inner_model)
        elif model_type in _JAX_MODEL_FNS:
            inner_jax_fns_dict[inner_model] = _JAX_MODEL_FNS[model_type]
        else:
            raise NotImplementedError(
                "No JAX implementation for inner model {} inside "
                "SD1WatsonDistributed.".format(model_type.__name__)
            )
    return build_watson_distributed_jax_fn(model_obj, inner_jax_fns_dict)


def _make_bingham_jax_fn(model_obj, acquisition_scheme=None):
    """Factory for SD2BinghamDistributed.

    Uses SH convolution when acquisition_scheme is provided and all inner
    models have JAX RH functions.  Falls back to numerical integration.
    """
    if acquisition_scheme is not None and _all_inner_models_have_rh(model_obj):
        return build_bingham_sh_jax_fn(model_obj, acquisition_scheme)

    inner_jax_fns_dict = {}
    for inner_model in model_obj.models:
        model_type = type(inner_model)
        if model_type in _JAX_MODEL_FACTORIES:
            inner_jax_fns_dict[inner_model] = _JAX_MODEL_FACTORIES[model_type](inner_model)
        elif model_type in _JAX_MODEL_FNS:
            inner_jax_fns_dict[inner_model] = _JAX_MODEL_FNS[model_type]
        else:
            raise NotImplementedError(
                "No JAX implementation for inner model {} inside "
                "SD2BinghamDistributed.".format(model_type.__name__)
            )
    return build_bingham_distributed_jax_fn(model_obj, inner_jax_fns_dict)


def _s1dot_jax_fn(scheme_jax, params):
    """S1Dot: stationary (non-diffusing) compartment, E = 1."""
    E = jnp.ones_like(scheme_jax['bvalues'])
    return _apply_t2_weighting(E, scheme_jax, params)


def _p2plane_jax_fn(scheme_jax, params):
    """P2 plane (Balinov SGP): 2(1-cos(2 pi q d)) / (2 pi q d)^2, 1 at q=0."""
    E = jplane.sgp_long_time(scheme_jax['qvalues'], params['diameter'])
    return _apply_t2_weighting(E, scheme_jax, params)


def _make_p3plane_jax_fn(model_obj, acquisition_scheme=None):
    """P3 plane (finite-time Callaghan): root sum over xi (cos) and zeta (sin)."""
    D = float(model_obj.Dintra)
    n_roots = model_obj.number_of_roots

    def fn(scheme_jax, params):
        E = jplane.sgp(scheme_jax['qvalues'], scheme_jax['tau'], params['diameter'], D, n_roots)
        return _apply_t2_weighting(E, scheme_jax, params)

    return fn


def _make_occupancy_gated_jax_fn(model_obj, acquisition_scheme=None):
    """Factory for OccupancyGatedModel: base diffusion JAX fn x factor product.

    The gated compartment's signal is the base model's JAX forward multiplied by
    each occupancy-gated factor (transverse relaxation, surface relaxivity).
    Factor-owned parameters (e.g. T2) are withheld from the base call so the base
    does not double-apply them. Factors receive ``mu`` so an orientation-coupled
    factor is applied per micro-orientation inside a dispersed bundle.
    """
    base = model_obj.model
    base_type = type(base)
    if base_type in _JAX_MODEL_FNS:
        base_fn = _JAX_MODEL_FNS[base_type]
    elif base_type in _JAX_MODEL_FACTORIES:
        base_fn = _JAX_MODEL_FACTORIES[base_type](base, acquisition_scheme)
    else:
        raise NotImplementedError(
            "No JAX implementation for base model {} inside "
            "OccupancyGatedModel.".format(base_type.__name__))

    factor_fns = [build_jax_factor(f) for f in model_obj.factors]
    base_names = list(model_obj._base_names)   # base params NOT owned by factors

    def occupancy_gated_fn(scheme_jax, params):
        base_params = {k: params.get(k) for k in base_names}
        E = base_fn(scheme_jax, base_params)
        mu = params.get('mu', getattr(base, 'mu', None))
        mu_cart = unitsphere2cart_1d_jax(mu) if mu is not None else None
        for ffn in factor_fns:
            E = E * ffn(scheme_jax, mu_cart, params)
        return E

    return occupancy_gated_fn


def _make_dd1gamma_jax_fn(model_obj, acquisition_scheme=None):
    """Factory for DD1GammaDistributed(C3 or C4 Cylinder).

    Integrates the cylinder signal over a gamma distribution of diameters
    using a fixed N_QUAD-point quadrature grid.  Gamma PDF weights are
    computed dynamically from (alpha, beta) at each forward call.

    Dispatches on the inner cylinder type:
      C3CylinderCallaghanApproximation  — needs qvalues, tau
      C4CylinderGaussianPhaseApproximation — needs gradient_strengths, delta, Delta
    """
    # Detect inner cylinder type
    inner_c3 = next(
        (m for m in model_obj.models
         if isinstance(m, C3CylinderCallaghanApproximation)), None)
    inner_c4 = next(
        (m for m in model_obj.models
         if isinstance(m, C4CylinderGaussianPhaseApproximation)), None)

    if inner_c3 is None and inner_c4 is None:
        raise NotImplementedError(
            "_make_dd1gamma_jax_fn: no C3 or C4 cylinder found inside "
            "DD1GammaDistributed. models={}".format(
                [type(m).__name__ for m in model_obj.models]))

    # Static quadrature grid over cylinder diameter (compile-time constants)
    _N_QUAD = 64
    _D_MIN, _D_MAX = 0.2e-6, 25.0e-6        # 0.2 µm – 25 µm (diameter, metres)
    _quad_d = jnp.array(
        np.linspace(_D_MIN, _D_MAX, _N_QUAD), dtype=np.float64)

    if inner_c3 is not None:
        # ── C3 Callaghan path ──────────────────────────────────────────────
        _cyl_param_prefix = 'C3CylinderCallaghanApproximation_1_'
        c3_fn = build_c3cylinder_jax_fn(
            inner_c3.number_of_roots,
            inner_c3.number_of_functions,
            inner_c3.diffusion_perpendicular,
        )

        def dd1gamma_jax_fn(scheme_jax, params):
            alpha      = params['DD1Gamma_1_alpha']
            beta       = params['DD1Gamma_1_beta']
            mu_cart    = unitsphere2cart_1d_jax(
                params[_cyl_param_prefix + 'mu'])
            lambda_par = params[_cyl_param_prefix + 'lambda_par']

            _quad_r = _quad_d / 2.0
            log_w = (alpha - 1.0) * jnp.log(_quad_r) - _quad_r / beta
            log_w = log_w - jnp.max(log_w)
            w = jnp.exp(log_w)
            w = w / jnp.sum(w)

            def cyl_at_d(d):
                return c3_fn(
                    scheme_jax['bvalues'],
                    scheme_jax['gradient_directions'],
                    scheme_jax['qvalues'],
                    scheme_jax['tau'],
                    mu_cart, lambda_par, d,
                )

            signals = jax.vmap(cyl_at_d)(_quad_d)
            E = jnp.sum(signals * w[:, None], axis=0)
            T2 = params.get(_cyl_param_prefix + 'T2')
            TE = scheme_jax.get('TE')
            if T2 is not None and TE is not None:
                t2_factor = jnp.where(jnp.isfinite(T2), jrelaxation.transverse_decay(TE, T2), 1.0)
                E = E * t2_factor
            return E

    else:
        # ── C4 Gaussian Phase path ─────────────────────────────────────────
        _cyl_param_prefix = 'C4CylinderGaussianPhaseApproximation_1_'
        _D_perp = float(inner_c4.diffusion_perpendicular)
        _gamma  = float(inner_c4.gyromagnetic_ratio)

        def dd1gamma_jax_fn(scheme_jax, params):
            alpha      = params['DD1Gamma_1_alpha']
            beta       = params['DD1Gamma_1_beta']
            mu_cart    = unitsphere2cart_1d_jax(
                params[_cyl_param_prefix + 'mu'])
            lambda_par = params[_cyl_param_prefix + 'lambda_par']

            _quad_r = _quad_d / 2.0
            log_w = (alpha - 1.0) * jnp.log(_quad_r) - _quad_r / beta
            log_w = log_w - jnp.max(log_w)
            w = jnp.exp(log_w)
            w = w / jnp.sum(w)

            def cyl_at_d(d):
                return c4cylinder_signal(
                    scheme_jax['bvalues'],
                    scheme_jax['gradient_directions'],
                    scheme_jax['gradient_strengths'],
                    scheme_jax['delta'],
                    scheme_jax['Delta'],
                    mu_cart, lambda_par, d,
                    _D_perp, _gamma,
                )

            signals = jax.vmap(cyl_at_d)(_quad_d)
            E = jnp.sum(signals * w[:, None], axis=0)
            T2 = params.get(_cyl_param_prefix + 'T2')
            TE = scheme_jax.get('TE')
            if T2 is not None and TE is not None:
                t2_factor = jnp.where(jnp.isfinite(T2), jrelaxation.transverse_decay(TE, T2), 1.0)
                E = E * t2_factor
            return E

    return dd1gamma_jax_fn


def _make_dd2poisson_jax_fn(model_obj, acquisition_scheme=None):
    """Factory for DD2PoissonDistributed(C3CylinderCallaghanApproximation).

    DD2Poisson is Gamma(alpha=mu/BETA_SCALING, scale=BETA_SCALING) where
    BETA_SCALING = 1 µm.  Single free parameter: mu = mean diameter (metres).

    params keys expected (local names, SI units):
        'C3CylinderCallaghanApproximation_1_mu'        — [theta, phi] rad
        'C3CylinderCallaghanApproximation_1_lambda_par' — m²/s
        'DD2Poisson_1_mean_diameter'                   — mean diameter, metres
    """
    inner_c3 = next(m for m in model_obj.models
                    if isinstance(m, C3CylinderCallaghanApproximation))
    c3_fn = build_c3cylinder_jax_fn(
        inner_c3.number_of_roots,
        inner_c3.number_of_functions,
        inner_c3.diffusion_perpendicular,
    )

    _N_QUAD = 64
    _D_MIN, _D_MAX = 0.2e-6, 25.0e-6
    _quad_d = jnp.array(
        np.linspace(_D_MIN, _D_MAX, _N_QUAD), dtype=np.float64)
    _BETA = np.float64(1e-6)   # fixed scale = 1 µm (Poisson parameterization)

    def dd2poisson_jax_fn(scheme_jax, params):
        mu_diam    = params['DD2Poisson_1_mean_diameter']   # mean diameter (m)
        mu_cart    = unitsphere2cart_1d_jax(
            params['C3CylinderCallaghanApproximation_1_mu'])
        lambda_par = params[
            'C3CylinderCallaghanApproximation_1_lambda_par']

        # Gamma(alpha=mu/BETA, scale=BETA) PDF weights — over RADII r = d/2.
        # Analytical model: stats.gamma(alpha, scale=BETA) over radii,
        # diameter = 2*r.  Evaluate PDF at r = _quad_d / 2.
        alpha = mu_diam / _BETA
        _quad_r = _quad_d / 2.0
        log_w = (alpha - 1.0) * jnp.log(_quad_r) - _quad_r / _BETA
        log_w = log_w - jnp.max(log_w)
        w = jnp.exp(log_w)
        w = w / jnp.sum(w)

        def c3_at_d(d):
            return c3_fn(
                scheme_jax['bvalues'],
                scheme_jax['gradient_directions'],
                scheme_jax['qvalues'],
                scheme_jax['tau'],
                mu_cart, lambda_par, d,
            )

        signals = jax.vmap(c3_at_d)(_quad_d)  # (N_QUAD, N_meas)
        E = jnp.sum(signals * w[:, None], axis=0)
        # T2 is diameter-independent: apply AFTER the diameter integral.
        T2 = params.get('C3CylinderCallaghanApproximation_1_T2')
        TE = scheme_jax.get('TE')
        if T2 is not None and TE is not None:
            t2_factor = jnp.where(jnp.isfinite(T2), jrelaxation.transverse_decay(TE, T2), 1.0)
            E = E * t2_factor
        return E

    return dd2poisson_jax_fn


def checked_compartment_exponent(signal, name):
    """Real effective-medium -log(E), checked also under JIT on CPU/GPU.

    A runtime callback raises CompartmentSignalDomainError for underflowed,
    negative or nonfinite restricted signals. Gaussian pools use exponents
    directly, so valid high attenuation never enters this domain check.
    """
    from ..signal_models.exchange_models import CompartmentSignalDomainError
    def raise_domain():
        raise CompartmentSignalDomainError(
            f'{name}: effective-medium -log(E) requires finite E > 0')
    def fail(_):
        jax.debug.callback(raise_domain)
        return jnp.zeros_like(signal)
    return jax.lax.cond(jnp.all(jnp.isfinite(signal) & (signal > 0)),
                        lambda _: -jnp.log(signal), fail, operand=None)


def _resolve_karger_diffusion_fn(sub_model, karger_model, acquisition_scheme):
    """Diffusion-only JAX fn + relaxation keys for one Karger sub-model.

    Returns ``(diff_fn, T2_key, T1_key)`` where ``diff_fn(scheme_jax, params)``
    is the sub-model's diffusion-only signal. Mirrors NumPy ``_sub_kwargs``: an
    OccupancyGatedModel is unwrapped to its *base* diffusion model (so no
    relaxation factor is applied), and the base's diffusion params are mapped
    from the public Karger ``compartment_parameter_names`` record. The
    per-pool ``…_T2`` / ``…_T1`` combined keys are returned for the propagator
    (None when the pool has no such parameter).
    """
    from ..signal_models.attenuation import OccupancyGatedModel
    inv = karger_model.compartment_parameter_names
    if isinstance(sub_model, OccupancyGatedModel):
        base = sub_model.model
        diff_param_names = [p for p in base.parameter_ranges if p not in ('T2', 'T1')]
    else:
        base = sub_model
        diff_param_names = [p for p in sub_model.parameter_ranges
                            if p not in ('T2', 'T1')]
    base_type = type(base)
    if base_type in _JAX_MODEL_FNS:
        base_fn = _JAX_MODEL_FNS[base_type]
    elif base_type in _JAX_MODEL_FACTORIES:
        base_fn = _JAX_MODEL_FACTORIES[base_type](base, acquisition_scheme)
    else:
        raise NotImplementedError(
            "No JAX implementation for Karger sub-model base {}.".format(
                base_type.__name__))
    key_map = {p: inv[(sub_model, p)] for p in diff_param_names}
    T2_key = inv.get((sub_model, 'T2'))
    T1_key = inv.get((sub_model, 'T1'))

    def exponent_fn(scheme_jax, params):
        p = {name: params[key] for name, key in key_map.items()}
        if base_type is G1Ball:
            return scheme_jax['bvalues'] * p['lambda_iso']
        if base_type in (C1Stick, G2Zeppelin):
            axis = unitsphere2cart_1d_jax(p['mu'])
            B = scheme_jax['btensor']
            along = jnp.einsum('i,mij,j->m', axis, B, axis)
            if base_type is C1Stick:
                return p['lambda_par'] * along
            return (p['lambda_perp'] * jnp.trace(B, axis1=-2, axis2=-1)
                    + (p['lambda_par'] - p['lambda_perp']) * along)
        return checked_compartment_exponent(base_fn(scheme_jax, p), base_type.__name__)
    return exponent_fn, T2_key, T1_key


def _make_karger_waveform_jax_fn(model_obj, acquisition_scheme):
    """The JAX twin of the NumPy model's filter-exchange route (``exchange_models.waveform_signal``): each row of each
    filter-exchange sequence by :func:`dmipy_analytic.jax.exchange.karger_waveform` along the row's effective gradient,
    the pools' diffusivities along the row's direction from the sub-models' dephasing exponents, any per-pool T2 / T1
    from the namespace, the pathway amplitude left out."""
    intra_fn, intra_T2k, intra_T1k = _resolve_karger_diffusion_fn(model_obj.model_intra, model_obj, acquisition_scheme)
    extra_fn, extra_T2k, extra_T1k = _resolve_karger_diffusion_fn(model_obj.model_extra, model_obj, acquisition_scheme)
    from ..core.waveform import acquisition_waveform_blocks
    blocks = [(jnp.asarray(G), dt, stored, rows)
              for G, dt, stored, rows in acquisition_waveform_blocks(acquisition_scheme)]

    def _T(params, key):
        return jnp.inf if key is None else params[key]

    from ..signal_models.exchange_models import double_encodings
    is_double = bool(double_encodings(acquisition_scheme))
    has_relaxation = any(k is not None for k in (intra_T2k, extra_T2k, intra_T1k, extra_T1k))
    is_stimulated = getattr(acquisition_scheme, 'TM', None) is not None

    def fn(scheme_jax, params):
        params = model_obj.add_linked_parameters_to_parameters(params)
        b = scheme_jax['bvalues']
        f = params['f']
        K = jexchange.exchange_matrix(params['kappa'], f)
        M0 = jnp.stack([f, 1.0 - f])
        Ri = intra_fn(scheme_jax, params)
        Re = extra_fn(scheme_jax, params)
        D = jnp.stack([Ri, Re], axis=-1) / jnp.where(b > 0, b, 1.)[:, None]
        T2 = jnp.stack([_T(params, intra_T2k), _T(params, extra_T2k)])
        T1 = jnp.stack([_T(params, intra_T1k), _T(params, extra_T1k)])
        E = jnp.zeros(b.shape[0], dtype=D.dtype)
        for G, dt, stored, rows in blocks:
            def propagate(sub_steps):
                return jax.vmap(lambda g, d: jexchange.karger_waveform(
                    g, dt, d, K, M0, stored=stored, T2=T2, T1=T1, sub_steps=sub_steps))(G, D[rows])
            sig = jax.lax.cond(-jnp.trace(K)*dt > 1,
                               lambda _: propagate(8), lambda _: propagate(1), operand=None)
            E = E.at[rows].set(sig)
        if has_relaxation and not is_double:
            active_relaxation = jnp.any(jnp.isfinite(T1)) | jnp.any(jnp.isfinite(T2))
            if is_stimulated:
                E = jnp.where(active_relaxation, E*.5, E)
            E = jnp.where(active_relaxation & (b < 1e3), 1., E)
        return E
    return fn


def _make_x1karger_jax_fn(model_obj, acquisition_scheme=None):
    """Factory for X0GeneralizedKarger.

    PGSE and FEXI share the analytic held-waveform propagator, including
    coupled T1/T2. Finite-duration RF outside FEXI remains unsupported.
    """
    if acquisition_scheme is None:
        raise ValueError('X0GeneralizedKarger JAX requires an acquisition scheme for its waveform')
    from ..signal_models.attenuation import OccupancyGatedModel
    bases = [m.model if isinstance(m, OccupancyGatedModel) else m for m in model_obj.models]
    types = tuple(type(m) for m in bases)
    gated = any(isinstance(m, OccupancyGatedModel) for m in model_obj.models)
    if not gated and types not in ((G1Ball, G1Ball), (S4SphereGaussianPhaseApproximation, G1Ball),
                                   (C1Stick, G2Zeppelin)):
        raise NotImplementedError(f'No JAX route for X0GeneralizedKarger of {types[0].__name__} and {types[1].__name__}')
    for name in ('tau_exc', 'tau_180', 'tau_90'):
        duration = getattr(acquisition_scheme, name, None)
        if duration is not None and np.any(np.asarray(duration) > 0):
            from ..signal_models.exchange_models import double_encodings
            if not double_encodings(acquisition_scheme):
                raise NotImplementedError('JAX Karger requires instantaneous RF outside FEXI')
    return _make_karger_waveform_jax_fn(model_obj, acquisition_scheme)


# Simple models: no per-instance constants needed
_JAX_MODEL_FNS = {
    G1Ball: _g1ball_jax_fn,
    C1Stick: _c1stick_jax_fn,
    G2Zeppelin: _g2zeppelin_jax_fn,
    G3TemporalZeppelin: _g3temporal_zeppelin_jax_fn,
    S2SphereStejskalTannerApproximation: _s2sphere_jax_fn,
    C2CylinderStejskalTannerApproximation: _c2cylinder_jax_fn,
    S1Dot: _s1dot_jax_fn,
    P2PlaneStejskalTannerApproximation: _p2plane_jax_fn,
}

# Models that need a per-instance factory (have instance-level constants)
_JAX_MODEL_FACTORIES = {
    C3CylinderCallaghanApproximation: _make_c3cylinder_jax_fn,
    C4CylinderGaussianPhaseApproximation: _make_c4cylinder_jax_fn,
    S4SphereGaussianPhaseApproximation: _make_s4sphere_ogse_jax_fn,
    X1KargerModel: _make_x1karger_jax_fn,
    SD1WatsonDistributed: _make_watson_jax_fn,
    SD2BinghamDistributed: _make_bingham_jax_fn,
    DD1GammaDistributed: _make_dd1gamma_jax_fn,
    DD2PoissonDistributed: _make_dd2poisson_jax_fn,
    OccupancyGatedModel: _make_occupancy_gated_jax_fn,
    P3PlaneCallaghanApproximation: _make_p3plane_jax_fn,
    # SD1WatsonDistributed / SD2BinghamDistributed (mapped above) auto-detect a
    # gated EA compartment: its inner model lacks a JAX RH fn, so the watson/
    # bingham factories fall to the numerical per-orientation path that applies
    # the occupancy-gated factors per micro-orientation.
}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Spherical-mean dispatch (for MultiCompartmentSphericalMeanModel)
# ---------------------------------------------------------------------------

def _g1ball_sm_fn(shell_bvals_jax, params):
    """G1Ball spherical mean: same as signal (isotropic)."""
    return jgaussian.ball(shell_bvals_jax, params['lambda_iso'])


def _c1stick_sm_fn(shell_bvals_jax, params):
    """C1Stick spherical mean (analytical)."""
    return jgaussian.stick_powder_average(shell_bvals_jax, params['lambda_par'])


def _g2zeppelin_sm_fn(shell_bvals_jax, params):
    """G2Zeppelin spherical mean (analytical)."""
    return jgaussian.zeppelin_powder_average(
        shell_bvals_jax, params['lambda_par'], params['lambda_perp'])


# Spherical mean models indexed by class; no orientation params needed.
_JAX_SM_MODEL_FNS = {
    G1Ball:     _g1ball_sm_fn,
    C1Stick:    _c1stick_sm_fn,
    G2Zeppelin: _g2zeppelin_sm_fn,
}


def _make_nexi_sm_jax_fn(model, acquisition_scheme):
    if acquisition_scheme.shell_delta is None or acquisition_scheme.shell_Delta is None:
        raise ValueError('NEXI requires PGSE delta and Delta')
    mask = ~acquisition_scheme.shell_b0_mask
    delta = jnp.asarray(acquisition_scheme.shell_delta[mask])
    big_delta = jnp.asarray(acquisition_scheme.shell_Delta[mask])
    nodes = jnp.asarray(model.cos_theta)
    weights = jnp.asarray(model.quadrature_weights)

    def forward(bvalues, params):
        return nexi_spherical_mean(bvalues, delta, big_delta,
                                   params['D_i'], params['D_e'], params['f'],
                                   params['t_ex'], nodes, weights)

    return forward


_JAX_SM_MODEL_FACTORIES = {X1NEXI: _make_nexi_sm_jax_fn}


def build_mc_sm_forward_fn(model, acquisition_scheme, broadcast=True):
    """Build a jax.jit-compiled forward function for
    MultiCompartmentSphericalMeanModel.

    Parameters
    ----------
    model : MultiCompartmentSphericalMeanModel
    acquisition_scheme : DmipyAcquisitionScheme
    broadcast : bool
        If True (default) the per-shell signal is broadcast to all measurements,
        giving shape (N_meas,) for comparison against full-measurement data.
        If False, returns the per-shell signal (N_shells,) -- which is what the
        spherical-mean FIT needs, since it reduces the data to per-shell means
        (estimate_spherical_mean_multi_shell).  Broadcasting the model to N_meas
        while the data is (N_shells,) is a shape mismatch in the loss.

    Returns
    -------
    forward_fn : callable
        forward_fn(params_scaled) -> jnp.array, shape (N_meas,) if broadcast
        else (N_shells,).
    """
    for m in model.models:
        if type(m) not in set(_JAX_SM_MODEL_FNS) | set(_JAX_SM_MODEL_FACTORIES):
            raise NotImplementedError(
                "No JAX spherical mean implementation for model {} ({}). "
                "Add it to _JAX_SM_MODEL_FNS or _JAX_SM_MODEL_FACTORIES in "
                "multicompartment_jax.py.".format(m.__class__.__name__, type(m))
            )

    dispatch = _build_dispatch(model, acquisition_scheme, spherical_mean=True)
    N_shells   = acquisition_scheme.N_shells
    b0_mask    = acquisition_scheme.shell_b0_mask
    non_b0_idx = np.where(~b0_mask)[0]           # static indices into shell array
    shell_bvals_non_b0 = jnp.array(
        acquisition_scheme.shell_bvalues[~b0_mask])   # (N_non_b0,), static
    is_multi   = model.N_models > 1
    # Static mapping from measurement → shell index for broadcasting
    shell_indices_jax = jnp.array(acquisition_scheme.shell_indices)  # (N_meas,)

    def sm_forward_fn(params_scaled):
        non_b0_signal = jnp.zeros(len(non_b0_idx))
        for entry in dispatch:
            model_params  = _extract_params_jax(params_scaled, entry['param_slices'])
            # Call the SM function (uses shell bvalues, no orientation)
            compartment_sm = entry['jax_fn'](shell_bvals_non_b0, model_params)
            vf = params_scaled[entry['vf_idx']] if is_multi else jnp.array(1.0)
            non_b0_signal = non_b0_signal + vf * compartment_sm

        # Build full (N_shells,) output: b0 = 1.0, others = computed
        signal_shells = jnp.ones(N_shells)
        signal_shells = signal_shells.at[non_b0_idx].set(non_b0_signal)
        # (N_meas,) broadcast for full-data comparison, or (N_shells,) for the
        # spherical-mean fit (data is per-shell means).
        return signal_shells[shell_indices_jax] if broadcast else signal_shells

    return jax.jit(sm_forward_fn)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_mc_forward_fn(model, acquisition_scheme):
    """Build a jax.jit-compiled forward function for a MultiCompartmentModel.

    Parameters
    ----------
    model : MultiCompartmentModel
        Must only contain models present in _JAX_MODEL_FNS. Raises
        NotImplementedError otherwise.
    acquisition_scheme : DmipyAcquisitionScheme

    Returns
    -------
    forward_fn : callable
        Signature: forward_fn(params_scaled) -> jnp.array, shape (N_meas,)
        where params_scaled is a 1-D jnp.array in SI units (same layout as
        model.scales_for_optimization * normalized_params).
    """
    # Cylinder models use bessel_jn / Van Gelderen sums that overflow float32.
    # Enable x64 globally when any cylinder model is present so that the JIT
    # function is compiled in float64 mode. Ball/Stick/Zeppelin/CSD models
    # don't trigger this path and continue to run in the default precision.
    _FLOAT64_REQUIRED = {
        C2CylinderStejskalTannerApproximation,
        C3CylinderCallaghanApproximation,
        C4CylinderGaussianPhaseApproximation,
        DD1GammaDistributed,   # wraps C3 internally via vmap
        DD2PoissonDistributed,  # wraps C3 internally via vmap
    }
    if any(type(m) in _FLOAT64_REQUIRED for m in model.models):
        jax.config.update("jax_enable_x64", True)

    # Validate all models have JAX implementations
    _all_jax_models = set(_JAX_MODEL_FNS) | set(_JAX_MODEL_FACTORIES)
    for m in model.models:
        if type(m) not in _all_jax_models:
            raise NotImplementedError(
                "No JAX implementation for model {} ({}). "
                "Port it in Phase 5 by adding a JAX function to "
                "dmipy/jax/signal_models_jax.py and registering it in "
                "_JAX_MODEL_FNS.".format(m.__class__.__name__, type(m))
            )

    # --- Pre-compute static dispatch table at factory time (Python, not JAX) --
    # Build slice info: for each compartment, which indices in params_scaled
    # correspond to its parameters and (if multi-compartment) its vf.
    dispatch = _build_dispatch(model, acquisition_scheme)
    scheme_jax = scheme_to_jax(acquisition_scheme)
    N_meas = acquisition_scheme.number_of_measurements
    is_multi = model.N_models > 1

    # Per-compartment S0 scale factors (baked into closure as compile-time constants)
    if hasattr(model, 'S0_responses'):
        rho_list = [float(r) for r in model.S0_responses]
    else:
        rho_list = [1.0] * model.N_models

    # --- Static indices for global parameters (T1, eta) ---------------------
    name_to_start = {}
    idx = 0
    for param_name, card in model.parameter_cardinality.items():
        name_to_start[param_name] = idx
        idx += card

    has_eta = 'eta' in model.parameter_cardinality
    has_S0_global = 'S0_global' in model.parameter_cardinality
    eta_idx = name_to_start.get('eta')
    S0_global_idx = name_to_start.get('S0_global')

    # --- single-TE b0 normalisation of the MODEL --------------------------
    # The fit normalises the DATA by its measured b0 when (single TE, no
    # S0_tissue, no T2 fit); see MultiCompartmentModel.fit. Standard diffusion
    # models satisfy E(b0)=1 by construction, so for them this is a no-op. But a
    # b-INDEPENDENT attenuation factor (e.g. the extra-axonal surface-relaxivity
    # weight W = exp(-rho (S/V) . TE)) makes E(b0) = f + (1-f)W != 1, so the
    # un-normalised model would be compared against b0-normalised data on a
    # different scale -- biasing the recovered microstructure. In a single-TE
    # experiment the b-independent amplitude is degenerate with S0 and only the
    # b0-normalised (relative IA<->EA) reshaping is identifiable, so we divide
    # the model by its own b0 here to match the data convention exactly.
    has_T2 = any(p.endswith('_T2') for p in model.parameter_cardinality)
    _TE = getattr(acquisition_scheme, 'TE', None)
    _single_TE = _TE is None or len(np.unique(np.atleast_1d(_TE))) == 1
    _has_S0_tissue = getattr(model, 'S0_tissue_responses', None) is not None
    normalize_b0 = (_single_TE and not has_S0_global and not has_eta
                    and not has_T2 and not _has_S0_tissue)
    b0_idx_jax = jnp.array(np.where(acquisition_scheme.b0_mask)[0]) \
        if normalize_b0 else None

    # Capture dispatch and scheme_jax in closure; they are Python-level static.
    def forward_fn(params_scaled):
        """params_scaled : 1-D jnp.array, SI units, layout matching
        model.scales_for_optimization."""
        signal = jnp.zeros(N_meas)
        for i, entry in enumerate(dispatch):
            # Extract this compartment's parameter dict from the flat vector.
            model_params = _extract_params_jax(params_scaled, entry['param_slices'])
            compartment_signal = entry['jax_fn'](scheme_jax, model_params)
            if is_multi:
                vf = params_scaled[entry['vf_idx']]
            else:
                vf = jnp.array(1.0)
            rho = jnp.array(rho_list[i])
            signal = signal + rho * vf * compartment_signal

        # Single-TE b0 normalisation (no-op for standard models, E(b0)=1;
        # divides out the b-independent surface-relaxivity amplitude otherwise).
        if normalize_b0:
            signal = signal / jnp.mean(signal[b0_idx_jax])

        # Global S0: absorbs absolute per-voxel signal scale.
        if has_S0_global:
            S0_g = params_scaled[S0_global_idx]
            signal = signal * S0_g

        # Rician noise floor: sqrt(S^2 + eta^2).
        if has_eta:
            eta = params_scaled[eta_idx]
            signal = jnp.sqrt(signal ** 2 + eta ** 2)

        return signal

    return jax.jit(forward_fn)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_link_resolvers(model, name_to_start):
    """Build JAX resolver closures for parameters removed from the flat vector.

    A parameter is absent from ``name_to_start`` when it was removed by:
      - ``set_fixed_parameter``  → ReturnFixedValue link (no dependencies)
      - ``set_equal_parameter``  → parameter_equality link (one dependency)
      - ``set_tortuous_parameter`` → T1_tortuosity link (lambda_par + vf deps)

    Returns
    -------
    resolvers : dict  {full_param_name: callable(params_scaled) -> jnp scalar/array}
    """
    from ..utils.utils import T1_tortuosity, parameter_equality
    from ..core.model_properties import ReturnFixedValue

    resolvers = {}
    for link in model.parameter_links:
        dep_model, local_param, fn, deps = link

        # Reconstruct full name for this dependent parameter
        dep_model_idx = model.models.index(dep_model)
        dep_model_name = model.model_names[dep_model_idx]
        full_name = dep_model_name + local_param

        if full_name in name_to_start:
            continue   # parameter is free — no resolver needed

        if isinstance(fn, ReturnFixedValue):
            # Fixed constant — cast to the parameter vector's dtype at call time
            # so it matches the computation precision (float32 or, under x64,
            # float64). Forcing float32 here silently corrupts a float64 forward
            # under jit (eager promotion hides it, XLA does not).
            resolvers[full_name] = (
                lambda _vec, _v=fn.value: jnp.asarray(_v, dtype=_vec.dtype))

        elif fn is parameter_equality:
            # Equality link: dep = master
            master_model_obj, master_local = deps[0]
            if master_model_obj is None:
                # Global parameter (e.g. partial volume) — find by local name
                master_full = master_local
            else:
                master_model_idx = model.models.index(master_model_obj)
                master_model_name = model.model_names[master_model_idx]
                master_full = master_model_name + master_local
            if master_full not in name_to_start:
                continue   # master is also absent (chained link) — skip for now
            master_start = name_to_start[master_full]
            master_card = model.parameter_cardinality[master_full]
            if master_card == 1:
                resolvers[full_name] = lambda vec, s=master_start: vec[s]
            else:
                resolvers[full_name] = (
                    lambda vec, s=master_start, c=master_card: vec[s:s + c])

        elif isinstance(fn, T1_tortuosity):
            # Tortuosity: lambda_perp = (1 - f) * lambda_par
            # deps: [(model_lpar, 'lambda_par'), (None, 'partial_volume_X'), ...]
            lpar_model_obj, lpar_local = deps[0]
            lpar_model_idx = model.models.index(lpar_model_obj)
            lpar_model_name = model.model_names[lpar_model_idx]
            lpar_full = lpar_model_name + lpar_local

            # Resolve lambda_par — may itself be a free param or an equality link
            if lpar_full in name_to_start:
                lpar_start = name_to_start[lpar_full]
                lpar_card = model.parameter_cardinality[lpar_full]
                get_lpar = (lambda vec, s=lpar_start: vec[s])
            elif lpar_full in resolvers:
                get_lpar = resolvers[lpar_full]
            else:
                continue   # can't resolve lambda_par — skip

            # Volume fraction deps (partial_volume_X entries)
            vf_getters = []
            for vf_model_obj, vf_local in deps[1:]:
                if vf_model_obj is None:
                    vf_full = vf_local   # e.g. 'partial_volume_0'
                else:
                    vf_model_idx = model.models.index(vf_model_obj)
                    vf_full = model.model_names[vf_model_idx] + vf_local
                if vf_full in name_to_start:
                    s = name_to_start[vf_full]
                    vf_getters.append(lambda vec, _s=s: vec[_s])

            S0_intra = fn.S0_intra
            S0_extra = fn.S0_extra

            def _tortuosity_resolver(vec, _get_lpar=get_lpar,
                                     _vf_getters=vf_getters,
                                     _s0i=S0_intra, _s0e=S0_extra):
                lpar = _get_lpar(vec)
                if len(_vf_getters) == 1:
                    # nested fraction: vf_intra + (1 - vf_intra) = 1
                    vf_intra = _vf_getters[0](vec)
                    vf_extra = 1.0 - vf_intra
                else:
                    vf_intra = _vf_getters[0](vec)
                    vf_extra = _vf_getters[1](vec)
                f = ((vf_intra * _s0e) /
                     (vf_intra * _s0e + vf_extra * _s0i + 1e-12))
                return (1.0 - f) * lpar

            resolvers[full_name] = _tortuosity_resolver

    return resolvers


def _build_dispatch(model, acquisition_scheme=None, *, spherical_mean=False):
    """Build the static per-compartment dispatch table.

    Returns a list of dicts, one per model, containing:
      - 'jax_fn'      : callable (scheme_jax, params_dict) -> signal
      - 'param_slices': list of param entries, each one of:
            ('direct',   local_name, start, card)  — read from flat vector
            ('computed', local_name, resolver_fn)  — resolver_fn(vec) -> value
      - 'vf_idx'      : int index of this compartment's volume fraction
                        in the flat parameter vector (None if single model)

    acquisition_scheme is forwarded to Watson/Bingham factories so they can
    use SH convolution instead of numerical integration.
    spherical_mean selects the shell-level registry, including factories
    that capture per-shell timing (NEXI).
    """
    # Walk parameter_cardinality (ordered) to compute flat vector indices
    name_to_start = {}
    idx = 0
    for param_name, card in model.parameter_cardinality.items():
        name_to_start[param_name] = idx
        idx += card

    # Build resolvers for all linked/fixed parameters once
    link_resolvers = _build_link_resolvers(model, name_to_start)

    dispatch = []
    for model_obj, model_name in zip(model.models, model.model_names):
        # Collect this model's param slices (full MC name → local name → slice)
        param_slices = []
        for local_param in model_obj.parameter_ranges:
            full_name = model_name + local_param
            if full_name in name_to_start:
                card = model.parameter_cardinality[full_name]
                start = name_to_start[full_name]
                param_slices.append(('direct', local_param, start, card))
            elif full_name in link_resolvers:
                # Linked or fixed — evaluated via JAX closure at call time
                param_slices.append(('computed', local_param,
                                     link_resolvers[full_name]))
            # else: parameter genuinely absent (e.g. chained link not resolved)
            # — leave it out; the JAX fn will raise a clear KeyError if it
            # truly needs it.

        # Volume fraction index
        vf_idx = None
        if model.N_models > 1:
            # partial_volume_i corresponds to position i in partial_volume_names
            pv_idx = model.model_names.index(model_name)
            pv_name = model.partial_volume_names[pv_idx]
            vf_idx = name_to_start[pv_name]

        # Resolve per-instance JAX function (factory models take priority)
        model_type = type(model_obj)
        if spherical_mean and model_type in _JAX_SM_MODEL_FACTORIES:
            jax_fn = _JAX_SM_MODEL_FACTORIES[model_type](model_obj, acquisition_scheme)
        elif spherical_mean:
            jax_fn = _JAX_SM_MODEL_FNS[model_type]
        elif model_type in _JAX_MODEL_FACTORIES:
            jax_fn = _JAX_MODEL_FACTORIES[model_type](model_obj, acquisition_scheme)
        else:
            jax_fn = _JAX_MODEL_FNS[model_type]

        dispatch.append({
            'jax_fn':     jax_fn,
            'model_type': model_type,      # needed by SM dispatch
            'param_slices': param_slices,
            'vf_idx':     vf_idx,
        })

    return dispatch


def _extract_params_jax(params_scaled, param_slices):
    """Extract a model's parameters from the flat scaled vector into a dict.

    param_slices entries are either:
      ('direct',   local_name, start, card)  — slice from flat vector
      ('computed', local_name, resolver_fn)  — call resolver_fn(params_scaled)

    Orientation parameters (cardinality 2) are returned as a 2-element array.
    Scalar parameters (cardinality 1) are returned as 0-d arrays (scalars).
    """
    out = {}
    for entry in param_slices:
        if entry[0] == 'direct':
            _, local_name, start, card = entry
            if card == 1:
                out[local_name] = params_scaled[start]
            else:
                out[local_name] = params_scaled[start:start + card]
        else:  # 'computed'
            _, local_name, resolver_fn = entry
            out[local_name] = resolver_fn(params_scaled)
    return out
