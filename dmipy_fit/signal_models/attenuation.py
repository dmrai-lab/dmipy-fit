r"""Composable, occupancy-gated attenuation factors for compartment models.

The physical attenuations that ride on top of the diffusion signal --- transverse
($T_2$) relaxation and surface relaxivity --- are multiplicative factors gated by the
transverse occupancy $\tau_\perp$ (``tau_perp`` from the acquisition scheme, or TE
when unset: a spin echo is transverse throughout). They are
not diffusion models and not compartment-specific, so they are expressed here as
composable *factors* that attach to any compartment via :class:`OccupancyGatedModel`,
rather than baked into each compartment (which makes them opt-*out*) or multiplied out
as a class per effect-combination (which explodes combinatorially)::

    from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
    from dmipy_fit.signal_models.attenuation import (
        OccupancyGatedModel, TransverseRelaxation, SurfaceRelaxivity)

    ea = OccupancyGatedModel(G2Zeppelin(),
                             [TransverseRelaxation(), SurfaceRelaxivity()])

``OccupancyGatedModel`` is itself a ``ModelProperties`` compartment: it re-exposes
the base model's diffusion parameters plus each factor's parameter, and returns
``base(scheme) * prod(factor_i)``.  It therefore plugs into ``MultiCompartmentModel``
/ CSD with no framework changes, and **any subset of effects composes by listing
more factors**.

Opt-in is at the *compartment* level (the dmipy idiom): the base compartments are
untouched, so existing models keep their exact parameter set.  A factor that also
exists as a baked-in base parameter (``T2``) is *bypassed* on the base when its
factor is present, so it is never double-counted.
"""
from collections import OrderedDict
import numpy as np

from ..utils import utils
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import AnisotropicSignalModelProperties

__all__ = [
    'OccupancyGatedModel', 'TransverseRelaxation', 'LongitudinalRelaxation',
    'SurfaceRelaxivity',
    'IntraPoreSurfaceRelaxivity', 'ExteriorSurfaceRelaxivity',
    'IntraPoreSurfaceMT', 'ExteriorSurfaceMT', 'MTSaturation',
]


def _tau_perp(scheme):
    """Transverse occupancy time -- the window over which magnetisation is transverse
    and thus subject to $T_2$ / surface relaxivity.

    A plain spin echo keeps the magnetisation transverse for the whole echo, so the
    transverse time equals the echo time TE (``tau_perp`` is unset). A stimulated
    echo stores the magnetisation longitudinally during the mixing time, so its
    transverse occupancy (the two encoding lobes, ``2*delta``) is shorter than TE;
    such schemes carry an explicit ``tau_perp``. Falls back to TE when unset."""
    tau_perp = getattr(scheme, 'tau_perp', None)
    if tau_perp is not None:
        return tau_perp
    return scheme.TE


def _tau_par(scheme):
    """Longitudinal (storage) occupancy time.

    During a stimulated-echo mixing time TM the magnetisation is stored along the
    field, so only longitudinal ($T_1$) relaxation accrues. A plain spin echo has
    no mixing time (``TM`` unset) and returns ``None`` -> the factor is identity."""
    return getattr(scheme, 'TM', None)


def _is_set(x):
    return x is not None and not np.isnan(np.asarray(x, dtype=float).flat[0])


class AttenuationFactor(object):
    """Base class: a named, occupancy-gated multiplicative factor.

    Subclasses declare ``parameter_ranges``/``parameter_scales``/``parameter_types``
    (dicts of the parameter(s) they add) and implement ``factor(scheme, mu_cart,
    base_params, **params)`` returning the multiplicative attenuation (scalar or
    per-measurement array).  Return ``1.0`` when the factor is inactive."""
    parameter_ranges = {}
    parameter_scales = {}
    parameter_types = {}
    # Orientation-INDEPENDENT factors (T2, surface relaxivity) are a scalar per
    # shell, so they factor straight through the angular spherical mean. An
    # orientation-DEPENDENT factor would set this False and be handled via the
    # rotational-harmonics path instead of a scalar multiply in spherical_mean.
    spherical_mean_separable = True

    def factor(self, acquisition_scheme, mu_cart, base_params, **params):
        raise NotImplementedError


class TransverseRelaxation(AttenuationFactor):
    r"""Transverse relaxation $\exp(-\tau_\perp/T_2)$ (gated by transverse occupancy)."""
    parameter_ranges = {'T2': (1e-3, 10.)}
    parameter_scales = {'T2': 1.}
    parameter_types = {'T2': 'normal'}

    def factor(self, acquisition_scheme, mu_cart, base_params, T2=None):
        if not _is_set(T2) or getattr(acquisition_scheme, 'TE', None) is None:
            return 1.0
        return np.exp(-_tau_perp(acquisition_scheme) / T2)


class LongitudinalRelaxation(AttenuationFactor):
    r"""Longitudinal relaxation $\exp(-\tau_\parallel/T_1)$ (gated by longitudinal storage).

    The longitudinal sibling of :class:`TransverseRelaxation`.  During a
    stimulated-echo mixing time the magnetisation is parked along the field, so
    transverse effects ($T_2$, surface relaxivity) are switched off and only $T_1$
    acts over the storage time $\tau_\parallel = \mathrm{TM}$.  A plain spin echo has
    no mixing time (``TM`` unset -> ``_tau_par`` is None), so the factor is the
    identity 1.0."""
    parameter_ranges = {'T1': (1e-2, 10.)}
    parameter_scales = {'T1': 1.}
    parameter_types = {'T1': 'normal'}

    def factor(self, acquisition_scheme, mu_cart, base_params, T1=None):
        tau_par = _tau_par(acquisition_scheme)
        if not _is_set(T1) or tau_par is None:
            return 1.0
        return np.exp(-tau_par / T1)


class SurfaceRelaxivity(AttenuationFactor):
    r"""Surface relaxivity $\exp(-\rho\,(S/V)\,\tau_\perp)$ (transverse-gated).

    ``S/V`` is taken from ``surface_to_volume`` if given (1/m), else from a base
    ``diameter`` parameter (``S/V = 4/d`` for a cylinder) when present."""
    parameter_ranges = {'surface_relaxivity': (0., 50e-6)}
    parameter_scales = {'surface_relaxivity': 1e-6}
    parameter_types = {'surface_relaxivity': 'normal'}

    def __init__(self, surface_to_volume=None):
        self.surface_to_volume = surface_to_volume

    def factor(self, acquisition_scheme, mu_cart, base_params, surface_relaxivity=None):
        if not _is_set(surface_relaxivity) \
                or getattr(acquisition_scheme, 'TE', None) is None:
            return 1.0
        sv = self.surface_to_volume
        if sv is None:
            d = base_params.get('diameter')
            if not _is_set(d):
                return 1.0
            sv = 4.0 / d
        return np.exp(-surface_relaxivity * sv * _tau_perp(acquisition_scheme))


class IntraPoreSurfaceRelaxivity(AttenuationFactor):
    r"""Intra-pore surface relaxivity for an axon modelled as a Gamma distribution of
    myelinated cylinders (eq:b_hat_int_dist, Brownstein-Tarr, Gamma-averaged).

    The backend substrate is parameterized by the *outer* (myelin) diameter
    distribution and the g-ratio; the lumen the water actually relaxes against has
    *inner* diameter ``d_in = g * d_out``.  So a Stick (zero-radius) intra-axonal
    compartment still carries a real geometry: the Gamma scale used here is
    ``g * gamma_scale_outer_diameter`` and ``g_ratio`` is exposed as a fittable
    (and linkable) parameter, derived inner radius = g x outer radius."""
    parameter_ranges = {'surface_relaxivity': (0., 50e-6), 'g_ratio': (0.5, 0.95)}
    parameter_scales = {'surface_relaxivity': 1e-6, 'g_ratio': 1.}
    parameter_types = {'surface_relaxivity': 'normal', 'g_ratio': 'normal'}

    def __init__(self, gamma_shape=2.0, gamma_scale_outer_diameter=0.304e-6,
                 volume_weighted=True):
        self.gamma_shape = gamma_shape
        self.gamma_scale_outer_diameter = gamma_scale_outer_diameter
        self.volume_weighted = volume_weighted

    def factor(self, acquisition_scheme, mu_cart, base_params,
               surface_relaxivity=None, g_ratio=None):
        if not _is_set(surface_relaxivity) \
                or getattr(acquisition_scheme, 'TE', None) is None:
            return 1.0
        from ..white_matter.surface import b_hat_ia
        g = float(g_ratio) if _is_set(g_ratio) else 1.0
        inner_scale = g * self.gamma_scale_outer_diameter      # inner = g x outer
        return b_hat_ia(self.gamma_shape, inner_scale, float(surface_relaxivity),
                        _tau_perp(acquisition_scheme), self.volume_weighted)


class ExteriorSurfaceRelaxivity(AttenuationFactor):
    r"""Extra-axonal long-time surface relaxivity $\exp(-\rho_{ext}(S_{ext}/V)\tau_\perp)$
    (eq:b_hat_ext_long).  ``S_ext_over_V`` (1/m) is the exterior surface-to-EA-volume
    ratio of the substrate, supplied at construction."""
    parameter_ranges = {'surface_relaxivity': (0., 50e-6)}
    parameter_scales = {'surface_relaxivity': 1e-6}
    parameter_types = {'surface_relaxivity': 'normal'}

    def __init__(self, S_ext_over_V):
        self.S_ext_over_V = S_ext_over_V

    def factor(self, acquisition_scheme, mu_cart, base_params, surface_relaxivity=None):
        if not _is_set(surface_relaxivity) \
                or getattr(acquisition_scheme, 'TE', None) is None:
            return 1.0
        from ..white_matter.surface import b_hat_ea_long
        return b_hat_ea_long(float(surface_relaxivity), self.S_ext_over_V,
                             _tau_perp(acquisition_scheme))


class IntraPoreSurfaceMT(IntraPoreSurfaceRelaxivity):
    r"""Intra-pore surface relaxivity **+ magnetization transfer** on the same wall.

    MT is a second wall reactivity, so the free-water transverse factor is
    $\hat B_{ia}(\rho + \kappa_{MT})$ -- a SINGLE Gamma-averaged Bessel evaluated at
    the COMBINED reactivity, not a product $\hat B(\rho)\,\hat B(\kappa)$ (the
    interior Bessel factor is not exponential, so it does not factorise).  This is
    why MT is degenerate with surface relaxivity in the diffusion signal; only the
    off-resonance qMT Z-spectrum (:mod:`dmipy_fit.white_matter.magnetization_transfer`)
    separates them.  ``kappa_MT`` (m/s) defaults inert (0)."""
    parameter_ranges = {**IntraPoreSurfaceRelaxivity.parameter_ranges,
                        'kappa_MT': (0., 50e-6)}
    parameter_scales = {**IntraPoreSurfaceRelaxivity.parameter_scales,
                        'kappa_MT': 1e-6}
    parameter_types = {**IntraPoreSurfaceRelaxivity.parameter_types,
                       'kappa_MT': 'normal'}

    def factor(self, acquisition_scheme, mu_cart, base_params,
               surface_relaxivity=None, g_ratio=None, kappa_MT=None):
        rho = float(surface_relaxivity) if _is_set(surface_relaxivity) else 0.0
        kmt = float(kappa_MT) if _is_set(kappa_MT) else 0.0
        total = rho + kmt
        if total == 0.0 or getattr(acquisition_scheme, 'TE', None) is None:
            return 1.0
        from ..white_matter.surface import b_hat_ia
        g = float(g_ratio) if _is_set(g_ratio) else 1.0
        inner_scale = g * self.gamma_scale_outer_diameter      # inner = g x outer
        return b_hat_ia(self.gamma_shape, inner_scale, total,
                        _tau_perp(acquisition_scheme), self.volume_weighted)


class ExteriorSurfaceMT(ExteriorSurfaceRelaxivity):
    r"""Extra-axonal surface relaxivity **+ magnetization transfer**: the exterior
    long-time factor $\exp(-(\rho+\kappa_{MT})(S_{ext}/V)\,\tau_\perp)$.  The exterior
    factor IS exponential, so $\rho$ and $\kappa_{MT}$ add in the rate -- the same
    combined wall reactivity as the interior factor.  ``kappa_MT`` (m/s) inert (0) by
    default."""
    parameter_ranges = {**ExteriorSurfaceRelaxivity.parameter_ranges,
                        'kappa_MT': (0., 50e-6)}
    parameter_scales = {**ExteriorSurfaceRelaxivity.parameter_scales,
                        'kappa_MT': 1e-6}
    parameter_types = {**ExteriorSurfaceRelaxivity.parameter_types,
                       'kappa_MT': 'normal'}

    def factor(self, acquisition_scheme, mu_cart, base_params,
               surface_relaxivity=None, kappa_MT=None):
        rho = float(surface_relaxivity) if _is_set(surface_relaxivity) else 0.0
        kmt = float(kappa_MT) if _is_set(kappa_MT) else 0.0
        total = rho + kmt
        if total == 0.0 or getattr(acquisition_scheme, 'TE', None) is None:
            return 1.0
        from ..white_matter.surface import b_hat_ea_long
        return b_hat_ea_long(total, self.S_ext_over_V, _tau_perp(acquisition_scheme))


class MTSaturation(AttenuationFactor):
    r"""qMT off-resonance saturation: the degeneracy-LIFTING observable.

    For measurements carrying a saturation block on the scheme (``mt_offset_hz`` and
    ``mt_b1_hz`` per measurement, e.g. via
    :func:`dmipy_fit.white_matter.composition.attach_mt_saturation`), the free-water
    signal is multiplied by the two-pool longitudinal steady state $M_{z,a}/M_{0a}$:
    an off-resonance pulse saturates the broad (short-$T_2^b$) bound pool and transfers
    to the free pool via exchange, reducing the excited magnetisation.  A pure
    surface-relaxivity sink (no bound pool, $\kappa_{MT}=0$) produces NO off-resonance
    saturation, so this is what separates $\kappa_{MT}$ from $\rho$ (which are
    degenerate in the diffusion signal).

    Steady-state (Sled--Pike / Henkelman) CW/long-pulse approximation.  ``k_f =
    kappa_MT * (S/V)`` uses the SAME wall geometry as surface relaxivity; the factor
    READS the shared ``kappa_MT``/``T1``/``T2`` (free pool) and OWNS the bound-pool
    ``dwell_time`` (=1/k_r), ``T2_bound``, ``T1_bound``.  Returns 1.0 (identity) for
    measurements with no saturation block, and exactly 1.0 when ``kappa_MT=0``."""
    parameter_ranges = {'kappa_MT': (0., 50e-6), 'T1': (1e-2, 10.), 'T2': (1e-3, 10.),
                        'dwell_time': (1e-4, 1.0), 'T2_bound': (1e-7, 1e-3),
                        'T1_bound': (1e-2, 10.)}
    parameter_scales = {'kappa_MT': 1e-6, 'T1': 1., 'T2': 1.,
                        'dwell_time': 1e-3, 'T2_bound': 1e-5, 'T1_bound': 1.}
    parameter_types = {k: 'normal' for k in parameter_ranges}

    def __init__(self, S_over_V=None, gamma_shape=None,
                 gamma_scale_outer_diameter=None, volume_weighted=True):
        # exterior: pass a fixed S_over_V (1/m).  interior: pass gamma_shape +
        # gamma_scale_outer_diameter -> S/V = <4/d_inner> uses the fittable g_ratio.
        self.S_over_V = S_over_V
        self.gamma_shape = gamma_shape
        self.gamma_scale_outer_diameter = gamma_scale_outer_diameter
        self.volume_weighted = volume_weighted
        if gamma_shape is not None:                       # interior: expose g_ratio too
            self.parameter_ranges = {**self.parameter_ranges, 'g_ratio': (0.5, 0.95)}
            self.parameter_scales = {**self.parameter_scales, 'g_ratio': 1.}
            self.parameter_types = {**self.parameter_types, 'g_ratio': 'normal'}

    def factor(self, acquisition_scheme, mu_cart, base_params, kappa_MT=None,
               T1=None, T2=None, dwell_time=None, T2_bound=None, T1_bound=None,
               g_ratio=None):
        off = getattr(acquisition_scheme, 'mt_offset_hz', None)
        b1 = getattr(acquisition_scheme, 'mt_b1_hz', None)
        if off is None or b1 is None:
            return 1.0                                     # no saturation block
        # need the bound-pool + free-pool T1 to solve the steady state; kappa_MT=0 is
        # allowed (k_f=0 -> pure free-pool direct saturation, ~1 in the qMT regime).
        if not (_is_set(kappa_MT) and _is_set(dwell_time)
                and _is_set(T2_bound) and _is_set(T1_bound) and _is_set(T1)):
            return 1.0
        from ..white_matter.surface import mean_inv_diameter_4
        from ..white_matter.magnetization_transfer import (
            two_pool_steady_state, lorentzian_lineshape)
        if self.S_over_V is not None:
            sv = self.S_over_V
        else:
            g = float(g_ratio) if _is_set(g_ratio) else 1.0
            sv = mean_inv_diameter_4(self.gamma_shape,
                                     g * self.gamma_scale_outer_diameter,
                                     self.volume_weighted)
        k_f = float(kappa_MT) * sv
        k_r = 1.0 / float(dwell_time)
        off = np.asarray(off, dtype=float)
        w1 = 2.0 * np.pi * np.asarray(b1, dtype=float)     # rad/s, per measurement
        T2a = float(T2) if _is_set(T2) else 0.08
        W_a = w1 ** 2 * lorentzian_lineshape(off, T2a)
        W_b = w1 ** 2 * lorentzian_lineshape(off, float(T2_bound))
        # b1=0 measurements -> W=0 -> two_pool_steady_state returns exactly 1.0
        return two_pool_steady_state(k_f, k_r, 1.0 / float(T1), 1.0 / float(T1_bound),
                                     W_a, W_b)


class OccupancyGatedModel(ModelProperties, AnisotropicSignalModelProperties):
    """A compartment = a base diffusion model x composable attenuation factors.

    Parameters
    ----------
    model : ModelProperties
        The base diffusion compartment (e.g. G2Zeppelin, G3TemporalZeppelin, C1Stick).
    factors : list of AttenuationFactor
        Occupancy-gated factors to layer on top, in any combination.
    """
    _model_type = 'CompartmentModel'

    def __init__(self, model, factors=None):
        self.model = model
        self.factors = list(factors or [])
        # delegate acquisition requirements to the base diffusion model
        self._required_acquisition_parameters = list(getattr(
            model, '_required_acquisition_parameters', []))

        fr, fs, ft = OrderedDict(), OrderedDict(), OrderedDict()
        for f in self.factors:
            fr.update(f.parameter_ranges)
            fs.update(f.parameter_scales)
            ft.update(f.parameter_types)

        base_r = model.parameter_ranges
        base_s = model.parameter_scales
        base_t = model.parameter_types
        # base diffusion params that a factor does NOT own (factors own T2/T1/...);
        # the rest are exposed by the factor and bypassed on the base to avoid
        # double-counting (e.g. T2 when a TransverseRelaxation factor is present).
        self._base_names = [k for k in base_r if k not in fr]
        self._base_bypass = [k for k in base_r if k in fr]

        self._parameter_ranges = OrderedDict(
            [(k, base_r[k]) for k in self._base_names] + list(fr.items()))
        self._parameter_scales = OrderedDict(
            [(k, base_s[k]) for k in self._base_names] + list(fs.items()))
        self._parameter_types = OrderedDict(
            [(k, base_t[k]) for k in self._base_names] + list(ft.items()))

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        base_params = {k: kwargs.get(k) for k in self._base_names}
        for k in self._base_bypass:
            base_params[k] = None          # factor owns it; don't let the base apply it
        E = self.model(acquisition_scheme, use_jax=use_jax, **base_params)

        mu = kwargs.get('mu', getattr(self.model, 'mu', None))
        mu_cart = utils.unitsphere2cart_1d(mu) if mu is not None else None
        for f in self.factors:
            fp = {k: kwargs.get(k) for k in f.parameter_ranges}
            E = E * f.factor(acquisition_scheme, mu_cart, base_params, **fp)
        return E

    def spherical_mean(self, acquisition_scheme, **kwargs):
        # The diffusion angular structure and any orientation-DEPENDENT factors go
        # through the rotational-harmonics spherical mean (base-class behaviour).
        # Orientation-INDEPENDENT factors (T2 / surface relaxivity) are a scalar per
        # shell, so they factor straight through the angular mean -- applied here on
        # every shell (b0 included) at the shell TE, which the rotational-harmonics
        # scheme (no TE) drops. This gives spherical-mean <-> full-model parity for
        # the separable factors.
        E_mean = super(OccupancyGatedModel, self).spherical_mean(
            acquisition_scheme, **kwargs)
        base_params = {k: kwargs.get(k) for k in self._base_names}
        for k in self._base_bypass:
            base_params[k] = None
        sms = acquisition_scheme.spherical_mean_scheme
        mu = kwargs.get('mu', getattr(self.model, 'mu', None))
        mu_cart = utils.unitsphere2cart_1d(mu) if mu is not None else None
        for f in self.factors:
            if not getattr(f, 'spherical_mean_separable', True):
                continue        # orientation-dependent -> already in the rh mean
            fp = {k: kwargs.get(k) for k in f.parameter_ranges}
            E_mean = E_mean * f.factor(sms, mu_cart, base_params, **fp)
        return E_mean
