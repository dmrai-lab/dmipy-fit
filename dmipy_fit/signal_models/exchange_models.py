# -*- coding: utf-8 -*-
"""Two-compartment water exchange models for diffusion MRI.

Implements the generalised Karger exchange framework and published NEXI,
a powder-averaged Stick+Ball signal without a tortuosity constraint
(Jelescu et al. 2022, Eqs. 3--6). The orientation-resolved, tortuous
Stick+Zeppelin pair is built from X0GeneralizedKarger directly.

Architecture
------------
X0GeneralizedKarger  — wraps any two CompartmentModel instances with Karger
                       exchange.  Accepts parameter_links in the same format
                       as MultiCompartmentModel and SD1WatsonDistributed.
X1NEXI              — powder-averages an exchanging Stick + Ball pair,
                       exposing f, D_i, D_e and t_ex for spherical-mean fitting.

References
----------
Jelescu IO, et al. (2022). NeuroImage 256, 119277.
    doi:10.1016/j.neuroimage.2022.119277
Karger J, Pfeifer H, Heink W (1988). Advances in Magnetic Resonance 12, 1-89.
    doi:10.1016/S0065-2385(08)60067-1
"""
from __future__ import division

from collections import OrderedDict

import numpy as np
from dmipy_analytic import exchange

from ..core.model_properties import ModelProperties
from ..core.signal_model_properties import (
    AnisotropicSignalModelProperties, IsotropicSignalModelProperties)
from ..distributions.distribute_models import DistributedModel

__all__ = [
    'X0GeneralizedKarger',
    'X1NEXI',
    'CompartmentSignalDomainError',
    'compartment_exponent',
]

from ..core.constants import DIFFUSIVITY_SCALING


class CompartmentSignalDomainError(ValueError):
    """An effective-medium compartment signal cannot define a real exponent."""


def compartment_exponent(model, scheme, parameters):
    """Diffusion exponent without exp/log round trips for Gaussian pools.

    Restricted pools retain the effective-medium interpretation -log(E).
    Its domain is finite E > 0: zero (including underflow), negative and
    nonfinite signals raise CompartmentSignalDomainError, never a floor.
    """
    from .attenuation import OccupancyGatedModel
    from .cylinder_models import C1Stick
    from .gaussian_models import G1Ball, G2Zeppelin
    from ..utils.utils import unitsphere2cart_1d
    base = model.model if isinstance(model, OccupancyGatedModel) else model
    if isinstance(base, G1Ball):
        return np.asarray(scheme.bvalues) * parameters['lambda_iso']
    if isinstance(base, (C1Stick, G2Zeppelin)):
        axis = unitsphere2cart_1d(parameters['mu'])
        B = scheme.btensor()
        along = np.einsum('i,mij,j->m', axis, B, axis)
        if isinstance(base, C1Stick):
            return parameters['lambda_par'] * along
        return (parameters['lambda_perp'] * np.trace(B, axis1=-2, axis2=-1)
                + (parameters['lambda_par'] - parameters['lambda_perp']) * along)
    signal = np.asarray(base(scheme, **parameters))
    if np.any(~np.isfinite(signal)) or np.any(signal <= 0):
        raise CompartmentSignalDomainError(
            f'{type(base).__name__}: effective-medium -log(E) requires finite E > 0')
    return -np.log(signal)


class _GLNodeScheme:
    """Minimal single-shell acquisition scheme for GL quadrature evaluation.

    Provides exactly the attributes that CompartmentModel.__call__ needs
    (bvalues, gradient_directions, delta, Delta, tau, TE, b0_mask, _G,
    and a btensor() method matching the RotationalHarmonicsAcquisitionScheme
    convention) without the overhead of a full AcquisitionScheme.
    """

    def __init__(self, b, delta, Delta, gradient_directions):
        n = len(gradient_directions)
        self.bvalues = np.full(n, float(b))
        self.gradient_directions = np.asarray(gradient_directions)
        self.delta = np.full(n, float(delta))
        self.Delta = np.full(n, float(Delta))
        self.tau = exchange.diffusion_time(float(delta), float(Delta))
        self.TE = None
        self.b0_mask = np.zeros(n, dtype=bool)
        self._G = None  # signals PGSE path to all models

    def btensor(self):
        """PGSE B-tensor: B[m] = b * n[m]⊗n[m]."""
        n = self.gradient_directions
        b = self.bvalues
        return b[:, None, None] * np.einsum('mi,mj->mij', n, n)


# ---------------------------------------------------------------------------
# X0GeneralizedKarger
# ---------------------------------------------------------------------------

class X0GeneralizedKarger(DistributedModel, AnisotropicSignalModelProperties):
    r"""Generalised two-compartment Karger exchange model.

    Couples any two ``CompartmentModel`` instances via Karger exchange:

        E = σ₊ exp(−λ₊) + σ₋ exp(−λ₋)

    where λ± are eigenvalues of the 2×2 Karger relaxation matrix built from
    the effective dephasing exponents  R1 = −log E1,  R2 = −log E2  of the
    two sub-models under the given acquisition.

    For Gaussian compartments (Ball, Stick, Zeppelin) R_i is linear in b.
    The closed form is exact for constant q in the narrow-pulse limit;
    replacing finite PGSE pulses by t_d = Delta - delta/3 is an approximation
    with exchange. Double-encoding sequences use the waveform propagator.
    For restricted compartments (Cylinder, Sphere) it
    is an effective-medium approximation that treats each compartment as a
    single-mode Gaussian with apparent diffusivity −log(E_i)/b.

    The model inherits the full ``DistributedModel`` parameter-namespace
    machinery, including ``set_fixed_parameter``, ``set_tortuous_parameter``
    (from ``DistributedModel``), and ``parameter_links`` in the same format
    used by ``MultiCompartmentModel`` and ``SD1WatsonDistributed``.

    Parameters
    ----------
    model_intra : CompartmentModel
        The intra-compartment model.
    model_extra : CompartmentModel
        The extra-compartment model.
    parameter_links : list of (model, param_name, callable, arg_list), optional
        Constraints on sub-model parameters resolved at call time.
        Example (tortuosity):
            [(zeppelin, 'lambda_perp', T1_tortuosity(),
              [(zeppelin, 'lambda_par'), (None, 'f')])]

    Own parameters (always present)
    --------------------------------
    mu    : [theta, phi] — shared fibre orientation (radians), if at least
            one sub-model is anisotropic.
    f     : float — intra-compartment volume fraction (0 < f < 1).
    kappa : float — intra→extra exchange rate (s⁻¹).

    This is a pure diffusion+exchange model: it does not expose any relaxation
    parameter of its own. Compartment-wise T2/T1 are an opt-in add-on, exactly
    as for the bare compartments — wrap a sub-model in
    :class:`~dmipy_fit.signal_models.attenuation.OccupancyGatedModel` with a
    ``TransverseRelaxation`` / ``LongitudinalRelaxation`` factor, and its
    ``…_T2`` / ``…_T1`` parameter is read here and folded into the coupled
    relaxation–exchange propagator (the sub-model itself is evaluated
    diffusion-only, so relaxation is never double-counted).

    Sub-model parameters
    --------------------
    Exposed with the prefix ``ClassName_N_``, exactly as in
    ``SD1WatsonDistributed``.  Example for C1Stick + G2Zeppelin:
        C1Stick_1_lambda_par, G2Zeppelin_1_lambda_par,
        G2Zeppelin_1_lambda_perp  (unless linked).

    Notes
    -----
    - Requires a PGSE acquisition scheme (delta and Delta must be set) for
      the exchange diffusion time t_d = Delta − delta/3.
    - Relaxation and exchange do not factorise: a sub-model's ``…_T2`` / ``…_T1``
      is read here and applied inside the matrix-exponential propagator (coupled
      to exchange), not multiplied on afterwards.
    - Compatible with ``SD1WatsonDistributed`` and ``SD2BinghamDistributed``
      (rotational harmonics via 32-point Gauss-Legendre quadrature).

    References
    ----------
    Karger J, Pfeifer H, Heink W (1988). Adv. Magn. Reson. 12, 1-89.
    """

    _citations = {
        'definition': [
            {'key': 'Karger1988',
             'authors': 'Karger J, Pfeifer H, Heink W',
             'title': 'Principles and application of self-diffusion '
                      'measurements by nuclear magnetic resonance',
             'journal': 'Advances in Magnetic Resonance',
             'year': 1988,
             'doi': '10.1016/S0065-2385(08)60067-1'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'pgse_only',
         'name': 'PGSE acquisition required',
         'condition_human': 'Karger exchange uses t_d = Delta - delta/3. '
                            'Not valid for arbitrary gradient waveforms.',
         'severity': 'warning',
         'source_key': 'Karger1988'},
        {'id': 'effective_medium',
         'name': 'Effective-medium approximation for restricted compartments',
         'condition_human': 'R_i = -log(E_i) is exact for Gaussian '
                            'compartments and approximate for restricted ones.',
         'severity': 'info',
         'source_key': 'Karger1988'},
    ]

    _model_type = 'CompartmentModel'
    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions', 'delta', 'Delta']

    # Ranges / scales / types for own parameters (pure diffusion+exchange;
    # relaxation is an opt-in add-on via OccupancyGatedModel on a sub-model).
    _own_ranges = {
        'mu':    ([0, np.pi], [-np.pi, np.pi]),
        'f':     (0.01, 0.99),
        'kappa': (0.1, 200.0),
    }
    _own_scales = {'mu': np.r_[1., 1.], 'f': 1., 'kappa': 1.}
    _own_types  = {
        'mu': 'orientation', 'f': 'normal', 'kappa': 'normal'
    }
    _own_cards  = {'mu': 2, 'f': 1, 'kappa': 1}

    def __init__(self, model_intra, model_extra, parameter_links=None):
        self.models = [model_intra, model_extra]
        self.model_intra = model_intra
        self.model_extra = model_extra
        self.parameter_links = list(parameter_links or [])

        # ── 1. Build combined namespace from both sub-models ──────────────
        # Any relaxation parameters a sub-model exposes (e.g. an
        # OccupancyGatedModel's T2/T1) are kept in the namespace: they are the
        # opt-in relaxation add-on, read in __call__ and folded into the
        # coupled propagator. They are simply not forwarded to the sub-model
        # (see _sub_kwargs), so the sub-model stays diffusion-only.
        self._prepare_parameters([model_intra, model_extra])

        # ── 2. Remove sub-model orientation params; add shared mu ─────────
        has_orientation = any(
            ptype == 'orientation'
            for m in self.models
            for ptype in m.parameter_types.values()
        )
        self._has_orientation = has_orientation
        if has_orientation:
            self._delete_orientation_from_parameters()
            # Redirect sub-model (model, 'mu') entries to the shared 'mu' key
            for model in self.models:
                for param, ptype in model.parameter_types.items():
                    if ptype == 'orientation':
                        self._inverted_parameter_map[(model, param)] = 'mu'

        # ── 3. Add own parameters ─────────────────────────────────────────
        own_names = (['mu'] if has_orientation else []) + ['f', 'kappa']
        for name in own_names:
            self.parameter_ranges[name] = self._own_ranges[name]
            self.parameter_scales[name] = self._own_scales[name]
            self.parameter_types[name]  = self._own_types[name]
            self.parameter_cardinality[name] = self._own_cards[name]
            self._parameter_map[name] = (None, name)
            self._inverted_parameter_map[(None, name)] = name

        # ── 4. Process parameter links ────────────────────────────────────
        self._prepare_parameter_links()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def parameter_names(self):
        return list(self.parameter_ranges.keys())

    def _sub_kwargs(self, model, full_kwargs):
        """Build keyword arguments for one sub-model from the combined dict."""
        params = {}
        for param in model.parameter_ranges:
            if param in ('T2', 'T1'):
                # Relaxation is folded into the coupled propagator in __call__;
                # keep the sub-model diffusion-only so it is never double-counted.
                continue
            key = self._inverted_parameter_map.get((model, param))
            if key is not None:
                v = full_kwargs.get(key)
                if v is not None:
                    params[param] = v
        return params

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def __call__(self, acquisition_scheme, **kwargs):
        r"""Evaluate the Karger exchange signal using the matrix-exponential propagator.

        Uses the finite-RF SE or STE propagator of :mod:`dmipy_analytic.exchange` when relaxation or radiofrequency
        durations are present, and its closed Karger form otherwise.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme
            Must have delta and Delta (PGSE).
        **kwargs : parameter values (full combined namespace)

        Returns
        -------
        E : ndarray, shape (N_measurements,)
        """
        kwargs = self.add_linked_parameters_to_parameters(kwargs)

        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta
        if delta is None or Delta is None:
            raise ValueError(
                "X0GeneralizedKarger requires PGSE timing parameters "
                "delta and Delta (t_d = Delta - delta/3)."
            )

        f     = float(kwargs['f'])
        kappa = float(kwargs['kappa'])

        # Per-compartment relaxation (fall back to inf = no relaxation)
        _INF = 1e10
        # Extract T2 / T1 from sub-model parameter namespaces
        T2_1 = _INF
        T2_2 = _INF
        T1_1 = _INF
        T1_2 = _INF
        for model_name, model in zip(self.model_names, self.models):
            t2_key = model_name + 'T2'
            t1_key = model_name + 'T1'
            t2_v = kwargs.get(t2_key)
            t1_v = kwargs.get(t1_key)
            if model is self.model_intra:
                if t2_v is not None and not np.isnan(float(t2_v)):
                    T2_1 = float(t2_v)
                if t1_v is not None and not np.isnan(float(t1_v)):
                    T1_1 = float(t1_v)
            else:
                if t2_v is not None and not np.isnan(float(t2_v)):
                    T2_2 = float(t2_v)
                if t1_v is not None and not np.isnan(float(t1_v)):
                    T1_2 = float(t1_v)

        # Sub-model signals (T2 handled inside models via tau_perp_SE or TE,
        # so sub-models receive no T2 keyword — _sub_kwargs already excludes T2)
        R1 = compartment_exponent(self.model_intra, acquisition_scheme,
                                  self._sub_kwargs(self.model_intra, kwargs))
        R2 = compartment_exponent(self.model_extra, acquisition_scheme,
                                  self._sub_kwargs(self.model_extra, kwargs))

        double = _double_encodings(acquisition_scheme)
        if double:
            # a filter-exchange (double-encoding) sequence: the generalised Karger equation along its own effective
            # gradient, the pools' diffusivities along each row's direction, exchange and relaxation throughout
            return _waveform_signal(acquisition_scheme, double, R1, R2, f, kappa,
                                    np.array([T2_1, T2_2]), np.array([T1_1, T1_2]), _INF)

        # RF pulse durations (0 = instantaneous; backward compatible)
        tau_exc  = getattr(acquisition_scheme, 'tau_exc', 0.0)
        tau_180  = getattr(acquisition_scheme, 'tau_180', 0.0)
        tau_90   = getattr(acquisition_scheme, 'tau_90',  0.0)

        TM_ = getattr(acquisition_scheme, 'TM', None)
        TE_ = acquisition_scheme.TE

        # Check whether we have meaningful relaxation or RF to warrant matrix expm
        has_relaxation = (T2_1 < _INF or T2_2 < _INF or T1_1 < _INF or T1_2 < _INF)
        has_finite_rf  = (tau_exc > 0 or tau_180 > 0 or tau_90 > 0)

        if has_relaxation or has_finite_rf:
            # Use matrix-exponential propagator
            K = exchange.exchange_matrix(kappa, f)
            M0 = np.array([f, 1.0 - f])
            T2 = np.array([T2_1, T2_2])
            T1 = np.array([T1_1, T1_2])
            E_out = np.zeros(len(R1))
            for idx in range(len(R1)):
                b = float(acquisition_scheme.bvalues[idx])
                if b < 1e3:  # b0 measurement
                    E_out[idx] = 1.0
                    continue

                D1_eff = float(R1[idx]) / max(b, 1.0)
                D2_eff = float(R2[idx]) / max(b, 1.0)

                d_arr = acquisition_scheme.delta
                D_arr = acquisition_scheme.Delta
                d = float(d_arr[idx]) if hasattr(d_arr, '__len__') else float(d_arr)
                De = float(D_arr[idx]) if hasattr(D_arr, '__len__') else float(D_arr)

                if TM_ is not None:
                    # PGSTE pathway. Convention: TE is the echo time (2*delta + TM
                    # in the zero-width-pulse limit); the transverse encoding is two
                    # lobes of delta each (2*delta total) and TM is the longitudinal
                    # storage window. Read the second encoding lobe from the geometry
                    # directly -- dt6 = delta -- rather than back-computing a
                    # transverse time from TE (delta and TM are unambiguous).
                    tm = (float(TM_[idx]) if hasattr(TM_, '__len__')
                          else float(TM_))
                    dt6 = d
                    E_out[idx] = float(exchange.stimulated_echo_signal(
                        K, M0, np.array([D1_eff, D2_eff]), T2, T1, b, d, tm, dt6, tau_90))
                else:
                    # SE pathway
                    te = (float(TE_[idx]) if TE_ is not None and hasattr(TE_, '__len__')
                          else float(TE_) if TE_ is not None
                          else 2 * De)
                    t180 = te / 2.0 - tau_exc / 2.0
                    dt1 = max(t180 - tau_exc, 0.0)
                    dt2 = max(te - t180 - tau_180, 0.0)
                    E_out[idx] = float(exchange.spin_echo_signal(
                        K, M0, np.array([D1_eff, D2_eff]), T2, T1, b, dt1, dt2, tau_exc, tau_180))

            return E_out

        else:
            # Fast path: no relaxation, instantaneous RF → scalar eigenvalue formula
            return exchange.karger(R1, R2, kappa, f, exchange.diffusion_time(delta, Delta))

    # ------------------------------------------------------------------
    # Rotational harmonics (GL quadrature)
    # ------------------------------------------------------------------

    def rotational_harmonics_representation(self, acquisition_scheme, **kwargs):
        r"""Numerical rotational harmonics via 32-point Gauss-Legendre quadrature.

        Computes RH coefficients c_l (l = 0, 2, …, l_max) per shell:

            c_l = 2√(π(2l+1)) · ∫₀¹ E(b, u) P_l(u) du

        where u = cosθ is the angle between gradient and fibre (fixed at z).
        Both sub-models are evaluated at GL orientation nodes.  Any
        ``parameter_links`` (e.g. tortuosity) are resolved before evaluation.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme
        **kwargs : parameter values

        Returns
        -------
        rh_array : ndarray, shape (N_dwi_shells, N_rh_coef)
        """
        from scipy.special import legendre as legendre_poly

        kwargs = self.add_linked_parameters_to_parameters(kwargs)

        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        if rh_scheme.shell_delta is None or rh_scheme.shell_Delta is None:
            raise ValueError(
                "rotational_harmonics_representation requires a PGSE scheme "
                "with shell_delta and shell_Delta."
            )

        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        n_shells     = len(list(rh_scheme.shell_sh_orders))
        rh_array     = np.zeros((n_shells, max_sh_order // 2 + 1))

        f     = float(kwargs['f'])
        kappa = float(kwargs['kappa'])

        # GL nodes on [0,1]; fibre fixed at z, gradient sweeps elevation θ
        n_quad = 32
        xi, omega = np.polynomial.legendre.leggauss(n_quad)
        u_nodes = (xi + 1.0) / 2.0          # cos θ ∈ [0, 1]
        w_nodes = omega / 2.0                # Jacobian

        sin_u  = np.sqrt(np.maximum(1.0 - u_nodes ** 2, 0.0))
        G_dirs = np.column_stack([sin_u, np.zeros(n_quad), u_nodes])

        # Sub-model kwargs with mu fixed to z-axis
        rh_kwargs = dict(kwargs)
        rh_kwargs['mu'] = np.array([0., 0.])

        kwargs_intra = self._sub_kwargs(self.model_intra, rh_kwargs)
        kwargs_extra = self._sub_kwargs(self.model_extra, rh_kwargs)

        for i, (shell_index, sh_order) in enumerate(
                rh_scheme.shell_sh_orders.items()):
            b       = float(rh_scheme.bvalues[i * rh_scheme.Nsamples])
            delta_v = float(rh_scheme.shell_delta[shell_index])
            Delta_v = float(rh_scheme.shell_Delta[shell_index])

            node_scheme = _GLNodeScheme(b, delta_v, Delta_v, G_dirs)

            R1_u = compartment_exponent(self.model_intra, node_scheme, kwargs_intra)
            R2_u = compartment_exponent(self.model_extra, node_scheme, kwargs_extra)
            t_d  = exchange.diffusion_time(delta_v, Delta_v)

            E_u = exchange.karger(R1_u, R2_u, kappa, f, t_d)

            for l_half in range(sh_order // 2 + 1):
                l     = 2 * l_half
                P_l   = legendre_poly(l)(u_nodes)
                integ = np.dot(w_nodes, E_u * P_l)
                rh_array[i, l_half] = (
                    2.0 * np.sqrt(np.pi * (2 * l + 1)) * integ
                )

        return rh_array


class X1NEXI(ModelProperties, IsotropicSignalModelProperties):
    r"""Published NEXI: spherical mean of exchanging Stick and isotropic Ball.

    The four required parameters are ``f``, ``D_i``, ``D_e`` (m²/s) and
    ``t_ex`` (s), with detailed balance k_ie=(1-f)/t_ex, k_ei=f/t_ex.
    A 64-point Gauss-Legendre integral averages the *coupled* orientation
    kernels, rather than exchanging already averaged compartment signals.
    There is no fitted orientation or tortuosity link.

    Wrap in MultiCompartmentSphericalMeanModel for fitting. As with the
    general Karger PGSE path, t=Delta-delta/3 approximates finite pulses;
    Eqs. (3)--(6) are the narrow-pulse model, not a waveform propagator.
    Gaussian exponents are formed directly, without a compartment-signal floor.

    Jelescu et al. 2022, NeuroImage 256, 119277.
    doi:10.1016/j.neuroimage.2022.119277
    """

    _model_type = 'CompartmentModel'
    _required_acquisition_parameters = ['bvalues', 'delta', 'Delta']
    _parameter_ranges = {
        'f': (0.01, 0.99), 'D_i': (0.1, 3.), 'D_e': (0.1, 3.),
        't_ex': (1., 500.),
    }
    _parameter_scales = {
        'f': 1., 'D_i': DIFFUSIVITY_SCALING, 'D_e': DIFFUSIVITY_SCALING,
        't_ex': 1e-3,
    }
    _parameter_types = {p: 'normal' for p in ('f', 'D_i', 'D_e', 't_ex')}
    _citations = {
        'definition': [
            {'key': 'Jelescu2022', 'authors': 'Jelescu IO et al.',
             'title': 'Neurite Exchange Imaging (NEXI): A minimal model of '
                      'diffusion in gray matter with inter-compartment water exchange',
             'journal': 'NeuroImage', 'year': 2022,
             'doi': '10.1016/j.neuroimage.2022.119277'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'narrow_pulse', 'name': 'Narrow-pulse PGSE approximation',
         'condition_human': 'Uses t=Delta-delta/3 in the constant-q Karger '
                            'kernel; finite-pulse exchange is approximate.',
         'severity': 'warning', 'source_key': 'Jelescu2022'},
    ]

    def __init__(self):
        from .cylinder_models import C1Stick
        from .gaussian_models import G1Ball

        self.exchange_model = X0GeneralizedKarger(C1Stick(), G1Ball())
        nodes, weights = np.polynomial.legendre.leggauss(64)
        self.cos_theta = (nodes + 1.) / 2.
        self.quadrature_weights = weights / 2.

    def __call__(self, acquisition_scheme, **kwargs):
        f, di, de, t_ex = (float(kwargs[p]) for p in ('f', 'D_i', 'D_e', 't_ex'))
        if not np.isfinite(f) or not 0 < f < 1:
            raise ValueError('f must be strictly between zero and one')
        if not np.isfinite(di) or not np.isfinite(de) or di < 0 or de < 0:
            raise ValueError('D_i and D_e must be finite and nonnegative')
        if np.isnan(t_ex) or t_ex <= 0:
            raise ValueError('t_ex must be positive (seconds)')
        if acquisition_scheme.delta is None or acquisition_scheme.Delta is None:
            raise ValueError('NEXI requires PGSE delta and Delta')
        b, delta, big_delta = np.broadcast_arrays(
            acquisition_scheme.bvalues, acquisition_scheme.delta, acquisition_scheme.Delta)
        directions = np.column_stack([
            np.sqrt(1. - self.cos_theta**2), np.zeros(64), self.cos_theta])
        parameters = {
            'C1Stick_1_lambda_par': di, 'G1Ball_1_lambda_iso': de,
            'mu': [0., 0.], 'f': f, 'kappa': (1. - f) / t_ex,
        }
        signal = np.empty(b.shape)
        for index in np.ndindex(b.shape):
            nodes = _GLNodeScheme(b[index], delta[index], big_delta[index], directions)
            signal[index] = self.exchange_model(nodes, **parameters) @ self.quadrature_weights
        return signal


def _double_encodings(acquisition_scheme):
    """The ``(sequence, rows)`` of a scheme's filter-exchange sequences -- stimulated echoes whose encoding before the
    store refocuses on its own (:func:`dmipy_replay.sequences.fexi_bvalues` reads them) -- or ``[]``. Their signal is
    not the narrow-pulse Karger form of one diffusion time: the filter, the stored mixing time and the detection each
    act in turn."""
    protocol = getattr(acquisition_scheme, "protocol", None)
    if protocol is None:
        return []
    from dmipy_replay.sequences import fexi_bvalues
    out = []
    for seq, rows in zip(protocol, protocol.rows):
        if not seq.stimulated_echo:
            continue
        try:
            fexi_bvalues(seq)
        except ValueError:
            continue
        out.append((seq, rows))
    if out and len(out) != len(protocol):
        raise ValueError("a scheme mixes filter-exchange sequences with others; the Karger model evaluates one kind "
                         "per scheme: split it into two schemes")
    return out


def _waveform_signal(acquisition_scheme, double, R1, R2, f, kappa, T2, T1, inf):
    """The two-pool signal of each row of each filter-exchange sequence: :func:`dmipy_analytic.exchange.karger_waveform`
    along the row's effective gradient, the store-to-recall span stored (``T1``), the rest transverse (``T2``), from
    the equilibrium ``(f, 1 - f)``. A pool's diffusivity along the row's direction is its dephasing exponent over the
    row's b-value, exact for a Gaussian pool under a colinear encoding (the two blocks share their direction) and the
    effective-medium reading for a restricted one. The stimulated echo's pathway amplitude is common to every row of a
    sequence and left out, so the unweighted row of a sequence is its relaxation alone (1 without relaxation)."""
    K = exchange.exchange_matrix(kappa, f)
    M0 = np.array([f, 1.0 - f])
    T2 = np.where(T2 < inf, T2, np.inf)
    T1 = np.where(T1 < inf, T1, np.inf)
    b = np.asarray(acquisition_scheme.bvalues, dtype=float)
    E = np.empty(b.shape[0])
    for seq, rows in double:
        stored = ~np.asarray(seq.rf.transverse_gate(seq.n_t, seq.dt), dtype=bool)
        G = np.asarray(seq.G_eff, dtype=float)
        for k, m in enumerate(rows):
            D = np.where(b[m] > 0, np.array([R1[m], R2[m]]) / max(b[m], 1e-300), 0.0)
            E[m] = exchange.karger_waveform(G[k], seq.dt, D, K, M0, stored=stored, T2=T2, T1=T1)
    return E
