# -*- coding: utf-8 -*-
"""Two-compartment water exchange models for diffusion MRI.

Implements the generalised Karger exchange framework. NEXI (Neurite
Exchange Imaging) is the Stick+Zeppelin+tortuosity special case, built via
X0GeneralizedKarger (see reference_models.nexi()).

Architecture
------------
X0GeneralizedKarger  — wraps any two CompartmentModel instances with Karger
                       exchange.  Accepts parameter_links in the same format
                       as MultiCompartmentModel and SD1WatsonDistributed.

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
from scipy.linalg import expm as _expm

from ..core.signal_model_properties import AnisotropicSignalModelProperties
from ..distributions.distribute_models import DistributedModel

__all__ = [
    'X0GeneralizedKarger',
]

DIFFUSIVITY_SCALING = 1e-9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _karger_formula(R1, R2, kappa, f, t_d):
    """Karger exchange signal from effective dephasing exponents.

    Parameters
    ----------
    R1, R2 : ndarray, shape (N,)
        Effective dephasing exponents -log(E1), -log(E2) for each measurement.
    kappa  : float  — intra→extra exchange rate (s⁻¹)
    f      : float  — intra volume fraction
    t_d    : ndarray or float — effective diffusion time (s)

    Returns
    -------
    E : ndarray, shape (N,)
    """
    EPS = 1e-10
    kee = kappa * f / (1.0 - f)   # back-rate (detailed balance)
    kt = kappa * t_d               # (N,) or scalar
    keet = kee * t_d

    Tr = R1 + R2 + kt + keet
    Det = R1 * R2 + R1 * keet + R2 * kt

    disc_sq = np.maximum(Tr ** 2 - 4.0 * Det, 0.0)
    disc = np.sqrt(disc_sq)

    lam_plus = (Tr + disc) / 2.0
    lam_minus = (Tr - disc) / 2.0

    safe_disc = np.where(disc > EPS, disc, np.ones_like(disc))
    safe_keet = keet if np.ndim(keet) == 0 else np.where(
        keet > EPS, keet, np.ones_like(keet))

    sp_full = f * (R1 - lam_minus) * (lam_minus - R2) / (safe_keet * safe_disc)

    # κ=0 limit: differentiate from initial rate → σ₊ = (f·R1+(1-f)·R2−λ-)/disc
    # This reduces to (1-f) when R2>R1, f when R1>R2 (correct mixture weights).
    sp_kappa0 = (f * R1 + (1.0 - f) * R2 - lam_minus) / safe_disc
    sigma_plus = np.where(
        disc <= EPS,
        0.5,                                         # degenerate: E = exp(-λ) anyway
        np.where(np.asarray(keet) <= EPS, sp_kappa0, sp_full),
    )

    return sigma_plus * np.exp(-lam_plus) + (1.0 - sigma_plus) * np.exp(-lam_minus)


# ---------------------------------------------------------------------------
# Matrix-exponential Kärger propagators (eq:karger_se_finite, eq:karger_ste_finite)
# ---------------------------------------------------------------------------

def _build_K(kappa, f):
    """2×2 exchange-rate matrix (detailed-balance: kee = kappa * f / (1-f))."""
    kee = kappa * f / (1.0 - f)
    return np.array([[-kappa, kee],
                     [kappa, -kee]], dtype=float)


def _karger_propagator_se(D1, D2, T2_1, T2_2, T1_1, T1_2, kappa, f,
                          B1, B2, dt1, dt2, tau_exc, tau_180):
    """Finite-RF SE Kärger propagator — eq:karger_se_finite.

    Returns the magnetisation vector M(TE) given M0 = [f, 1-f].
    Signal = sum(M_TE).

    Parameters
    ----------
    D1, D2 : float — effective diffusion coefficients for each compartment (m²/s)
    T2_1, T2_2 : float — T2 relaxation times (s); use 1e10 for no relaxation
    T1_1, T1_2 : float — T1 relaxation times (s); use 1e10 for no relaxation
    kappa : float — intra→extra exchange rate (s⁻¹)
    f : float — intra volume fraction
    B1, B2 : float — partial b-values for the two free-precession intervals (s/m²)
    dt1, dt2 : float — durations of the two free-precession intervals (s)
    tau_exc, tau_180 : float — RF pulse durations (s); 0 for instantaneous RF

    Returns
    -------
    M_TE : ndarray, shape (2,)
    """
    K   = _build_K(kappa, f)
    RT2 = np.diag([1.0 / T2_1, 1.0 / T2_2])
    RT1 = np.diag([1.0 / T1_1, 1.0 / T1_2])
    RD  = np.diag([D1, D2])
    # During RF pulses, both T1 and T2 contribute (mixed relaxation)
    R12 = (2.0 / np.pi) * (RT2 + RT1)
    M0  = np.array([f, 1.0 - f])

    # Four factors in chronological order (rightmost applied first):
    # 1. excitation pulse    2. first free precession
    # 3. 180° refocus pulse  4. second free precession
    P_exc  = _expm((K - R12) * tau_exc)
    P_fp1  = _expm((K - RT2) * dt1 - B1 * RD)
    P_180  = _expm((K - R12) * tau_180)
    P_fp2  = _expm((K - RT2) * dt2 - B2 * RD)

    M_TE = P_fp2 @ P_180 @ P_fp1 @ P_exc @ M0
    return M_TE


def _karger_propagator_ste(D1, D2, T2_1, T2_2, T1_1, T1_2, kappa, f,
                           B1, B2, delta, TM, tau_90, dt6):
    """Finite-RF STE Kärger propagator — eq:karger_ste_finite.

    Parameters
    ----------
    D1, D2 : float — effective diffusion coefficients (m²/s)
    T2_1, T2_2, T1_1, T1_2 : float — relaxation times (s); 1e10 = no relaxation
    kappa : float — exchange rate (s⁻¹)
    f : float — intra volume fraction
    B1, B2 : float — partial b-values for the two encoding intervals (s/m²)
    delta : float — encoding duration (s)
    TM : float — mixing time (s)
    tau_90 : float — 90° RF pulse duration (s); 0 for instantaneous
    dt6 : float — second encoding lobe duration (s); = delta in the zero-width-pulse
        limit (transverse encoding total = 2*delta)

    Returns
    -------
    M_TE : ndarray, shape (2,)
        Includes the 0.5 STE pathway factor.
    """
    K   = _build_K(kappa, f)
    RT2 = np.diag([1.0 / T2_1, 1.0 / T2_2])
    RT1 = np.diag([1.0 / T1_1, 1.0 / T1_2])
    RD  = np.diag([D1, D2])
    R12 = (2.0 / np.pi) * (RT2 + RT1)
    M0  = np.array([f, 1.0 - f])

    # Six factors in chronological order:
    P_exc   = _expm((K - R12) * tau_90)
    P_enc1  = _expm((K - RT2) * delta - B1 * RD)
    P_store = _expm((K - R12) * tau_90)
    P_mix   = _expm((K - RT1) * TM)
    P_rec   = _expm((K - R12) * tau_90)
    P_enc2  = _expm((K - RT2) * dt6 - B2 * RD)

    M_TE = 0.5 * P_enc2 @ P_rec @ P_mix @ P_store @ P_enc1 @ P_exc @ M0
    return M_TE


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
        self.tau = float(Delta) - float(delta) / 3.0
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

    For Gaussian compartments (Ball, Stick, Zeppelin) R_i is linear in b and
    the formula is exact.  For restricted compartments (Cylinder, Sphere) it
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

        Uses the finite-RF SE (eq:karger_se_finite) or STE (eq:karger_ste_finite)
        propagator when timing parameters are available.  Falls back gracefully to
        the scalar eigenvalue formula (_karger_formula) via matrix expm at kappa=0.

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
        E1 = self.model_intra(acquisition_scheme,
                              **self._sub_kwargs(self.model_intra, kwargs))
        E2 = self.model_extra(acquisition_scheme,
                              **self._sub_kwargs(self.model_extra, kwargs))

        EPS = 1e-10
        R1 = -np.log(np.maximum(E1, EPS))   # effective dephasing exponent
        R2 = -np.log(np.maximum(E2, EPS))

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
                    B1 = b / 2.0
                    B2 = b / 2.0
                    M_TE = _karger_propagator_ste(
                        D1_eff, D2_eff, T2_1, T2_2, T1_1, T1_2,
                        kappa, f, B1, B2, d, tm, tau_90, dt6)
                else:
                    # SE pathway
                    te = (float(TE_[idx]) if TE_ is not None and hasattr(TE_, '__len__')
                          else float(TE_) if TE_ is not None
                          else 2 * De)
                    t180 = te / 2.0 - tau_exc / 2.0
                    dt1 = max(t180 - tau_exc, 0.0)
                    dt2 = max(te - t180 - tau_180, 0.0)
                    B1 = b / 2.0
                    B2 = b / 2.0
                    M_TE = _karger_propagator_se(
                        D1_eff, D2_eff, T2_1, T2_2, T1_1, T1_2,
                        kappa, f, B1, B2, dt1, dt2, tau_exc, tau_180)

                E_out[idx] = float(np.sum(M_TE))
            return E_out

        else:
            # Fast path: no relaxation, instantaneous RF → scalar eigenvalue formula
            t_d = Delta - delta / 3.0
            return _karger_formula(R1, R2, kappa, f, t_d)

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

            E1_u = self.model_intra(node_scheme, **kwargs_intra)
            E2_u = self.model_extra(node_scheme, **kwargs_extra)

            EPS  = 1e-10
            R1_u = -np.log(np.maximum(E1_u, EPS))
            R2_u = -np.log(np.maximum(E2_u, EPS))
            t_d  = Delta_v - delta_v / 3.0

            E_u = _karger_formula(R1_u, R2_u, kappa, f, t_d)

            for l_half in range(sh_order // 2 + 1):
                l     = 2 * l_half
                P_l   = legendre_poly(l)(u_nodes)
                integ = np.dot(w_nodes, E_u * P_l)
                rh_array[i, l_half] = (
                    2.0 * np.sqrt(np.pi * (2 * l + 1)) * integ
                )

        return rh_array
