# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division

import numpy as np
from scipy.special import erf

from ..utils import utils
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import (
    IsotropicSignalModelProperties, AnisotropicSignalModelProperties)
from dipy.utils.optpkg import optional_package

numba, have_numba, _ = optional_package("numba")

DIFFUSIVITY_SCALING = 1e-9
A_SCALING = 1e-12

__all__ = [
    'G1Ball',
    'G2Zeppelin',
    'G3TemporalZeppelin'
]


class G1Ball(ModelProperties, IsotropicSignalModelProperties):
    r""" The Ball model - an isotropic Tensor with one diffusivity.

    Parameters
    ----------
    lambda_iso : float,
        isotropic diffusivity in m^2/s.
    """
    _citations = {
        'definition': [
            {'key': 'behrens2003', 'authors': 'Behrens TEJ, Woolrich MW, Jenkinson M, et al.',
             'title': 'Characterization and propagation of uncertainty in diffusion-weighted MR imaging',
             'journal': 'Magnetic Resonance in Medicine',
             'year': 2003, 'doi': '10.1002/mrm.10609'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'gaussian_diffusion', 'name': 'Gaussian diffusion',
         'condition_human': 'Signal is exactly exp(-b*D); valid for isotropic free diffusion',
         'severity': 'info',
         'source_key': 'behrens2003'}
    ]
    _required_acquisition_parameters = ['bvalues']

    _parameter_ranges = {
        'lambda_iso': (.1, 3),
    }
    _parameter_scales = {
        'lambda_iso': DIFFUSIVITY_SCALING,
    }
    _parameter_types = {
        'lambda_iso': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        use_jax : bool, optional
            If True and JAX is available, evaluate using the JAX backend.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        if use_jax:
            from ..jax.jax_compat import scheme_to_jax
            from ..jax.signal_models_jax import g1ball_signal
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            return np.array(g1ball_signal(scheme_jax['bvalues'],
                                          float(lambda_iso)))
        bvals = acquisition_scheme.bvalues
        E_ball = np.exp(-bvals * lambda_iso)
        return E_ball


class G2Zeppelin(ModelProperties, AnisotropicSignalModelProperties):
    r""" The Zeppelin model - an axially symmetric Tensor - typically used
    for extra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    lambda_perp : float,
        perpendicular diffusivity in m^2/s.

    Returns
    -------
    E_zeppelin : float or array, shape(N),
        signal attenuation.
    """
    _citations = {
        'definition': [
            {'key': 'panagiotaki2012', 'authors': 'Panagiotaki E, Schneider T, Siow B, Hall MG, Lythgoe MF, Alexander DC',
             'title': 'Compartment models of the diffusion MR signal in brain white matter: a taxonomy and comparison',
             'journal': 'NeuroImage',
             'year': 2012, 'doi': '10.1016/j.neuroimage.2012.01.032'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'gaussian_diffusion', 'name': 'Gaussian diffusion',
         'condition_human': 'Signal is Gaussian (mono-exponential in b). Deviations from Gaussian behavior increase with b-value due to restriction effects in the extra-neurite space.',
         'severity': 'warning',
         'source_key': 'panagiotaki2012'}
    ]
    _required_acquisition_parameters = ['bvalues', 'gradient_directions']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'lambda_perp': (.1, 3),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_perp': DIFFUSIVITY_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'lambda_perp': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        use_jax : bool, optional
            If True and JAX is available, evaluate using the JAX backend.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)
        if use_jax:
            from ..jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
            from ..jax.signal_models_jax import g2zeppelin_signal
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            mu_cart = unitsphere2cart_1d_jax(jnp.array(mu))
            return np.array(g2zeppelin_signal(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                mu_cart, float(lambda_par), float(lambda_perp)))
        mu_cart = utils.unitsphere2cart_1d(mu)
        # B-tensor path: E = exp(-(λ_perp Tr(B) + (λ_par - λ_perp) uᵀBu))
        # Correct for PGSE (rank-1 B = b·n⊗n) and tensor-valued encoding (STE, etc.)
        B = acquisition_scheme.btensor()                           # (n_m, 3, 3)
        b_trace = np.trace(B, axis1=1, axis2=2)                   # (n_m,)
        u_dot_B_u = np.einsum('i,mij,j->m', mu_cart, B, mu_cart)  # (n_m,)
        E_zeppelin = np.exp(
            -(lambda_perp * b_trace + (lambda_par - lambda_perp) * u_dot_B_u)
        )
        return E_zeppelin

    def rotational_harmonics_representation(self, acquisition_scheme, **kwargs):
        r"""Analytical RH coefficients for the Zeppelin kernel.

        Replaces the 10-point angular-sampling approximation with the exact
        formula using the Gaussian zonal harmonics recurrence.
        """
        from ..utils.sh_analytical import gaussian_kernel_rh
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)

        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        # rh_scheme.bvalues repeats each shell's b-value Nsamples times.
        # Per-shell b-value = bvalues[i * Nsamples] for the i-th DWI shell.
        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        n_shells = len(list(rh_scheme.shell_sh_orders))
        rh_array = np.zeros((n_shells, max_sh_order // 2 + 1))

        for i, (shell_index, sh_order) in enumerate(
                rh_scheme.shell_sh_orders.items()):
            b = float(rh_scheme.bvalues[i * rh_scheme.Nsamples])
            rh = gaussian_kernel_rh(b, lambda_par, lambda_perp,
                                    sh_order=max_sh_order)
            rh_array[i, :sh_order // 2 + 1] = rh[:sh_order // 2 + 1]
        return rh_array

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model for every acquisition shell.
        """
        bvals = acquisition_scheme.shell_bvalues[
            ~acquisition_scheme.shell_b0_mask]

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)

        from ..utils.sh_analytical import gaussian_J_l
        E_mean = np.ones_like(acquisition_scheme.shell_bvalues)
        kappa_per_shell = -bvals * (lambda_par - lambda_perp)
        J0_per_shell = np.array(
            [gaussian_J_l(float(k), l_max=0)[0] for k in kappa_per_shell])
        E_mean[~acquisition_scheme.shell_b0_mask] = (
            np.exp(-bvals * lambda_perp) * J0_per_shell / 2.0)
        return E_mean

    def signal_lm(self, acquisition_scheme, **kwargs):
        """SH coefficients E_lm of the Zeppelin signal as a function of fiber direction.

        For each measurement m, E_lm[m] are the Tournier real SH coefficients
        (l_max=8) of E(n̂) = exp(−n̂ᵀ D n̂ · B[m]), analytically computed from
        the eigendecomposition of the traceless B-tensor.

        Requires acquisition_scheme with stored waveform (_G).

        Parameters
        ----------
        acquisition_scheme : AcquisitionScheme
            Must have _G stored (use AcquisitionScheme.from_waveform()).
        **kwargs : lambda_par, lambda_perp overrides.

        Returns
        -------
        E_lm : ndarray, shape (n_m, 45), float64
        """
        if not hasattr(acquisition_scheme, '_G') or acquisition_scheme._G is None:
            raise ValueError(
                "signal_lm() requires AcquisitionScheme with stored waveform "
                "(_G). Use AcquisitionScheme.from_waveform()."
            )
        from ..utils.sh_analytical import gaussian_signal_lm as _gsl
        lambda_par = float(kwargs.get('lambda_par', self.lambda_par))
        lambda_perp = float(kwargs.get('lambda_perp', self.lambda_perp))

        B = acquisition_scheme.btensor()          # (n_m, 3, 3)
        n_m = B.shape[0]
        l_max = 8
        n_coef = (l_max + 1) * (l_max + 2) // 2
        E_lm = np.zeros((n_m, n_coef), dtype=np.float64)

        for m in range(n_m):
            Bm = B[m]
            trB = float(np.trace(Bm))
            M_l2 = (lambda_par - lambda_perp) * (Bm - trB / 3.0 * np.eye(3))
            pre_factor = np.exp(-trB * (lambda_par + 2.0 * lambda_perp) / 3.0)
            E_lm[m] = _gsl(M_l2, pre_factor, l_max=l_max)

        return E_lm


class G3TemporalZeppelin(ModelProperties, AnisotropicSignalModelProperties):
    r"""
    The temporal Zeppelin model - an axially symmetric Tensor - typically
    used to describe extra-axonal diffusion. The G3TemporalZeppelin differs
    from G2Zeppelin in that it has a time-dependent perpendicular parameter
    "A", which describe extra-axonal diffusion hindrance due to axon packing,
    and that lambda_perp is instead called lambda_inf, as it describes the
    perpendicular diffusivity when diffusion time is infinite.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    lambda_inf : float,
        bulk diffusivity constant 10^9 m^2/s.
    A: float,
        characteristic coefficient in 10^12 m^2

    Returns
    -------
    E_zeppelin : float or array, shape(N),
        signal attenuation.
    """
    _citations = {
        'definition': [
            {'key': 'burcaw2015', 'authors': 'Burcaw LM, Fieremans E, Novikov DS',
             'title': 'Mesoscopic structure of neuronal tracts from time-dependent diffusion',
             'journal': 'NeuroImage',
             'year': 2015, 'doi': '10.1016/j.neuroimage.2015.06.061'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'time_dependent_diffusion', 'name': 'Time-dependent extra-axonal diffusion',
         'condition_human': 'Requires Delta and delta; captures structural disorder via log(Delta/delta) term',
         'severity': 'info',
         'source_key': 'burcaw2015'}
    ]
    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions', 'delta', 'Delta']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'lambda_inf': (.1, 3),
        'A': (0, 10),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_inf': DIFFUSIVITY_SCALING,
        'A': A_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'lambda_inf': 'normal',
        'A': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None, lambda_inf=None, A=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_inf = lambda_inf
        self.A = A

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta

        if delta is None or Delta is None:
            raise ValueError(
                "G3TemporalZeppelin requires PGSE timing parameters delta and "
                "Delta to compute the time-dependent perpendicular diffusivity "
                "D_perp(delta, Delta) = lambda_inf + A*(ln(Delta/delta)+3/2)/"
                "(Delta-delta/3).  The provided acquisition scheme has "
                "delta=None or Delta=None.  G3TemporalZeppelin cannot be used "
                "with arbitrary waveform schemes; use G2Zeppelin instead."
            )

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        mu = kwargs.get('mu', self.mu)
        mu_cart = utils.unitsphere2cart_1d(mu)

        # Time-dependent perpendicular diffusivity (vectorised over measurements)
        restricted_term = (
            A * (np.log(Delta / delta) + 3. / 2.) / (Delta - delta / 3.)
        )
        D_perp = lambda_inf + restricted_term   # (n_m,)

        # B-tensor path: E = exp(-(D_perp Tr(B) + (λ_par - D_perp) uᵀBu))
        # Works for PGSE (rank-1 B) and tensor-valued encoding (STE, etc.)
        # D_perp(δ,Δ) is computed from PGSE timing which must be present.
        B = acquisition_scheme.btensor()                             # (n_m, 3, 3)
        b_trace = np.trace(B, axis1=1, axis2=2)                     # (n_m,)
        u_dot_B_u = np.einsum('i,mij,j->m', mu_cart, B, mu_cart)    # (n_m,)
        E_zeppelin = np.exp(
            -(D_perp * b_trace + (lambda_par - D_perp) * u_dot_B_u)
        )
        return E_zeppelin

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Restricted Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Restricted Zeppelin model for every
            acquisition shell.
        """
        bvals = acquisition_scheme.shell_bvalues[
            ~acquisition_scheme.shell_b0_mask]
        delta = acquisition_scheme.shell_delta[
            ~acquisition_scheme.shell_b0_mask]
        Delta = acquisition_scheme.shell_Delta[
            ~acquisition_scheme.shell_b0_mask]
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        E_mean = np.ones_like(acquisition_scheme.shell_bvalues)

        restricted_term = (
            A * (np.log(Delta / delta) + 3 / 2.) / (Delta - delta / 3.)
        )
        lambda_perp = lambda_inf + restricted_term
        from ..utils.sh_analytical import gaussian_J_l
        kappa_per_shell = -bvals * (lambda_par - lambda_perp)
        J0_per_shell = np.array(
            [gaussian_J_l(float(k), l_max=0)[0] for k in kappa_per_shell])
        E_mean[~acquisition_scheme.shell_b0_mask] = (
            np.exp(-bvals * lambda_perp) * J0_per_shell / 2.0)
        return E_mean

    def rotational_harmonics_representation(self, acquisition_scheme, **kwargs):
        r"""Analytical RH coefficients for the TemporalZeppelin kernel.

        Replaces the 10-point angular-sampling approximation with the exact
        formula.  Uses the per-shell δ/Δ to compute λ_⊥(δ,Δ) = λ_inf + A·f(δ,Δ).
        """
        from ..utils.sh_analytical import gaussian_kernel_rh
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)

        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        n_shells = len(list(rh_scheme.shell_sh_orders))
        rh_array = np.zeros((n_shells, max_sh_order // 2 + 1))

        for i, (shell_index, sh_order) in enumerate(
                rh_scheme.shell_sh_orders.items()):
            b = float(rh_scheme.bvalues[i * rh_scheme.Nsamples])
            delta = float(rh_scheme.shell_delta[shell_index])
            Delta = float(rh_scheme.shell_Delta[shell_index])
            lambda_perp = float(np.asarray(
                lambda_inf
                + A * (np.log(Delta / delta) + 1.5) / (Delta - delta / 3.0)
            ).flat[0])
            rh = gaussian_kernel_rh(b, lambda_par, lambda_perp,
                                    sh_order=max_sh_order)
            rh_array[i, :sh_order // 2 + 1] = rh[:sh_order // 2 + 1]
        return rh_array


def _attenuation_zeppelin(bvals, lambda_par, lambda_perp, n, mu):
    "Signal attenuation for Zeppelin model."
    mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
    magnitude_parallel = np.dot(n, mu)
    proj = np.dot(mu_perpendicular_plane, n.T)
    magnitude_perpendicular = np.sqrt(
        proj[0] ** 2 + proj[1] ** 2 + proj[2] ** 2)
    E_zeppelin = np.exp(-bvals *
                        (lambda_par * magnitude_parallel ** 2 +
                         lambda_perp * magnitude_perpendicular ** 2)
                        )
    return E_zeppelin


if have_numba:
    _attenuation_zeppelin = numba.njit()(_attenuation_zeppelin)
