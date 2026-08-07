
import numpy as np
from ..core.modeling_framework import ModelProperties
from ..core.constants import CONSTANTS
from ._restricted_matrix import (
    matrix_restricted_signal, matrix_restricted_batch, project_fixed_direction, pgse_waveform)

from ..core.constants import DIAMETER_SCALING


__all__ = [
    'P3PlaneCallaghanApproximation',
    'P5PlaneMatrixMethod'
]


class P2PlaneStejskalTannerApproximation(ModelProperties):
    r""" Stejskal-Tanner approximation of diffusion between two infinitely
    large parallel planes. Assumes short-gradient pulse (SGP) approximation
    (pulse length towards zero) and the long diffusion time limit (pulse
    separation towards infinity).
    """
    _citations = {
        'definition': [
            {'key': 'balinov1993', 'authors': 'Balinov B, Jonsson B, Linse P, Soderman O',
             'title': 'The NMR self-diffusion method applied to restricted diffusion. Simulation of echo attenuation from molecules in spheres and between planes',
             'journal': 'Journal of Magnetic Resonance, Series A',
             'year': 1993, 'doi': '10.1006/jmra.1993.1184'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'SGP', 'name': 'Short Gradient Pulse approximation',
         'condition_human': 'delta << Delta (gradient pulse duration much shorter than diffusion time)',
         'severity': 'warning',
         'source_key': 'balinov1993'},
        {'id': 'long_diffusion_time', 'name': 'Long diffusion time limit',
         'condition_human': 'Delta >> R^2/D (diffusion time long enough for complete restriction)',
         'severity': 'warning',
         'source_key': 'balinov1993'},
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes the restricting membrane is perfectly impermeable. No water exchange across the boundary. In reality, biological membranes have finite permeability (membrane permeability coefficient k_m ~ 1e-6 to 1e-4 m/s; see kappa_membrane in biophysical_constants).',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = ['qvalues']
    _parameter_ranges = {
        'diameter': (1e-2, 20)
    }

    _parameter_scales = {
        'diameter': DIAMETER_SCALING
    }

    _parameter_types = {
        'diameter': 'plane'
    }
    _model_type = 'NMRModel'

    def __init__(self, diameter=None):
        self.diameter = diameter

    def plane_attenuation(self, q, diameter):
        "Equation 6 in Balinov et al. (1993)."
        q_argument = 2 * np.pi * q * diameter
        return 2 * (1 - np.cos(q_argument)) / q_argument ** 2

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenuation.

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
        q = acquisition_scheme.qvalues
        diameter = kwargs.get('diameter', self.diameter)

        E_plane = np.ones_like(q)
        q_nonzero = q > 0
        E_plane[q_nonzero] = self.plane_attenuation(
            q[q_nonzero], diameter
        )
        return E_plane


class P3PlaneCallaghanApproximation(ModelProperties):
    r"""
    The Callaghan model of diffusion between two parallel infinite plates.

    Parameters
    ----------
    diameter : float
        Distance between the two plates in meters.
    diffusion_constant : float,
        The diffusion constant of the water particles between the two planes.
        The default value is the approximate diffusivity of water inside axons
        as 1.7e-9 m^2/s.
    number_of_roots : integer,
        The number of roots for the Callaghan approximation.
    """
    _citations = {
        'definition': [
            {'key': 'callaghan1995', 'authors': 'Callaghan PT',
             'title': 'Pulsed-gradient spin-echo NMR for planar, cylindrical, and spherical pores under conditions of wall relaxation',
             'journal': 'Journal of Magnetic Resonance, Series A',
             'year': 1995, 'doi': '10.1006/jmra.1995.1055'}
        ],
        'default_parameters': {
            'diffusion_constant': {'value': 1.7e-9, 'unit': 'm^2/s',
                                   'source_key': 'beaulieu2002'},
        },
    }
    _validity_constraints = [
        {'id': 'SGP', 'name': 'Short Gradient Pulse approximation',
         'condition_human': 'delta << Delta (gradient pulse duration much shorter than diffusion time)',
         'severity': 'warning',
         'source_key': 'callaghan1995'},
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes the restricting membrane is perfectly impermeable. No water exchange across the boundary. In reality, biological membranes have finite permeability (membrane permeability coefficient k_m ~ 1e-6 to 1e-4 m/s; see kappa_membrane in biophysical_constants).',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = ['qvalues', 'tau']

    _parameter_ranges = {
        'diameter': (1e-2, 20)
    }

    _parameter_scales = {
        'diameter': DIAMETER_SCALING
    }

    _parameter_types = {
        'diameter': 'plane'
    }
    _model_type = 'NMRModel'

    def __init__(
        self,
        diameter=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
        number_of_roots=40,
    ):

        self.diameter = diameter
        self.Dintra = diffusion_constant
        self.xi = np.arange(number_of_roots) * np.pi
        self.zeta = np.arange(number_of_roots) * np.pi + np.pi / 2.0

    def plane_attenuation(self, q, tau, diameter):
        """Implements the finite time Callaghan model for planes."""
        radius = diameter / 2.0
        q_argument = 2 * np.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = np.zeros_like(q)
        for n in range(len(self.xi)):
            xi_n = self.xi[n]
            xi_n2 = self.xi[n] ** 2

            if xi_n == 0.:
                div = 1.
            else:
                div = np.sin(2 * xi_n) / 2 * xi_n

            update = (
                2 * np.exp(-xi_n2 * self.Dintra * tau / radius ** 2) /
                (1 + div) *
                (q_argument * np.sin(q_argument) * np.cos(xi_n) - xi_n *
                 np.cos(q_argument) * np.sin(xi_n)) ** 2 /
                (q_argument_2 - xi_n2) ** 2
            )

            update[~np.isfinite(update)] = 0.

            res += update

        for m in range(len(self.zeta)):
            zeta_m = self.zeta[m]
            zeta_m2 = self.zeta[m] ** 2

            update = (
                2 * np.exp(-zeta_m2 * self.Dintra * tau / radius ** 2) /
                (1 - np.sin(2 * zeta_m) / (2 * zeta_m)) *
                (q_argument * np.cos(q_argument) * np.sin(zeta_m) - zeta_m *
                 np.sin(q_argument) * np.cos(zeta_m)) ** 2 /
                (q_argument_2 - zeta_m2) ** 2
            )

            update[~np.isfinite(update)] = 0.
            res += update
        return res

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenuation.

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
        q = acquisition_scheme.qvalues
        tau = acquisition_scheme.tau
        diameter = kwargs.get('diameter', self.diameter)

        E_plane = np.ones_like(q)
        q_nonzero = q > 0
        E_plane[q_nonzero] = self.plane_attenuation(
            q[q_nonzero], tau[q_nonzero], diameter
        )
        return E_plane


class P5PlaneMatrixMethod(ModelProperties):
    r"""Exact matrix / multiple-correlation-function model of diffusion between two parallel planes
    (a 1-D slab) for an *arbitrary* gradient waveform (Callaghan 1997; Grebenkov 2007). The
    Stejskal-Tanner (P2) and Callaghan (P3) plane models are its short-pulse limits. Like P2/P3 this is
    a 1-D building block: the gradient is taken along the restricting axis. Consumes the stored waveform
    ``_G`` when present; otherwise reconstructs a rectangular PGSE waveform from the scalar timing.

    Parameters
    ----------
    diameter : float
        slab thickness in meters.
    n_modes : int, optional
        Laplacian eigenmodes kept (accuracy knob; default 24). Increase for very high q or short
        gradient pulses (the plane needs more modes than the cylinder/sphere at matched q).

    See ``examples/02_signal_models/exact_matrix_method.md`` for a worked comparison against the
    Gaussian-phase and short-pulse plane models.
    """
    _citations = {
        'definition': [
            {'key': 'callaghan1997', 'authors': 'Callaghan PT',
             'title': 'A simple matrix formalism for spin echo analysis of restricted diffusion under generalized gradient waveforms',
             'journal': 'Journal of Magnetic Resonance', 'year': 1997,
             'doi': '10.1006/jmre.1997.1233'},
            {'key': 'grebenkov2007', 'authors': 'Grebenkov DS',
             'title': 'NMR survey of reflected Brownian motion',
             'journal': 'Reviews of Modern Physics', 'year': 2007,
             'doi': '10.1103/RevModPhys.79.1077'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes the restricting walls are perfectly impermeable (no exchange).',
         'severity': 'info'},
        {'id': 'mode_truncation', 'name': 'Eigenmode truncation',
         'condition_human': 'Exact up to the number of eigenmodes kept (n_modes); increase for very high q or short pulses.',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = [
        'gradient_strengths', 'delta', 'Delta']
    _supports_waveform_scheme = True

    _parameter_ranges = {'diameter': (1e-2, 20)}
    _parameter_scales = {'diameter': DIAMETER_SCALING}
    _parameter_types = {'diameter': 'plane'}
    _model_type = 'NMRModel'

    def __init__(self, diameter=None,
                 diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
                 n_modes=24):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = CONSTANTS['water_gyromagnetic_ratio']
        self.n_modes = int(n_modes)

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        """Signal of the exact-matrix slab for the acquisition's gradient waveform.
        ``use_jax=True`` uses the differentiable GPU batch path (needs the ``[jax]`` extra and a stored
        waveform ``_G``; falls back to NumPy for scalar-timing schemes)."""
        L = kwargs.get('diameter', self.diameter)          # slab thickness
        D = self.diffusion_constant
        gamma = self.gyromagnetic_ratio
        _G = getattr(acquisition_scheme, '_G', None)
        if _G is not None:
            dt = float(acquisition_scheme._dt)
            _Gd = np.asarray(_G, np.float64)
            g_axes = np.stack([project_fixed_direction(_Gd[m], dt)[1] for m in range(_Gd.shape[0])])
            return matrix_restricted_batch(
                'plane', g_axes, dt, D, L, gamma, self.n_modes, use_jax=use_jax)
        n_meas = len(acquisition_scheme.gradient_strengths)
        E = np.ones(n_meas)
        for m in range(n_meas):
            g = acquisition_scheme.gradient_strengths[m]
            if g == 0:
                continue
            G_m, dt = pgse_waveform(
                g, acquisition_scheme.delta[m], acquisition_scheme.Delta[m],
                acquisition_scheme.gradient_directions[m])
            _, g_signed = project_fixed_direction(G_m, dt)
            E[m] = matrix_restricted_signal('plane', g_signed, dt, D, L, gamma, self.n_modes)
        return E
