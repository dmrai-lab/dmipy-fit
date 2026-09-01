
import numpy as np
from ..core.modeling_framework import ModelProperties
from ..core.constants import CONSTANTS
from ._restricted_matrix import (
    matrix_restricted_signal, matrix_restricted_batch, project_fixed_direction, pgse_waveform)

from ..core.constants import DIAMETER_SCALING


__all__ = [
    'P3PlaneCallaghanApproximation',
    'P4PlaneGaussianPhaseApproximation',
    'P5PlaneMatrixMethod',
    'P6MonteCarloReplayPlane'
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


class P4PlaneGaussianPhaseApproximation(ModelProperties):
    r"""Gaussian-phase (finite gradient pulse) model of restricted diffusion between two parallel plates
    (a 1-D slab). The plane analogue of the Van Gelderen cylinder (``C4``) and Murday-Cotts sphere
    (``S4``) Gaussian-phase models, and the finite-pulse companion of the short-pulse plane models
    ``P2``/``P3``. The exact matrix-method plane ``P5PlaneMatrixMethod`` is its infinite-order limit;
    ``P4`` is accurate where the Gaussian-phase approximation holds (``delta >> L^2/D``, many wall
    collisions per pulse) and departs at high b / short pulses, where ``P5`` should be used.

    Restriction is across the slab (gradient taken along the plate normal). For a slab of thickness
    ``diameter`` = L the second-cumulant phase is

        ln E = - gamma^2 G^2  sum_{k odd}  |<k|x|0>|^2  I(D (k pi / L)^2),

    with the position-coupling weight ``|<k|x|0>|^2 = 8 L^2 / (k pi)^4`` (only odd modes couple the
    uniform ground state to an excited cosine mode) and ``I`` the Van Gelderen PGSE temporal bracket.

    Parameters
    ----------
    diameter : float
        slab thickness (plate separation) in meters.
    diffusion_constant : float
        intrinsic diffusivity in m^2/s (default: water in axons).
    number_of_modes : int, optional
        number of odd eigenmodes summed (converges as 1/k^4; default 100).
    """
    _citations = {
        'definition': [
            {'key': 'balinov1993', 'authors': 'Balinov B, Jonsson B, Linse P, Soderman O',
             'title': 'The NMR self-diffusion method applied to restricted diffusion. Simulation of echo attenuation from molecules in spheres and between planes',
             'journal': 'Journal of Magnetic Resonance, Series A', 'year': 1993,
             'doi': '10.1006/jmra.1993.1184'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'GPA', 'name': 'Gaussian Phase Approximation',
         'condition_human': 'delta >> L^2/D (many wall collisions during the gradient pulse); use P5PlaneMatrixMethod outside this regime.',
         'severity': 'warning', 'source_key': 'balinov1993'},
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes perfectly impermeable plates (no exchange across the walls).',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = ['gradient_strengths', 'delta', 'Delta']

    _parameter_ranges = {'diameter': (1e-2, 20)}
    _parameter_scales = {'diameter': DIAMETER_SCALING}
    _parameter_types = {'diameter': 'plane'}
    _model_type = 'NMRModel'

    def __init__(self, diameter=None,
                 diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
                 number_of_modes=100):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = CONSTANTS['water_gyromagnetic_ratio']
        self._k_odd = np.arange(1, 2 * int(number_of_modes), 2)   # odd modes 1,3,5,...

    def plane_attenuation(self, gradient_strength, delta, Delta, diameter):
        "Plane Gaussian-phase signal attenuation (gradient along the plate normal)."
        D = self.diffusion_constant
        L = diameter
        k = self._k_odd
        lam_D = D * (k * np.pi / L) ** 2                          # eigenmode decay rates (1/s)
        bracket = (2 * lam_D * delta - 2 + 2 * np.exp(-lam_D * delta)
                   + 2 * np.exp(-lam_D * Delta)
                   - np.exp(-lam_D * (Delta - delta))
                   - np.exp(-lam_D * (Delta + delta))) / lam_D ** 2
        weight = 8.0 * L ** 2 / (k * np.pi) ** 4                  # |<k|x|0>|^2, odd modes
        phi2 = (self.gyromagnetic_ratio * gradient_strength) ** 2 * np.sum(weight * bracket)
        return np.exp(-phi2)

    def __call__(self, acquisition_scheme, **kwargs):
        r"""Signal attenuation for the acquisition scheme (gradient taken along the plate normal)."""
        diameter = kwargs.get('diameter', self.diameter)
        g = acquisition_scheme.gradient_strengths
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta
        E = np.ones_like(np.asarray(g, float))
        nz = np.asarray(g, float) > 0
        idx = np.where(nz)[0]
        E[idx] = [self.plane_attenuation(g[i], delta[i], Delta[i], diameter) for i in idx]
        return E


class P6MonteCarloReplayPlane(ModelProperties):
    r"""Parallel-plate (slab) compartment whose signal is computed by *replaying* a Monte-Carlo
    reference walk (Substrate Commons canonical replay-pack dataset). The 1-D sibling of
    ``C6MonteCarloReplayCylinder`` / ``S6MonteCarloReplaySphere``: exact to the Monte-Carlo floor for any
    gradient waveform, with no short-pulse or Gaussian-phase assumption. Diffusion is restricted across
    the slab (along the plane normal ``mu``) and free in the plane.

    Forward evaluation uses the compiled-scheme engine (``_replay_fit``): the waveform is projected onto
    the pack's DCT temporal basis once, then each replay is a single matmul. ``diameter`` (slab thickness)
    interpolates across the dataset's per-diameter packs; the normal ``mu`` rotates the waveform onto the
    pack's canonical restricted axis (x). Surface relaxivity is exact (the boundary-local-time replay knob,
    optionally coherence-gated), not the analytic ``exp(-TE rho S/V)`` tag-on.

    Parameters
    ----------
    mu : array, shape(2)      angles [theta, phi] of the plane NORMAL on the sphere.
    diameter : float          slab thickness in meters.
    diffusivity : float       intrinsic (fixed) diffusivity of the reference dataset, m^2/s (default 2e-9).
    dataset_dir : str, optional   directory holding the replay-pack dataset (else $SUBSTRATE_COMMONS_DATA).
    """
    _citations = {
        'definition': [
            {'key': 'substrate_commons', 'authors': 'Substrate Commons',
             'title': 'Canonical restricted-shape Monte-Carlo replay-pack reference dataset',
             'journal': 'https://substrate-commons.github.io', 'year': 2026, 'doi': ''},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'The reference walk is between perfectly impermeable parallel plates (no exchange).',
         'severity': 'info'},
        {'id': 'fixed_diffusivity', 'name': 'Fixed intrinsic diffusivity',
         'condition_human': 'Intrinsic diffusivity D0 is fixed by the reference dataset (walk-time), not fitted; pick the dataset whose D0 matches your regime.',
         'severity': 'warning', 'source_key': 'substrate_commons'},
        {'id': 'waveform_grid', 'name': 'Waveform save-grid resolution',
         'condition_human': 'The acquisition waveform is resampled onto the pack save grid; pulses finer than that grid are not resolved.',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = ['gradient_directions']
    _supports_waveform_scheme = True
    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'diameter': (0.1, 20),
    }
    _parameter_scales = {'mu': np.r_[1., 1.], 'diameter': DIAMETER_SCALING}
    _parameter_types = {'mu': 'orientation', 'diameter': 'plane'}
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, diameter=None, diffusivity=2.0e-9, dataset_dir=None):
        self.mu = mu
        self.diameter = diameter
        self.diffusivity = float(diffusivity)
        self.dataset_dir = dataset_dir

    def __call__(self, acquisition_scheme, **kwargs):
        from ..data.mc_replay import (load_replay_family, resample_waveform_to_grid, orient_to_x)
        from ..utils import utils
        diameter = kwargs.get('diameter', self.diameter)
        mu = kwargs.get('mu', self.mu)
        rho = float(kwargs.get('surface_relaxivity', 0.0) or 0.0)
        chi_hat = kwargs.get('chi_hat', None)
        G = getattr(acquisition_scheme, '_G', None)
        if G is None:
            raise ValueError(
                "P6MonteCarloReplayPlane needs a waveform-first AcquisitionScheme "
                "(build with AcquisitionScheme.from_pgse/from_waveform) — no ._G on this scheme.")
        fam = load_replay_family('plane', self.diffusivity, dataset_dir=self.dataset_dir)
        G_pack = resample_waveform_to_grid(np.asarray(G), float(acquisition_scheme._dt), fam.n_t, fam.dt)
        R = orient_to_x(utils.unitsphere2cart_1d(np.asarray(mu, float)))   # normal -> restricted axis x
        G_rot = G_pack @ R.T
        rho_over_D = (rho / fam.diffusivity) if rho else 0.0
        return fam.replay_interpolated(G_rot, float(diameter), rho_over_D=rho_over_D, chi_hat=chi_hat)
