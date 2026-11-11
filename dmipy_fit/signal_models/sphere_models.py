from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import IsotropicSignalModelProperties
from ..core.constants import CONSTANTS
from dmipy_analytic.constants import GAMMA
from dmipy_analytic.restricted import sphere
import numpy as np

from ..core.constants import DIAMETER_SCALING
from ._matrix_adapter import matrix_signal, matrix_signal_batch, pgse_waveform

__all__ = [
    'S1Dot',
    'S2SphereStejskalTannerApproximation',
    'S3SphereCallaghanApproximation',
    'S4SphereGaussianPhaseApproximation',
    'S5SphereMatrixMethod',
    'S6MonteCarloReplaySphere',
]


class S1Dot(ModelProperties, IsotropicSignalModelProperties):
    r"""
    The Dot model - a non-diffusing compartment.
    It has no parameters and returns 1 no matter the input.
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
        {'id': 'fully_restricted', 'name': 'Fully restricted compartment',
         'condition_human': 'Models completely trapped spins (zero displacement); valid for very small compartments',
         'severity': 'info',
         'source_key': 'panagiotaki2012'}
    ]
    _required_acquisition_parameters = []

    _parameter_ranges = {
    }
    _parameter_scales = {
    }
    _parameter_types = {
    }
    _model_type = 'CompartmentModel'

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenation.

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
        E_dot = np.ones(acquisition_scheme.number_of_measurements)
        return E_dot


class S2SphereStejskalTannerApproximation(
        ModelProperties, IsotropicSignalModelProperties):
    r"""
    The Stejskal Tanner signal approximation of a sphere model. It assumes
    that pulse length is infinitessimally small and diffusion time large enough
    so that the diffusion is completely restricted. Only depends on q-value.

    Parameters
    ----------
    diameter : float,
        sphere diameter in meters.
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
        'diameter': (1e-2, 20),
    }
    _parameter_scales = {
        'diameter': DIAMETER_SCALING,
    }
    _parameter_types = {
        'diameter': 'sphere',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, diameter=None):
        self.diameter = diameter

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        r'''
        Calculates the signal attenation.

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
        diameter = kwargs.get('diameter', self.diameter)
        if use_jax:
            from dmipy_analytic.jax.restricted import sphere as jsphere
            from ..jax.jax_compat import scheme_to_jax
            scheme_jax = scheme_to_jax(acquisition_scheme)
            return np.array(jsphere.sgp_long_time(scheme_jax['qvalues'], float(diameter)))
        q = acquisition_scheme.qvalues
        return sphere.sgp_long_time(q, diameter)  # only q>0 attenuate


class S3SphereCallaghanApproximation(
        ModelProperties, IsotropicSignalModelProperties):
    r"""
    The Callaghan model of diffusion inside a sphere.

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
        'diameter': (1e-2, 20),
    }

    _parameter_scales = {
        'diameter': DIAMETER_SCALING,
    }

    _parameter_types = {
        'diameter': 'sphere',
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        diameter=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
        number_of_roots=25,
        number_of_functions=16,
    ):

        self.diameter = diameter
        self.Dintra = diffusion_constant
        self.number_of_roots = number_of_roots
        self.number_of_functions = number_of_functions

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

        return sphere.sgp(q, tau, diameter, self.Dintra, self.number_of_roots, self.number_of_functions)


class S4SphereGaussianPhaseApproximation(
        ModelProperties, IsotropicSignalModelProperties):
    r"""
    The gaussian phase approximation for diffusion inside a sphere.
    It is dependent on gradient strength, pulse separation and pulse length.

    ``required_precision = 'float64'``: in the SANDI spherical-mean fit
    (``tests/test_sm_recovery.py``) ``solver='jax'`` in float32 recovers the
    sphere diameter to 0.898 of the truth where float64 recovers 0.997.
    """
    required_precision = 'float64'
    _citations = {
        'definition': [
            {'key': 'balinov1993', 'authors': 'Balinov B, Jonsson B, Linse P, Soderman O',
             'title': 'The NMR self-diffusion method applied to restricted diffusion. Simulation of echo attenuation from molecules in spheres and between planes',
             'journal': 'Journal of Magnetic Resonance, Series A',
             'year': 1993, 'doi': '10.1006/jmra.1993.1184'}
        ],
        'default_parameters': {
            'diffusion_constant': {'value': 1.7e-9, 'unit': 'm^2/s',
                                   'source_key': 'beaulieu2002'},
        },
    }
    _validity_constraints = [
        {'id': 'GPA', 'name': 'Gaussian Phase Approximation',
         'condition_human': 'delta >> R^2/D (many wall collisions during gradient pulse)',
         'severity': 'warning',
         'source_key': 'balinov1993'},
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes the restricting membrane is perfectly impermeable. No water exchange across the boundary. In reality, biological membranes have finite permeability (membrane permeability coefficient k_m ~ 1e-6 to 1e-4 m/s; see kappa_membrane in biophysical_constants).',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = ['gradient_strengths', 'delta', 'Delta']
    # Can evaluate a stored gradient waveform (_G) directly, including
    # multidimensional b-tensor schemes, via the per-component GPA path.
    _supports_waveform_scheme = True

    _parameter_ranges = {
        'diameter': (1e-2, 20),
    }
    _parameter_scales = {
        'diameter': DIAMETER_SCALING,
    }
    _parameter_types = {
        'diameter': 'sphere',
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self, diameter=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    ):
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = GAMMA
        self.diameter = diameter

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenation.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme or AcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
            If the scheme has oscillation_frequency set (OGSE), the OGSE
            analytical path (Xu 2009) is used per measurement. Otherwise
            the standard PGSE Murday-Cotts GPA path is used.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        diameter = kwargs.get('diameter', self.diameter)
        D = self.diffusion_constant
        R = diameter / 2.0

        osc_freq = getattr(acquisition_scheme, 'oscillation_frequency', None)

        n_m = acquisition_scheme.number_of_measurements
        E_sphere = np.ones(n_m, dtype=float)

        has_waveform = getattr(acquisition_scheme, '_G', None) is not None

        # Route to the general-waveform path when the scheme contains any genuine
        # multidimensional (non-rank-1) b-tensor measurement -- pure STE/PTE, or
        # a mixed concatenation of PGSE + b-tensor where a scalar
        # gradient_strengths is ill-defined on the b-tensor block. Detected from
        # the b-tensor rank so mixed schemes are handled per measurement rather
        # than silently treated as b0. Pure colinear PGSE/OGSE schemes keep
        # their (faster) analytic/OGSE branches below.
        if has_waveform:
            Bten = acquisition_scheme.btensor()          # (n_m, 3, 3), s/m^2
            evals = np.linalg.eigvalsh(Bten)             # ascending
            bmax = max(float(evals[:, -1].max()), 1.0)
            multidim = evals[:, -2] > 1e-3 * bmax
        else:
            multidim = np.zeros(n_m, dtype=bool)

        def _general_waveform_signal():
            # Isotropic sphere factorises: the GPA phase variance sums over the
            # three Cartesian gradient components, so E = prod_i E_1D(G_i(t)).
            # This reduces exactly to the colinear (single-projection) result
            # when a measurement's waveform is 1-D, so it is applied uniformly to
            # b-tensor, mixed, and colinear-without-scalar-strength schemes.
            dt = float(acquisition_scheme._dt)
            E = np.ones(n_m, dtype=float)
            for m in range(n_m):
                G_vec = np.asarray(acquisition_scheme._G[m], dtype=np.float64)
                if not np.any(G_vec):        # b0 / no gradient
                    continue
                Em = 1.0
                for i in range(3):
                    Em *= sphere.gaussian_phase_waveform(
                        G_vec[:, i], dt, diameter, D)
                E[m] = Em
            return E

        if np.any(multidim):
            # Any genuine multidimensional (b-tensor) measurement present -- pure
            # STE/PTE or a mixed PGSE+b-tensor concatenation -- so evaluate every
            # measurement from its waveform rather than a (b-tensor-undefined)
            # scalar gradient_strengths, which would silently read as b0.
            return _general_waveform_signal()

        if osc_freq is None or np.all(osc_freq == 0):
            # ----------------------------------------------------------------
            # Pure PGSE path — original Murday-Cotts GPA (unchanged)
            # ----------------------------------------------------------------
            g = acquisition_scheme.gradient_strengths
            if g is None and has_waveform:
                # Colinear waveform with no scalar gradient_strengths (e.g. a
                # general from_waveform scheme): use the per-component path,
                # which reduces to the analytic PGSE result for a 1-D waveform.
                return _general_waveform_signal()
            delta = acquisition_scheme.delta
            Delta = acquisition_scheme.Delta
            g_nonzero = g > 0
            E_sphere[g_nonzero] = sphere.gaussian_phase(
                g[g_nonzero], delta[g_nonzero], Delta[g_nonzero], diameter, D,
                gamma=self.gyromagnetic_ratio)
        else:
            # ----------------------------------------------------------------
            # Mixed or pure OGSE path — dispatch per measurement
            # ----------------------------------------------------------------
            g = acquisition_scheme.gradient_strengths
            delta = acquisition_scheme.delta
            Delta = acquisition_scheme.Delta

            for m in range(n_m):
                freq_m = osc_freq[m]
                if freq_m == 0:
                    # PGSE measurement inside a mixed scheme
                    if g is not None and g[m] > 0:
                        E_sphere[m] = float(np.asarray(sphere.gaussian_phase(
                            np.atleast_1d(g[m]),
                            float(delta[m]),
                            float(Delta[m]),
                            diameter,
                            D,
                            gamma=self.gyromagnetic_ratio,
                        )).reshape(-1)[0])
                else:
                    # OGSE measurement
                    t_r = float(acquisition_scheme.gradient_rise_time[m])
                    G_m = float(acquisition_scheme.gradient_strengths[m])
                    if G_m == 0:
                        E_sphere[m] = 1.0
                        continue
                    if t_r > 0:
                        # Trapezoidal OGSE: use numerical Stepisnik path from
                        # the signed scalar projection of the stored waveform.
                        # Project onto gradient direction (unit vector).
                        G_t_vec = acquisition_scheme._G[m].astype(
                            np.float64)  # (n_t, 3)
                        g_dir = acquisition_scheme.gradient_directions[m]
                        # Scalar projection: G_t[n] = G_vec[n] · g_hat
                        # For isotropic sphere, any direction gives same result.
                        # Use the gradient direction stored on the scheme.
                        g_dir_norm = np.linalg.norm(g_dir)
                        if g_dir_norm > 0:
                            g_hat = g_dir / g_dir_norm
                        else:
                            g_hat = np.r_[1., 0., 0.]
                        G_t = G_t_vec @ g_hat  # (n_t,) signed projection
                        E_sphere[m] = sphere.gaussian_phase_waveform(
                            G_t, acquisition_scheme._dt, diameter, D)
                    else:
                        # Pure cosine OGSE: numerical path from stored waveform
                        # (same as trapezoidal but uses pure cosine G(t))
                        G_t_vec = acquisition_scheme._G[m].astype(
                            np.float64)  # (n_t, 3)
                        g_dir = acquisition_scheme.gradient_directions[m]
                        g_dir_norm = np.linalg.norm(g_dir)
                        if g_dir_norm > 0:
                            g_hat = g_dir / g_dir_norm
                        else:
                            g_hat = np.r_[1., 0., 0.]
                        G_t = G_t_vec @ g_hat  # (n_t,) signed projection
                        E_sphere[m] = sphere.gaussian_phase_waveform(
                            G_t, acquisition_scheme._dt, diameter, D)

        return E_sphere


class S5SphereMatrixMethod(ModelProperties, IsotropicSignalModelProperties):
    r"""Sphere whose restricted signal is the Bloch-Torrey eigenmode propagator of
    ``dmipy_analytic.restricted.matrix_method`` (Callaghan 1997; Grebenkov 2007): exact for a piecewise-constant
    gradient waveform (one matrix exponential per run of constant amplitude), a Strang split of
    diffusion and precession otherwise, of which the Gaussian-phase model
    :class:`S4SphereGaussianPhaseApproximation` is the low-b limit. Consumes the stored gradient
    waveform ``_G`` when present (any PGSE / OGSE / fixed-direction waveform); otherwise a rectangular
    PGSE waveform is reconstructed from the scalar timing. Rotating / b-tensor waveforms are projected
    onto their dominant direction (exact only for fixed-direction).

    Parameters
    ----------
    diameter : float
        sphere diameter in meters.
    n_modes : int, optional
        Laplacian eigenmodes kept (accuracy knob; default 16 is converged well below 1e-4 for typical
        b). Increase for very high q or short gradient pulses. Mapped onto
        ``dmipy_analytic.restricted.matrix_method``'s ``(angular, radial)`` pair as
        ``(n_modes, max(4, n_modes // 2 + 2))``.

    See ``examples/02_signal_models/exact_matrix_method.md`` for a worked comparison against the
    Gaussian-phase model.
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
         'condition_human': 'Assumes the restricting membrane is perfectly impermeable (no exchange across the boundary).',
         'severity': 'info'},
        {'id': 'mode_truncation', 'name': 'Eigenmode truncation',
         'condition_human': 'Exact up to the number of eigenmodes kept (n_modes); the default is converged well below 1e-4 for typical b. Increase n_modes for very high q or short pulses.',
         'severity': 'info'},
        {'id': 'fixed_direction_waveform', 'name': 'Fixed-direction waveform',
         'condition_human': 'Exact for fixed-direction encodings (PGSE, standard OGSE). Rotating / b-tensor waveforms are projected onto their dominant direction (approximate).',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = [
        'gradient_directions', 'gradient_strengths', 'delta', 'Delta']
    _supports_waveform_scheme = True

    _parameter_ranges = {'diameter': (1e-2, 20)}
    _parameter_scales = {'diameter': DIAMETER_SCALING}
    _parameter_types = {'diameter': 'sphere'}
    _model_type = 'CompartmentModel'

    def __init__(self, diameter=None,
                 diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
                 n_modes=16):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant
        self.n_modes = int(n_modes)

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        """Signal of the exact-matrix sphere for the acquisition's gradient waveform.
        ``use_jax=True`` uses the differentiable GPU batch path (needs the ``[jax]`` extra and a stored
        waveform ``_G``; falls back to NumPy for scalar-timing schemes)."""
        diameter = kwargs.get('diameter', self.diameter)
        R = diameter / 2.
        D = self.diffusion_constant
        _G = getattr(acquisition_scheme, '_G', None)
        n = acquisition_scheme.gradient_directions
        if _G is not None:
            dt = float(acquisition_scheme._dt)
            g_axes = np.einsum('mtc,mc->mt', np.asarray(_G, np.float64), n)   # isotropic: along grad dir
            return matrix_signal_batch(
                'sphere', g_axes, dt, D, 2 * R, self.n_modes, use_jax=use_jax)
        n_meas = len(acquisition_scheme.gradient_strengths)
        E = np.ones(n_meas)
        for m in range(n_meas):
            g = acquisition_scheme.gradient_strengths[m]
            if g == 0:
                continue
            G_m, dt = pgse_waveform(
                g, acquisition_scheme.delta[m], acquisition_scheme.Delta[m], n[m])
            E[m] = matrix_signal('sphere', G_m @ n[m], dt, D, 2 * R, self.n_modes)
        return E


class S6MonteCarloReplaySphere(ModelProperties, IsotropicSignalModelProperties):
    r"""Restricted-sphere compartment whose signal is computed by *replaying* a Monte-Carlo reference
    walk (Substrate Commons canonical replay-pack dataset) rather than a closed-form approximation.
    Same fit parameter as the analytical spheres — ``diameter`` — but exact (to the Monte-Carlo floor)
    for **any** gradient waveform. The isotropic sibling of ``C6MonteCarloReplayCylinder``; see it for the
    compiled-scheme engine and the exact (coherence-gated) surface-relaxivity replay knob.

    Parameters
    ----------
    diameter : float          sphere diameter in meters.
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
         'condition_human': 'The reference walk is inside a perfectly impermeable sphere (no exchange).',
         'severity': 'info'},
        {'id': 'fixed_diffusivity', 'name': 'Fixed intrinsic diffusivity',
         'condition_human': 'Intrinsic diffusivity D0 is fixed by the reference dataset (walk-time), not fitted; pick the dataset whose D0 matches your regime.',
         'severity': 'warning', 'source_key': 'substrate_commons'},
        {'id': 'waveform_band', 'name': 'Waveform within the stored band',
         'condition_human': 'The acquisition waveform is projected exactly onto the bands the pack stores; gradient structure above the band K / (2 T) of the pack was never stored.',
         'severity': 'info'},
    ]
    _required_acquisition_parameters = ['gradient_directions']
    _supports_waveform_scheme = True
    _parameter_ranges = {'diameter': (0.1, 20)}
    _parameter_scales = {'diameter': DIAMETER_SCALING}
    _parameter_types = {'diameter': 'sphere'}
    _model_type = 'CompartmentModel'

    def __init__(self, diameter=None, diffusivity=2.0e-9, dataset_dir=None):
        self.diameter = diameter
        self.diffusivity = float(diffusivity)
        self.dataset_dir = dataset_dir

    def __call__(self, acquisition_scheme, **kwargs):
        from ..data.mc_replay import load_replay_family, surface_relaxivity_of
        diameter = kwargs.get('diameter', self.diameter)
        rho = surface_relaxivity_of(kwargs)
        chi = kwargs.get('coherence_gate', None)                    # the transverse gate on the acquisition's own grid
        G = getattr(acquisition_scheme, '_G', None)
        if G is None:
            raise ValueError(
                "S6MonteCarloReplaySphere needs a waveform-first AcquisitionScheme "
                "(build with AcquisitionScheme.from_pgse/from_waveform) — no ._G on this scheme.")
        fam = load_replay_family('sphere', self.diffusivity, dataset_dir=self.dataset_dir)
        G_pack = np.asarray(G, np.float64)                              # the scheme's own grid: compiled exactly per pack
        rho_over_D = (rho / fam.diffusivity) if rho else 0.0        # isotropic: no orientation rotation
        return fam.replay_interpolated_raw(G_pack, float(acquisition_scheme._dt), float(diameter), rho_over_D=rho_over_D, chi=chi)
