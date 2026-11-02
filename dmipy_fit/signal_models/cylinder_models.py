from __future__ import division

import numpy as np
from dmipy_analytic import gaussian
from dmipy_analytic.constants import GAMMA
from dmipy_analytic.restricted import cylinder

from ..utils import utils
from ..core.constants import CONSTANTS
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import AnisotropicSignalModelProperties
from ._matrix_adapter import matrix_signal, matrix_signal_batch, pgse_waveform
from ..utils.sh_basis import real_sh_tournier, symmetric_sphere

from ..core.constants import DIFFUSIVITY_SCALING, DIAMETER_SCALING, A_SCALING

# ---------------------------------------------------------------------------
# Sphere quadrature grid for SH dispersion integrals.
#
# Uses the symmetric 724-point grid (sh_basis.symmetric_sphere, the 'symmetric724' sphere of dipy) with
# uniform weights w_q = 1/724.  This grid integrates smooth functions to
# better than 0.1% accuracy for the SH orders relevant to the cylinder GPA
# (l <= 4).  The name LEBEDEV_50 is kept for API compatibility; the grid is
# actually a 724-point equal-weight quadrature.
#
# Convention: weights sum to 1, so the spherical integral is approximated as
#     integral f(mu) dOmega  ≈  4*pi * sum_q w_q f(mu_q)
# Projection to SH:
#     E_lm  =  4*pi * sum_q w_q  f(mu_q)  Y_lm(mu_q)
# ---------------------------------------------------------------------------
def _build_sphere_quad():
    """Build the 724-point sphere quadrature grid used for SH dispersion."""
    pts = symmetric_sphere()                          # (724, 3)
    N = pts.shape[0]
    w = np.ones(N, dtype=np.float64) / N              # uniform weights, sum=1
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    return pts, w, theta, phi


_SPHERE_QUAD_PTS, _SPHERE_QUAD_W, _SPHERE_QUAD_THETA, _SPHERE_QUAD_PHI = (
    _build_sphere_quad()
)


__all__ = [
    'C1Stick',
    'C2CylinderStejskalTannerApproximation',
    'C3CylinderCallaghanApproximation',
    'C4CylinderGaussianPhaseApproximation',
    'C5CylinderMatrixMethod',
    'C6MonteCarloReplayCylinder'
]


class C1Stick(ModelProperties, AnisotropicSignalModelProperties):
    r""" The Stick model - a cylinder with zero radius - typically used
    for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
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
        {'id': 'zero_radius', 'name': 'Stick: zero-radius limit',
         'condition_human': 'No restriction effects; valid for R << sqrt(D*delta)',
         'severity': 'info',
         'source_key': 'behrens2003'},
        {'id': 'gaussian_parallel', 'name': 'Gaussian parallel diffusion',
         'condition_human': 'Parallel diffusion is modeled as purely Gaussian (single exponential decay along the fiber axis). Non-Gaussian effects (e.g., from axonal beading, undulation, or finite cylinder length) are not captured.',
         'severity': 'info'},
        {'id': 'single_axon_diffusivity', 'name': 'Single-axon parallel diffusivity',
         'condition_human': 'lambda_par represents the intrinsic parallel diffusivity of a single axon, not the macroscopic apparent diffusivity of a dispersed white matter bundle. Setting it to a macroscopically measured value (which includes orientation dispersion effects) would double-count dispersion when combined with Watson/Bingham distributions.',
         'severity': 'warning'},
    ]

    _required_acquisition_parameters = ['bvalues', 'gradient_directions']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None):
        self.mu = mu
        self.lambda_par = lambda_par

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
        lambda_par_ = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        if use_jax:
            from dmipy_analytic.jax import gaussian as jgaussian
            from ..jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            mu_cart = unitsphere2cart_1d_jax(jnp.array(mu))
            return np.array(jgaussian.stick(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                mu_cart, float(lambda_par_)))
        mu_cart = utils.unitsphere2cart_1d(mu)
        # the b-tensor form is exact for a pulsed gradient (rank-1 B = b n n^T) and for tensor-valued encodings
        return gaussian.stick_btensor(acquisition_scheme.btensor(), mu_cart, lambda_par_)

    def rotational_harmonics_representation(self, acquisition_scheme, **kwargs):
        r"""Analytical RH coefficients for the Stick kernel.

        Stick = Zeppelin with λ_⊥ = 0, so kernel_rh[l//2] = 2π √((2l+1)/(4π)) J_l(−b·λ_∥).
        Replaces the 10-point angular-sampling approximation.
        """
        from ..utils.sh_analytical import gaussian_kernel_rh
        lambda_par = kwargs.get('lambda_par', self.lambda_par)

        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        # rh_scheme.bvalues repeats each shell's b-value Nsamples times.
        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        n_shells = len(list(rh_scheme.shell_sh_orders))
        rh_array = np.zeros((n_shells, max_sh_order // 2 + 1))

        for i, (shell_index, sh_order) in enumerate(
                rh_scheme.shell_sh_orders.items()):
            b = float(rh_scheme.bvalues[i * rh_scheme.Nsamples])
            rh = gaussian_kernel_rh(b, lambda_par, 0.0, sh_order=max_sh_order)
            rh_array[i, :sh_order // 2 + 1] = rh[:sh_order // 2 + 1]
        return rh_array

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Stick model.

        Parameters
        ----------
        acquisition_scheme : PGSEAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : array of size (Nshells)
            spherical mean of the Stick model for every acquisition shell.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        return gaussian.stick_powder_average(acquisition_scheme.shell_bvalues, lambda_par)

    def signal_lm(self, acquisition_scheme, **kwargs):
        """SH coefficients E_lm of the Stick signal as a function of fiber direction.

        For each measurement m, E_lm[m] are the Tournier real SH coefficients
        (l_max=8) of E(n̂) = exp(−λ_par · n̂ᵀ B[m] n̂), analytically computed
        from the eigendecomposition of the traceless B-tensor.

        Requires acquisition_scheme with stored waveform (_G).

        Parameters
        ----------
        acquisition_scheme : AcquisitionScheme
            Must have _G stored (use AcquisitionScheme.from_waveform()).
        **kwargs : lambda_par override.

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
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_par = float(lambda_par)

        B = acquisition_scheme.btensor()          # (n_m, 3, 3)
        n_m = B.shape[0]
        l_max = 8
        n_coef = (l_max + 1) * (l_max + 2) // 2
        E_lm = np.zeros((n_m, n_coef), dtype=np.float64)

        for m in range(n_m):
            Bm = B[m]
            trB = float(np.trace(Bm))
            M_l2 = lambda_par * (Bm - trB / 3.0 * np.eye(3))
            pre_factor = np.exp(-lambda_par * trB / 3.0)
            E_lm[m] = _gsl(M_l2, pre_factor, l_max=l_max)

        return E_lm


class C2CylinderStejskalTannerApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Stejskal-Tanner approximation of the cylinder model with finite
    radius. Assumes that both the short gradient pulse (SGP) approximation
    is met and long diffusion time limit is reached. The perpendicular
    cylinder diffusion therefore only depends on the q-value of the
    acquisition.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    diameter : float,
        cylinder diameter in meters.

    Returns
    -------
    E : array, shape (N,)
        signal attenuation
    """
    _citations = {
        'definition': [
            {'key': 'soderman1995', 'authors': 'Soderman O, Jonsson B',
             'title': 'Restricted diffusion in cylindrical geometry',
             'journal': 'Journal of Magnetic Resonance, Series A',
             'year': 1995, 'doi': '10.1006/jmra.1995.0014'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'SGP', 'name': 'Short Gradient Pulse approximation',
         'condition_human': 'delta << Delta (gradient pulse duration much shorter than diffusion time)',
         'severity': 'warning',
         'source_key': 'soderman1995'},
        {'id': 'long_diffusion_time', 'name': 'Long diffusion time limit',
         'condition_human': 'Delta >> R^2/D (diffusion time long enough for complete restriction)',
         'severity': 'warning',
         'source_key': 'soderman1995'},
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes the restricting membrane is perfectly impermeable. No water exchange across the boundary. In reality, biological membranes have finite permeability (membrane permeability coefficient k_m ~ 1e-6 to 1e-4 m/s; see kappa_membrane in biophysical_constants).',
         'severity': 'info'},
        {'id': 'gaussian_parallel', 'name': 'Gaussian parallel diffusion',
         'condition_human': 'Parallel diffusion is modeled as purely Gaussian (single exponential decay along the fiber axis). Non-Gaussian effects (e.g., from axonal beading, undulation, or finite cylinder length) are not captured.',
         'severity': 'info'},
        {'id': 'single_axon_diffusivity', 'name': 'Single-axon parallel diffusivity',
         'condition_human': 'lambda_par represents the intrinsic parallel diffusivity of a single axon, not the macroscopic apparent diffusivity of a dispersed white matter bundle. Setting it to a macroscopically measured value (which includes orientation dispersion effects) would double-count dispersion when combined with Watson/Bingham distributions.',
         'severity': 'warning'},
    ]

    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions', 'qvalues']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder',
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter

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
        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        if use_jax:
            from ..jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
            from ..jax.signal_models_jax import c2cylinder_signal
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            mu_cart = unitsphere2cart_1d_jax(jnp.array(mu))
            return np.array(c2cylinder_signal(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                scheme_jax['qvalues'], mu_cart,
                float(lambda_par), float(diameter)))
        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        q = acquisition_scheme.qvalues
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = gaussian.stick(bvals, n, mu, lambda_par)
        q_perp = q * magnitude_perpendicular
        E_perpendicular = cylinder.sgp_long_time(q_perp, diameter)  # only q>0 attenuate
        E = E_parallel * E_perpendicular
        return E


class C3CylinderCallaghanApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Callaghan model - a cylinder with finite radius - typically
    used for intra-axonal diffusion. The perpendicular diffusion is modelled
    after Callaghan's solution for the disk. Is dependent on both q-value
    and diffusion time.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.
    diffusion_perpendicular : float,
        the intra-cylindrical, perpenicular diffusivity. By default it is set
        to a typical value for intra-axonal diffusion as 1.7e-9 m^2/s.
    number_of_roots : integer,
        number of roots to use for the Callaghan cylinder model.
    number_of_function : integer,
        number of functions to use for the Callaghan cylinder model.
    """
    _citations = {
        'definition': [
            {'key': 'callaghan1995', 'authors': 'Callaghan PT',
             'title': 'Pulsed-gradient spin-echo NMR for planar, cylindrical, and spherical pores under conditions of wall relaxation',
             'journal': 'Journal of Magnetic Resonance, Series A',
             'year': 1995, 'doi': '10.1006/jmra.1995.1055'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'SGP', 'name': 'Short Gradient Pulse approximation',
         'condition_human': 'delta << Delta (gradient pulse duration much shorter than diffusion time)',
         'severity': 'warning',
         'source_key': 'callaghan1995'},
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes the restricting membrane is perfectly impermeable. No water exchange across the boundary. In reality, biological membranes have finite permeability (membrane permeability coefficient k_m ~ 1e-6 to 1e-4 m/s; see kappa_membrane in biophysical_constants).',
         'severity': 'info'},
        {'id': 'gaussian_parallel', 'name': 'Gaussian parallel diffusion',
         'condition_human': 'Parallel diffusion is modeled as purely Gaussian (single exponential decay along the fiber axis). Non-Gaussian effects (e.g., from axonal beading, undulation, or finite cylinder length) are not captured.',
         'severity': 'info'},
        {'id': 'single_axon_diffusivity', 'name': 'Single-axon parallel diffusivity',
         'condition_human': 'lambda_par represents the intrinsic parallel diffusivity of a single axon, not the macroscopic apparent diffusivity of a dispersed white matter bundle. Setting it to a macroscopically measured value (which includes orientation dispersion effects) would double-count dispersion when combined with Watson/Bingham distributions.',
         'severity': 'warning'},
    ]

    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions', 'qvalues', 'tau']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder',
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant'],
        number_of_roots=20,
        number_of_functions=50,
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.diameter = diameter
        self.number_of_roots = number_of_roots
        self.number_of_functions = number_of_functions

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
        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        if use_jax:
            from ..jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
            from ..jax.signal_models_jax import build_c3cylinder_jax_fn
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            mu_cart = unitsphere2cart_1d_jax(jnp.array(mu))
            fn = build_c3cylinder_jax_fn(self.number_of_roots, self.number_of_functions,
                                         self.diffusion_perpendicular)
            return np.array(fn(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                scheme_jax['qvalues'], scheme_jax['tau'],
                mu_cart, float(lambda_par), float(diameter)))
        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        q = acquisition_scheme.qvalues
        tau = acquisition_scheme.tau
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = gaussian.stick(bvals, n, mu, lambda_par)
        q_perp = q * magnitude_perpendicular
        E_perpendicular = cylinder.sgp(q_perp, tau, diameter, self.diffusion_perpendicular,
                                       self.number_of_roots, self.number_of_functions)
        E = E_parallel * E_perpendicular
        return E


class C4CylinderGaussianPhaseApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Gaussian phase model - a cylinder with finite radius -
    typically used for intra-axonal diffusion. The perpendicular diffusion is
    modelled after Van Gelderen's solution for the disk. It is dependent on
    gradient strength, pulse separation and pulse length.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.
    """
    _citations = {
        'definition': [
            {'key': 'vangelderen1994', 'authors': 'Van Gelderen P, DesPres D, van Zijl PCM, Moonen CTW',
             'title': 'Evaluation of restricted diffusion in cylinders. Phosphocreatine in rabbit leg muscle',
             'journal': 'Journal of Magnetic Resonance, Series B',
             'year': 1994, 'doi': '10.1006/jmrb.1994.1038'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'GPA', 'name': 'Gaussian Phase Approximation',
         'condition_human': 'delta >> R^2/D (many wall collisions during gradient pulse)',
         'severity': 'warning',
         'source_key': 'vangelderen1994'},
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'Assumes the restricting membrane is perfectly impermeable. No water exchange across the boundary. In reality, biological membranes have finite permeability (membrane permeability coefficient k_m ~ 1e-6 to 1e-4 m/s; see kappa_membrane in biophysical_constants).',
         'severity': 'info'},
        {'id': 'gaussian_parallel', 'name': 'Gaussian parallel diffusion',
         'condition_human': 'Parallel diffusion is modeled as purely Gaussian (single exponential decay along the fiber axis). Non-Gaussian effects (e.g., from axonal beading, undulation, or finite cylinder length) are not captured.',
         'severity': 'info'},
        {'id': 'single_axon_diffusivity', 'name': 'Single-axon parallel diffusivity',
         'condition_human': 'lambda_par represents the intrinsic parallel diffusivity of a single axon, not the macroscopic apparent diffusivity of a dispersed white matter bundle. Setting it to a macroscopically measured value (which includes orientation dispersion effects) would double-count dispersion when combined with Watson/Bingham distributions.',
         'severity': 'warning'},
    ]

    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions',
        'gradient_strengths', 'delta', 'Delta']
    # Can evaluate a stored gradient waveform (_G) directly, including rotating
    # / multidimensional b-tensor schemes, via the gamma_lm eigenmode path.
    _supports_waveform_scheme = True

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder',
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant'],
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.gyromagnetic_ratio = GAMMA
        self.diameter = diameter

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        r'''
        Calculates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : AcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
            Three code paths are selected automatically:
            Three code paths are selected automatically:
            (1) Rotating waveform (_G stored with non-colinear directions,
                R < 1 mm): fast-eigenmode Gamma_lm factorisation. Covers
                rotating OGSE, b-tensor encoding, arbitrary rotating waveforms.
            (2) Fixed-direction OGSE (oscillation_frequency > 0): the numerical
                Stepisnik GPA integral of the played waveform.
            (3) PGSE or fixed-direction waveform: Van Gelderen GPA.
        use_jax : bool, optional
            If True and JAX is available, evaluate using the JAX backend
            (PGSE path only; OGSE falls back to NumPy).
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        if use_jax:
            from ..jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
            from ..jax.signal_models_jax import c4cylinder_signal
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            mu_cart = unitsphere2cart_1d_jax(jnp.array(mu))
            return np.array(c4cylinder_signal(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                scheme_jax['gradient_strengths'], scheme_jax['delta'],
                scheme_jax['Delta'], mu_cart, float(lambda_par), float(diameter),
                float(self.diffusion_perpendicular), float(self.gyromagnetic_ratio)))

        D = self.diffusion_perpendicular
        R = diameter / 2.0

        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        mu_cart = utils.unitsphere2cart_1d(mu)

        osc_freq = getattr(acquisition_scheme, 'oscillation_frequency', None)

        # ------------------------------------------------------------------
        # Gamma_lm path: waveform with rotating gradient directions.
        # Triggered when _G is stored AND the gradient direction actually
        # rotates across timesteps (colinear waveforms stay on the existing
        # PGSE/OGSE paths which are more accurate for fixed directions).
        # Guard R < 1e-3 excludes unphysical radii (C_geom ∝ R⁴ overflows).
        # ------------------------------------------------------------------
        _G = getattr(acquisition_scheme, '_G', None)
        if (_G is not None and R < 1e-3
                and _waveform_directions_rotate(_G)):
            gamma_lm = acquisition_scheme.gamma_lm(l_max=4)
            b_par = bvals * np.dot(n, mu_cart) ** 2
            return cylinder.gaussian_phase_gamma_lm(
                gamma_lm, b_par, mu_cart, diameter, D, lambda_par)

        g = acquisition_scheme.gradient_strengths
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta
        mu_perpendicular_plane = np.eye(3) - np.outer(mu_cart, mu_cart)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = gaussian.stick(bvals, n, mu_cart, lambda_par)
        n_m = acquisition_scheme.number_of_measurements
        E_perpendicular = np.ones(n_m, dtype=float)

        if osc_freq is None or np.all(osc_freq == 0):
            # ----------------------------------------------------------------
            # Pure PGSE path: the Van Gelderen Gaussian phase of the perpendicular gradient
            # ----------------------------------------------------------------
            g_perp = g * magnitude_perpendicular
            E_perpendicular = cylinder.gaussian_phase(
                g_perp, delta, Delta, diameter, D, gamma=self.gyromagnetic_ratio)
        else:
            # ----------------------------------------------------------------
            # Mixed or pure OGSE path — dispatch per measurement
            # ----------------------------------------------------------------
            for m in range(n_m):
                G_m = float(g[m])
                G_perp_m = G_m * float(magnitude_perpendicular[m])
                if G_perp_m == 0.0:
                    continue  # E_perpendicular[m] stays 1.0

                freq_m = osc_freq[m]
                if freq_m == 0:
                    # PGSE measurement inside a mixed scheme
                    E_perpendicular[m] = float(np.asarray(
                        cylinder.gaussian_phase(
                            np.atleast_1d(G_perp_m),
                            float(delta[m]),
                            float(Delta[m]),
                            diameter,
                            D,
                            gamma=self.gyromagnetic_ratio,
                        )).reshape(-1)[0])
                else:
                    # OGSE (cosine or trapezoid train, whatever the ramps and the gap around the 180): the
                    # Stepisnik GPA integral of the played effective gradient, projected on the perpendicular
                    # plane. The closed form ``cylinder.gaussian_phase_cosine`` is its contiguous-cosine limit.
                    G_t_vec = acquisition_scheme._G[m].astype(np.float64)          # (n_t, 3)
                    n_perp = np.dot(mu_perpendicular_plane, n[m])                  # the encoding's perpendicular
                    n_perp = n_perp / np.linalg.norm(n_perp)                       # direction (unit)
                    G_t = G_t_vec @ n_perp                                         # (n_t,) signed projection
                    E_perpendicular[m] = cylinder.gaussian_phase_waveform(
                        G_t, acquisition_scheme._dt, diameter, D)

        E = E_parallel * E_perpendicular
        return E

    def signal_from_gamma_lm(self, acquisition_scheme, **kwargs):
        """Evaluate cylinder GPA signal via the fast-eigenmode Gamma_lm path.

        This is the Gamma_lm factorised path:
            phi(mu) = C_geom(R,D) * [2/3 * int|G|^2 dt - (8pi/15) * sum_m Gamma_2m * Y2m(mu)]
                    + b_par * lambda_par
            E = exp(-phi)

        This path requires AcquisitionScheme (with stored waveform _G).
        For validity, the fast-eigenmode condition must hold:
            alpha_1 * T >> 1  where alpha_1 = (mu_1/R)^2 * D, T = waveform duration.

        For R=2um, D=1.7e-9, T=40ms: alpha_1*T = 57.6 >> 1 (valid, error < 1%).
        For R=5um, D=1.7e-9, T=40ms: alpha_1*T = 9.2 (marginal, error ~12%).

        Parameters
        ----------
        acquisition_scheme : AcquisitionScheme
            Must have _G stored (waveform-based scheme).
        **kwargs : model parameter overrides (diameter, lambda_par, mu).

        Returns
        -------
        E : ndarray, shape (n_m,)
            Signal attenuation.

        Raises
        ------
        ValueError
            If acquisition_scheme does not have a stored waveform (_G is None).
        """
        if not hasattr(acquisition_scheme, '_G') or acquisition_scheme._G is None:
            raise ValueError(
                "signal_from_gamma_lm() requires AcquisitionScheme with stored "
                "waveform (_G). Use AcquisitionScheme.from_waveform() or "
                "AcquisitionScheme.from_pgse() to construct the scheme."
            )
        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)

        D = self.diffusion_perpendicular

        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        mu_cart = utils.unitsphere2cart_1d(mu)

        gamma_lm = acquisition_scheme.gamma_lm(l_max=4)
        b_par = bvals * np.dot(n, mu_cart) ** 2
        return cylinder.gaussian_phase_gamma_lm(gamma_lm, b_par, mu_cart, diameter, D, lambda_par)

    def signal_lm(self, acquisition_scheme, **kwargs):
        """Compute SH coefficients E_lm of the per-fiber cylinder GPA signal.

        For each measurement m, E_lm[m] are the real SH coefficients (Tournier
        ordering, l_max=8) of E_m(n̂) = exp(-φ_perp(n̂) - φ_par(n̂)), where n̂
        is the cylinder axis direction integrated over the sphere.

        This is the low-level kernel.  To compute a dispersed signal for any ODF:

            E_lm  = c4.signal_lm(scheme, diameter=d, lambda_par=lp)
            S     = dispersed_signal_from_E_lm(E_lm, odf_lm)

        Separating E_lm from the ODF allows any orientation distribution
        (Watson, Bingham, Gamma, empirical) to be plugged in without re-running
        the cylinder physics, restoring the modular dmipy architecture.

        Parameters
        ----------
        acquisition_scheme : AcquisitionScheme
            Must have _G stored (use AcquisitionScheme.from_waveform()).
        **kwargs : diameter, lambda_par overrides.

        Returns
        -------
        E_lm : (n_m, 45) ndarray, float64
            SH coefficients of exp(-φ(n̂)) per measurement, l_max=8 Tournier.

        Raises
        ------
        ValueError
            If acquisition_scheme does not have a stored waveform (_G).
        """
        if not hasattr(acquisition_scheme, '_G') or acquisition_scheme._G is None:
            raise ValueError(
                "signal_lm() requires AcquisitionScheme with stored waveform "
                "(_G). Use AcquisitionScheme.from_waveform()."
            )
        from ..utils.sh_analytical import _l2sh_to_matrix, gaussian_signal_lm as _gsl

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = float(kwargs.get('lambda_par', self.lambda_par))

        D = self.diffusion_perpendicular

        gamma_lm = acquisition_scheme.gamma_lm(l_max=4)
        # the perpendicular exponent in the harmonics of the axis direction: c0 (n_m,) and c2m (n_m, 5)
        c0_perp, c2m_perp = cylinder.gaussian_phase_exponent_coefficients(gamma_lm, diameter, D)

        B = acquisition_scheme.btensor()              # (n_m, 3, 3)
        n_m = B.shape[0]
        l_max = 8
        n_coef = (l_max + 1) * (l_max + 2) // 2
        E_lm = np.zeros((n_m, n_coef), dtype=np.float64)

        I3 = np.eye(3)
        for m in range(n_m):
            Bm = B[m]
            trB = float(np.trace(Bm))
            # Traceless l=2 phase matrix: perpendicular + parallel contributions
            M_l2 = (_l2sh_to_matrix(c2m_perp[m])
                    + lambda_par * (Bm - trB / 3.0 * I3))
            phi0 = float(c0_perp[m]) + lambda_par * trB / 3.0
            E_lm[m] = _gsl(M_l2, np.exp(-phi0), l_max=l_max)

        return E_lm



def _waveform_directions_rotate(G):
    """Return True if any measurement in G has non-colinear gradient directions.

    Parameters
    ----------
    G : ndarray, shape (n_m, n_t, 3), float32
        Stored gradient waveforms from AcquisitionScheme._G.

    Returns
    -------
    bool
    """
    G_f = G.astype(np.float64)
    mag = np.linalg.norm(G_f, axis=-1, keepdims=True)  # (n_m, n_t, 1)
    G_hat = G_f / np.where(mag > 0, mag, 1.0)          # (n_m, n_t, 3)
    ref = G_hat[:, 0:1, :]                              # (n_m, 1, 3)
    cross = np.cross(G_hat, ref)                        # (n_m, n_t, 3)
    return bool(np.any(np.linalg.norm(cross, axis=-1) > 1e-6))


def E_lm_from_exponent_coeffs(c0, c2m, l_max=4):
    """Compute SH coefficients of exp(-phi(mu)) via sphere quadrature.

    Given per-measurement GPA exponent coefficients:
        phi(mu) = c0[m] + sum_{m'=-2}^{2} c2m[m, m'] * Y_{2,m'}(mu)

    evaluates f(mu) = exp(-phi(mu)) at each quadrature point and projects to
    spherical harmonics up to order l_max.

    Parameters
    ----------
    c0 : (n_m,) ndarray, float64
        Isotropic (l=0) exponent coefficient per measurement.
    c2m : (n_m, 5) ndarray, float64
        Anisotropic (l=2) exponent coefficients [m=-2,-1,0,1,2] per measurement.
    l_max : int, optional
        Maximum SH order.  Default 4.  Must be even.

    Returns
    -------
    E_lm : (n_m, n_lm) ndarray, float64
        SH coefficients of the attenuated signal function exp(-phi), where
        n_lm = (l_max+1)*(l_max+2)//2  (half-shell, real SH, Tournier ordering).

    Notes
    -----
    The quadrature uses the 724-point symmetric sphere with uniform weights.
    Accuracy for smooth cylinder GPA exponentials (l<=4 significant content)
    is better than 0.1% in spherical mean.

    SH convention: Tournier/MRtrix real SHs with Y00 = 1/(2*sqrt(pi)).
    Projection: E_lm = 4*pi * sum_q w_q * exp(-phi(mu_q)) * Y_lm(mu_q)
    """
    n_m = len(c0)
    n_lm = (l_max + 1) * (l_max + 2) // 2

    # SH basis evaluated at quadrature points (724, n_lm)
    Y_Q = real_sh_tournier(l_max, _SPHERE_QUAD_PTS)  # (N_q, n_lm)
    N_q = Y_Q.shape[0]

    # l=2 SH basis at quadrature points: columns 1-5 in the Tournier ordering
    # Tournier ordering: l=0 (1 coeff), l=2 (5 coeffs), ...
    Y2m_Q = Y_Q[:, 1:6]  # (N_q, 5), m=-2,-1,0,1,2

    # Evaluate f(mu_q) = exp(-phi(mu_q)) for each measurement
    # phi(mu_q) = c0[m] + sum_{m'} c2m[m, m'] * Y2m_Q[q, m']
    # Shape: (n_m, N_q)
    phi_Q = c0[:, None] + (c2m[:, None, :] * Y2m_Q[None, :, :]).sum(axis=-1)
    f_Q = np.exp(-phi_Q)  # (n_m, N_q)

    # Project to SH: E_lm[m, lm] = 4*pi * sum_q w_q * f(mu_q) * Y_lm(mu_q)
    # = 4*pi * (f_Q * w_Q[None, :]) @ Y_Q     shape: (n_m, N_q) @ (N_q, n_lm)
    E_lm = 4.0 * np.pi * (f_Q * _SPHERE_QUAD_W[None, :]) @ Y_Q  # (n_m, n_lm)
    return E_lm


def dispersed_signal_from_E_lm(E_lm, odf_lm):
    """Compute dispersed signal from SH coefficients of E and ODF.

    The dispersed signal is the inner product:
        S[m] = integral ODF(mu) * E_m(mu) dOmega
             = sum_{lm'} odf_lm[lm'] * E_lm[m, lm']

    This is the SH inner product theorem: because both ODF and E are expanded
    in the same real orthonormal SH basis, the integral reduces to a dot product
    of their coefficient vectors.

    Parameters
    ----------
    E_lm : (n_m, n_lm) ndarray, float64
        SH coefficients of exp(-phi(mu)) per measurement, from
        E_lm_from_exponent_coeffs().
    odf_lm : (n_lm,) ndarray, float64
        SH coefficients of the orientation distribution function.
        Must satisfy: odf_lm[0] = 1/(2*sqrt(pi)) = Y00 (for any normalized ODF).

    Returns
    -------
    S : (n_m,) ndarray, float64
        Dispersed signal attenuation in [0, 1].

    Notes
    -----
    The SH convention must match E_lm_from_exponent_coeffs() (Tournier ordering).
    For a normalized ODF: odf_lm[0] = 1/sqrt(4*pi) = Y00_value.
    For an isotropic ODF: odf_lm[l>0] = 0, and S = E_lm[:, 0] * Y00.
    """
    n_e = E_lm.shape[1]
    n_o = len(odf_lm)
    if n_o < n_e:
        return E_lm[:, :n_o] @ odf_lm
    if n_o > n_e:
        return E_lm @ odf_lm[:n_e]
    return E_lm @ odf_lm


class C5CylinderMatrixMethod(ModelProperties, AnisotropicSignalModelProperties):
    r"""Finite-radius cylinder whose perpendicular restriction is the Bloch-Torrey eigenmode propagator
    of ``dmipy_analytic.restricted.matrix_method`` (Callaghan 1997; Barzykin 1999; Grebenkov 2007): exact for a
    piecewise-constant gradient waveform (one matrix exponential per run of constant amplitude), a
    Strang split of diffusion and precession otherwise, of which the Gaussian-phase model
    :class:`C4CylinderGaussianPhaseApproximation` is the low-b limit. Parallel diffusion is free
    (Gaussian). The stored gradient waveform ``_G`` is consumed when present (any PGSE / OGSE /
    fixed-direction waveform); otherwise a rectangular PGSE waveform is reconstructed from the scalar
    timing. Rotating / multidimensional b-tensor waveforms are projected onto their dominant direction
    (exact only for fixed-direction encodings).

    Parameters
    ----------
    mu : array, shape(2)
        angles [theta, phi] of the cylinder axis on the sphere.
    lambda_par : float
        parallel (intra-axonal) diffusivity in 10^9 m^2/s.
    diameter : float
        cylinder diameter in meters.
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
        {'id': 'gaussian_parallel', 'name': 'Gaussian parallel diffusion',
         'condition_human': 'Parallel diffusion is modeled as purely Gaussian along the fiber axis.',
         'severity': 'info'},
        {'id': 'single_axon_diffusivity', 'name': 'Single-axon parallel diffusivity',
         'condition_human': 'lambda_par is the intrinsic single-axon parallel diffusivity, not a dispersed macroscopic value.',
         'severity': 'warning'},
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

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None, diameter=None,
                 diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant'],
                 n_modes=16):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter
        self.diffusion_perpendicular = diffusion_perpendicular
        self.n_modes = int(n_modes)

    def __call__(self, acquisition_scheme, use_jax=False, **kwargs):
        """Signal of the exact-matrix cylinder for the acquisition's gradient waveform.
        ``use_jax=True`` uses the differentiable GPU batch path (needs the ``[jax]`` extra and a stored
        waveform ``_G``; falls back to NumPy for scalar-timing schemes)."""
        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        R = diameter / 2.
        D = self.diffusion_perpendicular
        mu_cart = utils.unitsphere2cart_1d(np.array(mu))

        _G = getattr(acquisition_scheme, '_G', None)
        n = acquisition_scheme.gradient_directions
        bvals = acquisition_scheme.bvalues
        # parallel axis is free (Gaussian): exact from the scheme's own b-value projection
        cos_par = np.dot(n, mu_cart)
        E_par = np.exp(-bvals * lambda_par * cos_par ** 2)
        if _G is not None:
            dt = float(acquisition_scheme._dt)
            n_meas, n_t = _G.shape[0], _G.shape[1]
            perp_frac = np.sqrt(np.clip(1. - cos_par ** 2, 0., None))   # (n_meas,)
            g_axes = (np.einsum('mtc,mc->mt', np.asarray(_G, np.float64), n)  # project onto gradient dir
                      * perp_frac[:, None])                                   # then take perp magnitude
            E_perp = matrix_signal_batch(
                'cylinder', g_axes, dt, D, 2 * R, self.n_modes, use_jax=use_jax)
        else:                                                          # scalar-timing fallback (NumPy)
            n_meas = len(acquisition_scheme.gradient_strengths)
            E_perp = np.ones(n_meas)
            for m in range(n_meas):
                g = acquisition_scheme.gradient_strengths[m]
                if g == 0:
                    continue
                G_m, dt = pgse_waveform(
                    g, acquisition_scheme.delta[m], acquisition_scheme.Delta[m], n[m])
                perp_frac = np.sqrt(max(0., 1. - cos_par[m] ** 2))
                E_perp[m] = matrix_signal(
                    'cylinder', (G_m @ n[m]) * perp_frac, dt, D, 2 * R, self.n_modes)
        return E_par * E_perp


class C6MonteCarloReplayCylinder(ModelProperties, AnisotropicSignalModelProperties):
    r"""Restricted-cylinder compartment whose signal is computed by *replaying* a Monte-Carlo reference
    walk (Substrate Commons canonical replay-pack dataset) rather than a closed-form approximation. Same
    fit parameters as the analytical cylinders — orientation ``mu`` and ``diameter`` — but exact (to the
    Monte-Carlo floor) for **any** gradient waveform, with no short-pulse or Gaussian-phase assumption.
    This is the *generalized pore*: the same machinery replays arbitrary (e.g. mesh) substrates.

    Forward evaluation is dmipy-sim's compiled replay: the waveform is projected onto the pack's stored sine
    bands once, then each replay is a single matmul — identical to the engine's ``pack.replay`` but fast enough
    to fit. ``diameter`` interpolates across the dataset's per-diameter
    packs; orientation rotates the waveform onto the pack's canonical (axis = z) frame.

    **Surface relaxivity is exact, not analytic.** When a surface-relaxivity factor supplies
    ``surface_relaxivity`` (rho, m/s) at call time, it is *not* applied as the mean-field
    ``exp(-TE rho S/V)`` tag-on the analytic compartments use; instead it activates the pack's boundary
    local-time replay knob — a per-walker reweighting by the stored surface contact, optionally
    coherence-gated by a transverse-occupancy schedule (``chi``). This captures the coherence dependence
    the analytic factor cannot.

    Parameters
    ----------
    mu : array, shape(2)      angles [theta, phi] of the cylinder axis on the sphere.
    diameter : float          cylinder diameter in meters.
    diffusivity : float       intrinsic (fixed) diffusivity of the reference dataset, m^2/s (default 2e-9).
    dataset_dir : str, optional   directory holding the replay-pack dataset (else $SUBSTRATE_COMMONS_DATA).
    """
    _citations = {
        'definition': [
            {'key': 'substrate_commons', 'authors': 'Substrate Commons',
             'title': 'Canonical restricted-shape Monte-Carlo replay-pack reference dataset',
             'journal': 'https://substrate-commons.github.io', 'year': 2026, 'doi': ''},
            {'key': 'callaghan1997', 'authors': 'Callaghan PT',
             'title': 'A simple matrix formalism for spin echo analysis of restricted diffusion under generalized gradient waveforms',
             'journal': 'Journal of Magnetic Resonance', 'year': 1997, 'doi': '10.1006/jmre.1997.1233'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'impermeable_membrane', 'name': 'Impermeable membrane assumption',
         'condition_human': 'The reference walk is inside a perfectly impermeable cylinder (no exchange).',
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
    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'diameter': (0.1, 20),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'diameter': DIAMETER_SCALING,
    }
    _parameter_types = {
        'mu': 'orientation',
        'diameter': 'cylinder',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, diameter=None, diffusivity=2.0e-9, dataset_dir=None):
        self.mu = mu
        self.diameter = diameter
        self.diffusivity = float(diffusivity)
        self.dataset_dir = dataset_dir

    def __call__(self, acquisition_scheme, **kwargs):
        from ..data.mc_replay import (load_replay_family, orient_to_z, surface_relaxivity_of)
        diameter = kwargs.get('diameter', self.diameter)
        mu = kwargs.get('mu', self.mu)
        rho = surface_relaxivity_of(kwargs)   # m/s; from a surface factor
        chi = kwargs.get('coherence_gate', None)                    # the transverse gate on the acquisition's own grid
        G = getattr(acquisition_scheme, '_G', None)
        if G is None:
            raise ValueError(
                "C6MonteCarloReplayCylinder needs a waveform-first AcquisitionScheme "
                "(build with AcquisitionScheme.from_pgse/from_waveform) — no ._G on this scheme.")
        fam = load_replay_family('cylinder', self.diffusivity, dataset_dir=self.dataset_dir)
        G_pack = np.asarray(G, np.float64)                              # the scheme's own grid: compiled exactly per pack
        R = orient_to_z(utils.unitsphere2cart_1d(np.asarray(mu, float)))
        G_rot = G_pack @ R.T                                        # lab → canonical (axis = z)
        rho_over_D = (rho / fam.diffusivity) if rho else 0.0        # exact surface replay knob
        return fam.replay_interpolated_raw(G_rot, float(acquisition_scheme._dt), float(diameter), rho_over_D=rho_over_D, chi=chi)
