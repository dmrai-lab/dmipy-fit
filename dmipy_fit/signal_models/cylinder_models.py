from __future__ import division

import numpy as np
from scipy import special
from scipy.special import erf

from ..utils import utils
from ..core.constants import CONSTANTS
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import AnisotropicSignalModelProperties
from dipy.utils.optpkg import optional_package
from dipy.reconst.shm import real_sh_tournier as _real_sh_tournier

numba, have_numba, _ = optional_package("numba")

DIFFUSIVITY_SCALING = 1e-9
DIAMETER_SCALING = 1e-6
A_SCALING = 1e-12

# ---------------------------------------------------------------------------
# Sphere quadrature grid for SH dispersion integrals.
#
# Uses the dipy 'symmetric724' grid (724 points on the full sphere) with
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
    from dipy.data import get_sphere
    sphere = get_sphere(name='symmetric724')
    pts = sphere.vertices.astype(np.float64)          # (724, 3)
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
    'C4CylinderGaussianPhaseApproximation'
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
            from ..jax.jax_compat import scheme_to_jax, unitsphere2cart_1d_jax
            from ..jax.signal_models_jax import c1stick_signal
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            mu_cart = unitsphere2cart_1d_jax(jnp.array(mu))
            return np.array(c1stick_signal(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                mu_cart, float(lambda_par_)))
        mu_cart = utils.unitsphere2cart_1d(mu)
        # B-tensor path: E = exp(-λ_par · uᵀBu)  (λ_perp = 0 for stick)
        # Correct for PGSE (rank-1 B = b·n⊗n) and tensor-valued encoding.
        B = acquisition_scheme.btensor()                              # (n_m, 3, 3)
        u_dot_B_u = np.einsum('i,mij,j->m', mu_cart, B, mu_cart)     # (n_m,)
        E_stick = np.exp(-lambda_par_ * u_dot_B_u)
        return E_stick

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
        bvals = acquisition_scheme.shell_bvalues
        bvals_ = bvals[~acquisition_scheme.shell_b0_mask]

        lambda_par = kwargs.get('lambda_par', self.lambda_par)

        E_mean = np.ones_like(bvals)
        bval_indices_above0 = bvals > 0
        bvals_ = bvals[bval_indices_above0]
        E_mean_ = ((np.sqrt(np.pi) * erf(np.sqrt(bvals_ * lambda_par))) /
                   (2 * np.sqrt(bvals_ * lambda_par)))
        E_mean[bval_indices_above0] = E_mean_
        return E_mean

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

    def perpendicular_attenuation(
        self, q, diameter
    ):
        "Returns the cylinder's perpendicular signal attenuation."
        radius = diameter / 2
        # Eq. [6] in the paper
        E = ((2 * special.jn(1, 2 * np.pi * q * radius)) ** 2 /
             (2 * np.pi * q * radius) ** 2)
        return E

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
        E_parallel = _attenuation_parallel_stick(bvals, lambda_par, n, mu)
        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular
        q_nonzero = q_perp > 0  # only q>0 attenuate
        E_perpendicular[q_nonzero] = self.perpendicular_attenuation(
            q_perp[q_nonzero], diameter
        )
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

        self.alpha = np.empty((number_of_roots, number_of_functions))
        self.alpha[0, 0] = 0
        if number_of_roots > 1:
            self.alpha[1:, 0] = special.jnp_zeros(0, number_of_roots - 1)
        for m in range(1, number_of_functions):
            self.alpha[:, m] = special.jnp_zeros(m, number_of_roots)

    def perpendicular_attenuation(self, q, tau, diameter):
        "Implements the finite time Callaghan model for cylinders"
        radius = diameter / 2.
        alpha = self.alpha
        q_argument = 2 * np.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = np.zeros_like(q)

        J = special.j1(q_argument) ** 2
        for k in range(0, self.alpha.shape[0]):
            alpha2 = alpha[k, 0] ** 2
            update = (
                4 * np.exp(-alpha2 * self.diffusion_perpendicular *
                           tau / radius ** 2) *
                q_argument_2 /
                (q_argument_2 - alpha2) ** 2 * J
            )
            res += update

        for m in range(1, self.alpha.shape[1]):
            J = special.jvp(m, q_argument, 1)
            q_argument_J = (q_argument * J) ** 2
            for k in range(self.alpha.shape[0]):
                alpha2 = self.alpha[k, m] ** 2
                update = (
                    8 * np.exp(-alpha2 * self.diffusion_perpendicular *
                               tau / radius ** 2) *
                    alpha2 / (alpha2 - m ** 2) *
                    q_argument_J /
                    (q_argument_2 - alpha2) ** 2
                )
                res += update
        return res

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
            fn = build_c3cylinder_jax_fn(self.alpha, self.diffusion_perpendicular)
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
        E_parallel = _attenuation_parallel_stick(bvals, lambda_par, n, mu)
        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular

        q_nonzero = q_perp > 0
        E_perpendicular[q_nonzero] = self.perpendicular_attenuation(
            q_perp[q_nonzero], tau[q_nonzero], diameter
        )
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
    _CYLINDER_TRASCENDENTAL_ROOTS = np.sort(special.jnp_zeros(1, 100))

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant'],
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.gyromagnetic_ratio = CONSTANTS['water_gyromagnetic_ratio']
        self.diameter = diameter

    def perpendicular_attenuation(
        self, gradient_strength, delta, Delta, diameter
    ):
        "Calculates the cylinder's perpendicular signal attenuation."
        D = self.diffusion_perpendicular
        gamma = self.gyromagnetic_ratio
        return _attenuation_perpendicular_gaussian_phase(
            diameter, gradient_strength, delta, Delta,
            D, gamma, self._CYLINDER_TRASCENDENTAL_ROOTS)

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
            (2) Fixed-direction OGSE (oscillation_frequency > 0): analytical
                cosine or numerical IIR path (Xu 2009 / Stepisnik).
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
            roots_jax = jnp.array(self._CYLINDER_TRASCENDENTAL_ROOTS)
            return np.array(c4cylinder_signal(
                scheme_jax['bvalues'], scheme_jax['gradient_directions'],
                scheme_jax['gradient_strengths'], scheme_jax['delta'],
                scheme_jax['Delta'], mu_cart, float(lambda_par), float(diameter),
                float(self.diffusion_perpendicular), float(self.gyromagnetic_ratio),
                roots_jax))

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
            C_geom = _compute_C_geom(R, D, self._CYLINDER_TRASCENDENTAL_ROOTS)
            b_par = bvals * np.dot(n, mu_cart) ** 2
            E = _cylinder_signal_from_gamma_lm(
                gamma_lm, b_par, C_geom, mu_cart, lambda_par)
            return E

        g = acquisition_scheme.gradient_strengths
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta
        mu_perpendicular_plane = np.eye(3) - np.outer(mu_cart, mu_cart)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = _attenuation_parallel_stick(bvals, lambda_par, n, mu_cart)
        n_m = acquisition_scheme.number_of_measurements
        E_perpendicular = np.ones(n_m, dtype=float)

        if osc_freq is None or np.all(osc_freq == 0):
            # ----------------------------------------------------------------
            # Pure PGSE path — original Van Gelderen GPA (unchanged)
            # ----------------------------------------------------------------
            g_perp = g * magnitude_perpendicular
            g_nonzero = g_perp > 0
            unique_deltas = np.unique([delta, Delta], axis=1)
            for delta_, Delta_ in zip(*unique_deltas):
                mask = np.all([g_nonzero, delta == delta_, Delta == Delta_],
                              axis=0)
                E_perpendicular[mask] = self.perpendicular_attenuation(
                    g_perp[mask], delta_, Delta_, diameter
                )
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
                        _attenuation_perpendicular_gaussian_phase(
                            diameter,
                            np.atleast_1d(G_perp_m),
                            float(delta[m]),
                            float(Delta[m]),
                            D,
                            self.gyromagnetic_ratio,
                            self._CYLINDER_TRASCENDENTAL_ROOTS,
                        )).reshape(-1)[0])
                else:
                    t_r = float(acquisition_scheme.gradient_rise_time[m])
                    if t_r > 0:
                        # Trapezoidal OGSE: numerical Stepisnik path
                        G_t_vec = acquisition_scheme._G[m].astype(
                            np.float64)  # (n_t, 3)
                        # Project onto perpendicular plane then take magnitude
                        G_t_perp_vec = np.dot(
                            mu_perpendicular_plane,
                            G_t_vec.T).T  # (n_t, 3)
                        G_t = np.linalg.norm(G_t_perp_vec, axis=-1)  # (n_t,)
                        # Preserve sign from first non-zero sample
                        first_nz = np.argmax(np.abs(G_t) > 0)
                        ref_sign = np.dot(
                            mu_perpendicular_plane,
                            G_t_vec[first_nz])
                        if np.sum(ref_sign) < 0:
                            G_t = -G_t
                        E_perpendicular[m] = _ogse_numerical_cylinder_signal(
                            G_t, acquisition_scheme._dt, D, R,
                            self._CYLINDER_TRASCENDENTAL_ROOTS)
                    else:
                        # Pure cosine OGSE: analytical path (Xu 2009)
                        sigma_m = float(acquisition_scheme.gradient_duration[m])
                        omega_m = 2.0 * np.pi * freq_m
                        E_perpendicular[m] = _ogse_cosine_cylinder_signal(
                            G_perp_m, omega_m, sigma_m, D, R,
                            self._CYLINDER_TRASCENDENTAL_ROOTS)

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

        R = diameter / 2.0
        D = self.diffusion_perpendicular

        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        mu_cart = utils.unitsphere2cart_1d(mu)

        gamma_lm = acquisition_scheme.gamma_lm(l_max=4)
        C_geom = _compute_C_geom(R, D, self._CYLINDER_TRASCENDENTAL_ROOTS)
        b_par = bvals * np.dot(n, mu_cart) ** 2
        E = _cylinder_signal_from_gamma_lm(gamma_lm, b_par, C_geom, mu_cart, lambda_par)

        return E

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

        R = diameter / 2.0
        D = self.diffusion_perpendicular

        gamma_lm = acquisition_scheme.gamma_lm(l_max=4)
        C_geom = _compute_C_geom(R, D, self._CYLINDER_TRASCENDENTAL_ROOTS)

        Gamma00 = gamma_lm[:, 0]
        int_G2_dt = Gamma00 * np.sqrt(4.0 * np.pi)
        Gamma2m = gamma_lm[:, 1:]          # (n_m, 5)

        c0_perp = C_geom * (2.0 / 3.0) * int_G2_dt        # (n_m,)
        c2m_perp = -C_geom * (8.0 * np.pi / 15.0) * Gamma2m  # (n_m, 5)

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


def _compute_C_geom(R, D, roots):
    """Scalar geometry factor for the fast-eigenmode Gamma_lm cylinder GPA path.

    C_geom = gamma^2 * sum_k B_k / alpha_k

    where B_k = 2*(R/mu_k)^2 / (mu_k^2 - 1) are the cylinder eigenmode
    amplitude weights, alpha_k = (mu_k/R)^2 * D are the eigenmode decay rates,
    and mu_k are the transcendental roots (zeros of J1').

    This is valid in the fast-eigenmode limit: alpha_1 * sigma >> 1, where
    sigma is the effective waveform duration (PGSE: delta; OGSE: 1/(2*f)).
    For R=2 um, D=1.7e-9: tau_1 = 1/alpha_1 = R^2/(mu_1^2*D) ~ 0.7 ms.
    Equivalently: f << alpha_1/(2*pi) ~ 230 Hz for R=2 um.
    Breaks for high-frequency OGSE (f > ~alpha_1/(2*pi)) or ultra-short
    PGSE pulses (delta < ~tau_1). The exact result requires the spectral
    factorisation c_2m = integral a_2m(omega) * L(omega, alpha_1) domega.

    Parameters
    ----------
    R : float
        Cylinder radius in metres.
    D : float
        Perpendicular diffusivity in m^2/s.
    roots : ndarray (n_roots,)
        Transcendental roots mu_k (J1' zeros = _CYLINDER_TRASCENDENTAL_ROOTS).

    Returns
    -------
    C_geom : float
        Scalar geometry factor in SI units (s/m^2 per (T/m)^2 = s^3/(kg^2*m)).
    """
    gamma = CONSTANTS['water_gyromagnetic_ratio']
    mu_n = roots
    B_k = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)
    alpha_k = (mu_n / R) ** 2 * D
    return gamma ** 2 * np.sum(B_k / alpha_k)


def _eval_Y2m_at_direction(mu_cart):
    """Evaluate the five l=2 real spherical harmonics at a Cartesian direction.

    Parameters
    ----------
    mu_cart : (3,) array
        Unit vector (cylinder axis direction).

    Returns
    -------
    Y2m : (5,) array, float64
        Values [Y2-2, Y2-1, Y20, Y21, Y22] at mu_cart.
    """
    x, y, z = mu_cart[0], mu_cart[1], mu_cart[2]
    Y2m2 = np.sqrt(15.0 / (4.0 * np.pi)) * x * y
    Y2m1 = np.sqrt(15.0 / (4.0 * np.pi)) * y * z
    Y20  = np.sqrt(5.0 / (16.0 * np.pi)) * (2.0 * z**2 - x**2 - y**2)
    Y21  = np.sqrt(15.0 / (4.0 * np.pi)) * x * z
    Y22  = np.sqrt(15.0 / (16.0 * np.pi)) * (x**2 - y**2)
    return np.array([Y2m2, Y2m1, Y20, Y21, Y22], dtype=np.float64)


def _cylinder_signal_from_gamma_lm(gamma_lm, b_par, C_geom, mu_cart, lambda_par):
    """Compute cylinder GPA signal from precomputed Gamma_lm coefficients.

    Uses the fast-eigenmode factorisation of the GPA phase variance:

        phi(mu) = C_geom * [2/3 * int|G|^2 dt - (8*pi/15) * sum_m Gamma_2m * Y2m(mu)]
                + b_par * lambda_par

    where int|G|^2 dt = Gamma_00 * sqrt(4*pi)  (inverse of Y00 normalisation).

    This path is valid when the eigenmode decay time (~0.7 us for R < 10 um,
    D ~ 1.7e-9) is far shorter than the gradient rotation period (~ms). The
    cylinder geometry acts as an l=2 low-pass filter on the encoding, so only
    l=0 and l=2 Gamma_lm contribute.

    Parameters
    ----------
    gamma_lm : (n_m, 6) array
        Columns: [Gamma_00, Gamma_2-2, Gamma_2-1, Gamma_20, Gamma_21, Gamma_22].
    b_par : (n_m,) array
        Parallel b-value for each measurement: bvalues * dot(n, mu)^2.
    C_geom : float
        Geometry factor from _compute_C_geom(R, D, roots).
    mu_cart : (3,) array
        Cylinder axis in Cartesian coordinates.
    lambda_par : float
        Parallel diffusivity in m^2/s.

    Returns
    -------
    E : (n_m,) array
        Signal attenuation in [0, 1].
    """
    # Gamma_00 column; recover int|G|^2 dt by inverting Y00 normalisation
    Gamma00 = gamma_lm[:, 0]           # (n_m,)
    int_G2_dt = Gamma00 * np.sqrt(4.0 * np.pi)   # = int |G(t)|^2 dt

    # l=2 Gamma coefficients: columns 1-5
    Gamma2m = gamma_lm[:, 1:]          # (n_m, 5)

    # Y2m evaluated at cylinder axis mu
    Y2m_mu = _eval_Y2m_at_direction(mu_cart)   # (5,)

    # Perpendicular phase exponent (fast-eigenmode)
    phi_perp = C_geom * (
        (2.0 / 3.0) * int_G2_dt
        - (8.0 * np.pi / 15.0) * Gamma2m.dot(Y2m_mu)
    )

    # Parallel phase exponent (Gaussian diffusion along axis)
    phi_par = b_par * lambda_par

    E = np.exp(-(phi_perp + phi_par))
    return E


def _attenuation_parallel_stick(bvals, lambda_par, n, mu):
    "Free gaussian diffusion for parallel cylinder direction."
    return np.exp(-bvals * lambda_par * np.dot(n, mu) ** 2)


def _attenuation_perpendicular_gaussian_phase(
        diameter, gradient_strength, delta, Delta,
        D, gamma, CYLINDER_TRASCENDENTAL_ROOTS):
    "Perpendicular Gaussian Phase signal attenuation."
    radius = diameter / 2.
    first_factor = -2 * (gradient_strength * gamma) ** 2
    alpha = CYLINDER_TRASCENDENTAL_ROOTS / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    summands = (
        2 * alpha2D * delta - 2 +
        2 * np.exp(-alpha2D * delta) +
        2 * np.exp(-alpha2D * Delta) -
        np.exp(-alpha2D * (Delta - delta)) -
        np.exp(-alpha2D * (Delta + delta))
    ) / (D ** 2 * alpha ** 6 * (radius ** 2 * alpha2 - 1))

    E = np.exp(first_factor * summands.sum())
    return E


def _ogse_cosine_cylinder_signal(G, omega, sigma, D, R, roots):
    """Evaluate GPA cylinder attenuation for pure cosine OGSE.

    Uses the Stepisnik GPA formula:
        φ = (γ²/2) Σ_n B_n ∫₀^σ ∫₀^σ G(t)G(t') exp(-λ_n D|t-t'|) dt dt'
        E = exp(-φ)

    where B_n = 2(R/μ_n)²/(μ_n²-1) and λ_n = (μ_n/R)² are the cylinder
    eigenvalues (J₁' zeros: μ₁=1.8412, ...).

    The double integral for G(t) = G₀ cos(ωt) on [0,σ] evaluates to:
        I_n = (2/d) * [α*(σ/2 + s*c/(2ω)) - α*F_s + s²/2]
    where α = λ_n D, d = α²+ω², c = cos(ωσ), s = sin(ωσ),
    and F_s = (α(1-c·e^{-ασ}) + ω·s·e^{-ασ})/d is the one-sided integral.

    Parameters
    ----------
    G : float, gradient amplitude (T/m) — perpendicular component
    omega : float, angular frequency 2πf (rad/s)
    sigma : float, total gradient duration (s)
    D : float, perpendicular diffusion coefficient (m²/s)
    R : float, cylinder radius (m)
    roots : ndarray (n_roots,), _CYLINDER_TRASCENDENTAL_ROOTS (μ_n, J1' zeros)

    Returns
    -------
    E_perp : float, perpendicular signal attenuation ∈ (0, 1]
    """
    gamma = CONSTANTS['water_gyromagnetic_ratio']
    mu_n = roots                        # shape (n_roots,)
    lam_n = (mu_n / R) ** 2            # eigenvalues (m⁻²)
    B_n = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)  # cylinder coefficients

    alpha = lam_n * D                   # α_n = λ_n D  (s⁻¹)
    denom = alpha ** 2 + omega ** 2     # (n_roots,)

    c = np.cos(omega * sigma)
    s = np.sin(omega * sigma)
    e_asm = np.exp(-alpha * sigma)

    # One-sided Laplace integral: F_s = ∫₀^σ cos(ωt) exp(-αt) dt
    F_s = (alpha * (1.0 - c * e_asm) + omega * s * e_asm) / denom  # (n_roots,)

    # Full symmetric double integral:
    # I_n = ∫₀^σ ∫₀^σ cos(ωt) cos(ωt') exp(-α|t-t'|) dt dt'
    #      = (2/d) * [α*(σ/2 + s*c/(2ω)) - α*F_s + s²/2]
    I_n = (2.0 / denom) * (
        alpha * (sigma / 2.0 + s * c / (2.0 * omega))
        - alpha * F_s
        + s ** 2 / 2.0
    )

    phi = 0.5 * gamma ** 2 * G ** 2 * np.sum(B_n * I_n)
    return np.exp(-phi)


def _ogse_numerical_cylinder_signal(G_t, dt, D, R, roots):
    """Evaluate GPA cylinder attenuation numerically from G(t) waveform.

    Uses the Stepisnik GPA formula:
        φ = (γ²/2) Σ_n B_n ∫₀^σ ∫₀^σ G(t)G(t') exp(-λ_n D|t-t'|) dt dt'
        E = exp(-φ)

    The double integral is evaluated via the causal IIR factorisation:
        I_n = 2·∫G(t)·H_n(t)·dt  where H_n(t) = ∫₀^t G(t')exp(-λ_nD(t-t'))dt'
    with a diagonal correction to avoid double-counting:
        I_n = 2·dot(G·dt, H_n) − dot(G,G)·dt²

    Parameters
    ----------
    G_t : ndarray (n_t,), scalar gradient projection along perpendicular axis (T/m)
    dt : float, timestep (s)
    D : float, diffusion coefficient (m²/s)
    R : float, cylinder radius (m)
    roots : ndarray (n_roots,), _CYLINDER_TRASCENDENTAL_ROOTS

    Returns
    -------
    E_perp : float
    """
    gamma = CONSTANTS['water_gyromagnetic_ratio']
    n_t = len(G_t)
    G_f64 = G_t.astype(np.float64)

    mu_n = roots
    lam_n = (mu_n / R) ** 2             # (n_roots,)
    B_n = 2.0 * (R / mu_n) ** 2 / (mu_n ** 2 - 1.0)

    phi = 0.0
    for k in range(len(mu_n)):
        lk_D = lam_n[k] * D
        decay_step = np.exp(-lk_D * dt)
        H = np.zeros(n_t, dtype=np.float64)
        H[0] = G_f64[0] * dt
        for i in range(1, n_t):
            H[i] = H[i - 1] * decay_step + G_f64[i] * dt
        # I_k = ∫∫ G(t)G(t') e^{-lkD|t-t'|} dt dt'  (diagonal-corrected)
        I_k = 2.0 * np.dot(G_f64 * dt, H) - np.dot(G_f64, G_f64) * dt ** 2
        phi += B_n[k] * I_k

    phi *= 0.5 * gamma ** 2
    return np.exp(-phi)


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
    The quadrature uses the 724-point dipy sphere with uniform weights.
    Accuracy for smooth cylinder GPA exponentials (l<=4 significant content)
    is better than 0.1% in spherical mean.

    SH convention: Tournier/MRtrix real SHs with Y00 = 1/(2*sqrt(pi)).
    Projection: E_lm = 4*pi * sum_q w_q * exp(-phi(mu_q)) * Y_lm(mu_q)
    """
    n_m = len(c0)
    n_lm = (l_max + 1) * (l_max + 2) // 2

    # SH basis evaluated at quadrature points (724, n_lm)
    Y_Q, _, _ = _real_sh_tournier(l_max, _SPHERE_QUAD_THETA, _SPHERE_QUAD_PHI,
                                  legacy=False)  # (N_q, n_lm)
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


def watson_odf_lm(mu_cart, kappa, l_max=4):
    """Compute SH coefficients of the Watson ODF for given axis and concentration.

    Uses the exact analytical erfi-based recurrence (no grid approximation).
    The l=0 coefficient is exactly 1/(2*sqrt(pi)) for all kappa.

    Parameters
    ----------
    mu_cart : (3,) ndarray
        Mean orientation as a Cartesian unit vector.
    kappa : float
        Watson concentration parameter.  kappa=0 → isotropic; large kappa →
        delta function at mu_cart.
    l_max : int, optional
        Maximum SH order.  Must be even.  Default 4.

    Returns
    -------
    odf_lm : (n_lm,) ndarray, float64
        SH coefficients (Tournier real SH half-shell ordering).
        odf_lm[0] = 1/(2*sqrt(pi)) for any normalized Watson ODF.
    """
    from ..utils.sh_analytical import watson_sh as _watson_sh
    mu_cart = np.asarray(mu_cart, dtype=np.float64)
    return _watson_sh(mu_cart, float(kappa), l_max=l_max)


if have_numba:
    _attenuation_parallel_stick = numba.njit()(_attenuation_parallel_stick)
    _attenuation_perpendicular_gaussian_phase = numba.njit()(
        _attenuation_perpendicular_gaussian_phase)
