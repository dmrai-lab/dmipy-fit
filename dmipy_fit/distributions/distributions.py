# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division
import os as _os
from os.path import join

import numpy as np
from scipy import stats
from scipy import special
from dipy.reconst.shm import real_sh_tournier as real_sym_sh_mrtrix

from dmipy_fit.utils import utils
from scipy import interpolate
from dmipy_fit.core.modeling_framework import ModelProperties
from dipy.utils.optpkg import optional_package
from dipy.data import get_sphere, HemiSphere
sphere = get_sphere(name='symmetric724')
hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)

numba, have_numba, _ = optional_package("numba")

GRADIENT_TABLES_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'gradient_tables')
SIGNAL_MODELS_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'signal_models')
DATA_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'data')
SPHERE_CARTESIAN = np.loadtxt(
    join(GRADIENT_TABLES_PATH, 'sphere_with_cap.txt')
)
SPHERE_SPHERICAL = utils.cart2sphere(SPHERE_CARTESIAN)

inverse_sh_matrix_kernel = {
    sh_order: np.linalg.pinv(real_sym_sh_mrtrix(
        sh_order, hemisphere.theta, hemisphere.phi, legacy=False
    )[0]) for sh_order in np.arange(0, 15, 2)
}
BETA_SCALING = 1e-6

__all__ = [
    'get_sh_order_from_odi',
    'SD1Watson',
    'SD2Bingham',
    'DD1Gamma',
    'DD2Poisson',
    'odi2kappa',
    'kappa2odi'
]


def get_sh_order_from_odi(odi):
    "Returns minimum sh_order to estimate spherical harmonics for given odi."
    odis = np.array([0.80606061, 0.46666667, 0.25333333,
                     0.15636364, 0.09818182, 0.06909091, 0.])
    sh_orders = np.arange(2, 15, 2)
    return sh_orders[np.argmax(odis < odi)]


class SD1Watson(ModelProperties):
    r""" The Watson spherical distribution model.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    kappa : float,
        concentration parameter of the Watson distribution.
    """
    _citations = {
        'definition': [
            {'key': 'kaden2007', 'authors': 'Kaden E, Knosche TR, Anwander A',
             'title': 'Parametric spherical deconvolution: inferring anatomical connectivity using diffusion MR imaging',
             'journal': 'NeuroImage',
             'year': 2007, 'doi': '10.1016/j.neuroimage.2007.07.023'},
            {'key': 'zhang2012', 'authors': 'Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC',
             'title': 'NODDI: practical in vivo neurite orientation dispersion and density imaging of the human brain',
             'journal': 'NeuroImage',
             'year': 2012, 'doi': '10.1016/j.neuroimage.2012.03.072'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'single_bundle',
         'name': 'Single fiber bundle assumption',
         'condition_human': 'Assumes a single fiber population per voxel. Cannot represent crossing fibers. In crossing regions, the parametric distribution will fit a compromise that does not represent either bundle and biases volume fractions in multi-compartment models like NODDI.',
         'severity': 'warning',
         'source_key': 'zhang2012'},
    ]

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'odi': (0.02, 0.99),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'odi': 1.,
    }
    _parameter_types = {
        'mu': 'orientation',
        'odi': 'normal'
    }
    _model_type = 'SphericalDistribution'

    def __init__(self, mu=None, odi=None):
        self.mu = mu
        self.odi = odi

    def __call__(self, n, **kwargs):
        r""" The Watson spherical distribution model [1, 2].

        Parameters
        ----------
        n : array of shape(3) or array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        Wn: float or array of shape(N),
            Probability density at orientations n, given mu and kappa.
        """
        odi = kwargs.get('odi', self.odi)
        mu = kwargs.get('mu', self.mu)

        kappa = odi2kappa(odi)
        mu_cart = utils.unitsphere2cart_1d(mu)
        numerator = np.exp(kappa * np.dot(n, mu_cart) ** 2)
        denominator = 4 * np.pi * special.hyp1f1(0.5, 1.5, kappa)
        Wn = numerator / denominator
        return Wn

    def spherical_harmonics_representation(self, sh_order=None, **kwargs):
        r""" The Watson spherical distribution model in spherical harmonics.
        The minimum order is automatically derived from numerical experiments
        to ensure fast function execution and accurate results.

        Uses an exact analytical formula (erfi-based recurrence) rather than
        a grid approximation, so all SH orders and concentrations are exact.

        Parameters
        ----------
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        watson_sh : array,
            spherical harmonics of Watson probability density.
        """
        from ..utils.sh_analytical import watson_sh as _watson_sh
        odi = kwargs.get('odi', self.odi)
        mu = kwargs.get('mu', self.mu)
        if sh_order is None:
            sh_order = get_sh_order_from_odi(odi)

        kappa = float(np.asarray(odi2kappa(odi)).flat[0])
        mu_cart = utils.unitsphere2cart_1d(mu)
        return _watson_sh(mu_cart, kappa, l_max=sh_order)


class SD2Bingham(ModelProperties):
    r""" The Bingham spherical distribution model using angles.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.
    """
    _citations = {
        'definition': [
            {'key': 'kaden2007', 'authors': 'Kaden E, Knosche TR, Anwander A',
             'title': 'Parametric spherical deconvolution: inferring anatomical connectivity using diffusion MR imaging',
             'journal': 'NeuroImage',
             'year': 2007, 'doi': '10.1016/j.neuroimage.2007.07.023'},
            {'key': 'sotiropoulos2012', 'authors': 'Sotiropoulos SN, Behrens TEJ, Jbabdi S',
             'title': 'Ball and rackets: inferring fiber fanning from diffusion-weighted MRI',
             'journal': 'NeuroImage',
             'year': 2012, 'doi': '10.1016/j.neuroimage.2012.01.056'},
            {'key': 'tariq2016', 'authors': 'Tariq M, Schneider T, Alexander DC, Wheeler-Kingshott CA, Zhang H',
             'title': 'Bingham-NODDI: Mapping anisotropic orientation dispersion of neurites using diffusion MRI',
             'journal': 'NeuroImage',
             'year': 2016, 'doi': '10.1016/j.neuroimage.2015.12.046'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'single_bundle',
         'name': 'Single fiber bundle assumption',
         'condition_human': 'Assumes a single fiber population per voxel. Cannot represent crossing fibers. In crossing regions, the parametric distribution will fit a compromise that does not represent either bundle and biases volume fractions in multi-compartment models like NODDI.',
         'severity': 'warning',
         'source_key': 'zhang2012'},
    ]

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'psi': (0, np.pi),
        'odi': (0.02, 0.99),
        'beta_fraction': (0, 1)  # beta<=kappa in fact
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'psi': 1.,
        'odi': 1.,
        'beta_fraction': 1.
    }
    _parameter_types = {
        'mu': 'orientation',
        'psi': 'circular',
        'odi': 'normal',
        'beta_fraction': 'normal'
    }
    _model_type = 'SphericalDistribution'

    def __init__(self, mu=None, psi=None, odi=None, beta_fraction=None):
        self.mu = mu
        self.psi = psi
        self.odi = odi
        self.beta_fraction = beta_fraction

    def __call__(self, n, **kwargs):
        r""" The Watson spherical distribution model.

        Parameters
        ----------
        n : array of shape(3) or array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        Bn: float or array of shape(N),
            Probability density at orientations n, given mu and kappa.
        """
        odi = kwargs.get('odi', self.odi)
        beta_fraction = kwargs.get('beta_fraction', self.beta_fraction)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)

        kappa = odi2kappa(odi)
        beta = beta_fraction * kappa

        mu_cart = utils.unitsphere2cart_1d(mu)

        R = utils.rotation_matrix_100_to_theta_phi_psi(mu[0], mu[1], psi)
        mu_beta = R.dot(np.r_[0., 1., 0.])
        from ..utils.sh_analytical import bingham_normalization
        numerator = _probability_bingham(kappa, beta, mu_cart, mu_beta, n)
        Bn = numerator / bingham_normalization(kappa, beta)
        return Bn

    def spherical_harmonics_representation(self, sh_order=None, **kwargs):
        r""" The Bingham spherical distribution model in spherical harmonics.
        The minimum order is automatically derived from numerical experiments
        to ensure fast function executation and accurate results.

        Parameters
        ----------
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        bingham_sh : array,
            spherical harmonics of Bingham probability density.
        """
        from ..utils.sh_analytical import bingham_sh as _bingham_sh
        odi = kwargs.get('odi', self.odi)
        beta_fraction = kwargs.get('beta_fraction', self.beta_fraction)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        if sh_order is None:
            sh_order = get_sh_order_from_odi(odi)

        kappa = float(np.asarray(odi2kappa(odi)).flat[0])
        beta = float(np.asarray(beta_fraction * kappa).flat[0])
        mu_cart = utils.unitsphere2cart_1d(mu)
        psi_val = float(np.asarray(psi).flat[0])
        return _bingham_sh(mu_cart, psi_val, kappa, beta, l_max=sh_order)


class SD3SphericalHarmonics(ModelProperties):
    r"""A real-valued spherical harmonics distribution.

    Parameters
    ----------
    sh_order: int,
        maximum spherical harmonics order.
    sh_coeff: np.ndarray that must be of shape corresponding to sh_order.
        spherical harmonics coefficients of the distribution.
    """
    _citations = {
        'definition': [
            {'key': 'descoteaux2007', 'authors': 'Descoteaux M, Angelino E, Fitzgibbons S, Bhatt R',
             'title': 'Regularized, fast, and robust analytical Q-ball imaging',
             'journal': 'Magnetic Resonance in Medicine',
             'year': 2007, 'doi': '10.1002/mrm.21277'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = []

    def __init__(self, sh_order, sh_coeff=None):
        self.sh_order = sh_order
        self.N_coeff = int((sh_order + 2) * (sh_order + 1) // 2)
        if sh_coeff is not None:
            if len(sh_coeff) != self.N_coeff:
                msg = 'if given, sh_coeff length must correspond to N_coeffs '\
                      'associated with sh_order ({} vs {}).'
                raise ValueError(msg.format(len(sh_coeff), self.N_coeff))
        self.sh_coeff = sh_coeff

        self._parameter_ranges = {'sh_coeff': [
            [None, None] for i in range(self.N_coeff)]}
        self._parameter_scales = {'sh_coeff':
                                  np.ones(self.N_coeff, dtype=float)}
        self._parameter_cardinality = {'sh_coeff': self.N_coeff}
        self._parameter_types = {'sh_coeff': 'sh_coefficients'}
        self._parameter_optimization_flags = {'sh_coeff': True}

    def __call__(self, n, **kwargs):
        r"""Returns the sphere function at cartesian orientations n given
        spherical harmonic coefficients.

        Parameters
        ----------
        n : array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        SHn: array of shape(N),
            Probability density at orientations n, given sh coeffs.
        """
        # calculate SHT matrix
        _, theta, phi = utils.cart2sphere(n).T
        SHT = real_sym_sh_mrtrix(self.sh_order, theta, phi, legacy=False)[0]
        # transform coefficients to sphere values
        sh_coeff = kwargs.get('sh_coeff', self.sh_coeff)
        SHn = SHT.dot(sh_coeff)
        return SHn

    def spherical_harmonics_representation(self, **kwargs):
        r"""Returns the spherical harmonic coefficients themselves.
        """
        return kwargs.get('sh_coeff', self.sh_coeff)


class DD1Gamma(ModelProperties):
    r"""A Gamma distribution of cylinder diameter for given alpha and beta
    parameters. NOTE: This is a distribution for axon DIAMETER and not SURFACE.
    To simulate the diffusion signal of an ensemble of gamma-distributed
    cylinders the probability still needs to be corrected for cylinder surface
    by multiplying by np.pi * radius ** 2 and renormalizing. Reason being
    that diffusion signals are generated by the volume of spins inside axons
    (cylinders), which is proportional to cylinder surface and not to diameter.

    Parameters
    ----------
    alpha : float,
        shape of the gamma distribution.
    beta : float,
        scale of the gamma distrubution. Different from Bingham distribution!
    """
    _citations = {
        'definition': [
            {'key': 'assaf2008', 'authors': 'Assaf Y, Blumenfeld-Katzir T, Yovel Y, Basser PJ',
             'title': 'AxCaliber: a method for measuring axon diameter distribution from diffusion MRI',
             'journal': 'Magnetic Resonance in Medicine',
             'year': 2008, 'doi': '10.1002/mrm.21577'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = []
    _parameter_ranges = {
        'alpha': (0.1, 30.),
        'beta': (1e-3, 2)
    }
    _parameter_scales = {
        'alpha': 1.,
        'beta': BETA_SCALING,
    }
    _parameter_types = {
        'alpha': 'normal',
        'beta': 'normal'
    }
    _model_type = 'SpatialDistribution'

    def __init__(self, alpha=None, beta=None, Nsteps=30,
                 normalization='standard'):
        self.alpha = alpha
        self.beta = beta
        self.Nsteps = Nsteps

        if normalization == 'standard':
            self.norm_func = self.unity
        elif normalization == 'plane':
            self.norm_func = self.length_plane
        elif normalization == 'cylinder':
            self.norm_func = self.surface_cylinder
        elif normalization == 'sphere':
            self.norm_func = self.volume_sphere
        else:
            msg = "Unknown normalization {}".format(normalization)
            raise ValueError(msg)
        self.calculate_sampling_start_and_end_points(self.norm_func)

    def length_plane(self, radius):
        "The distance normalization function for planes."
        return 2 * radius

    def surface_cylinder(self, radius):
        "The surface normalization function for cylinders."
        return np.pi * radius ** 2

    def volume_sphere(self, radius):
        "The volume normalization function for spheres."
        return (4. / 3.) * np.pi * radius ** 3

    def unity(self, radius):
        "The standard normalization for the Gamma distribution (none)."
        return np.ones(len(radius))

    def calculate_sampling_start_and_end_points(self, norm_func, gridsize=50):
        """
        For a given normalization function calculates the best start and end
        points to sample for all possible values of alpha, beta. This is done
        to make sure the function does not sample where the probability of
        basically zero.

        The function is based on first doing a dense sampling and then finding
        out which points need to be included to have sampled at least 99% of
        the area under the probability density curve.

        It sets two interpolator functions that can be called for any
        combination of alpha,beta and to return the appropriate start and end
        sampling points.

        Parameters
        ----------
        norm_func : normalization function,
            normalization of the model, depends on if it's a sphere/cylinder.
        gridsize : integer,
            value that decides how big the grid will be on which we define the
            start and end sampling points.
        """
        start_grid = np.ones([gridsize, gridsize])
        end_grid = np.ones([gridsize, gridsize])

        alpha_range = (np.array(self._parameter_ranges['alpha']) *
                       self._parameter_scales['alpha'])
        beta_range = (np.array(self._parameter_ranges['beta']) *
                      self._parameter_scales['beta'])

        alpha_linspace = np.linspace(alpha_range[0], alpha_range[1], gridsize)
        beta_linspace = np.linspace(beta_range[0], beta_range[1], gridsize)

        for i, alpha in enumerate(alpha_linspace):
            for j, beta in enumerate(beta_linspace):
                gamma_distribution = stats.gamma(alpha, scale=beta)
                outer_limit = (
                    gamma_distribution.mean() + 9 * gamma_distribution.std())
                x_grid = np.linspace(1e-8, outer_limit, 500)
                pdf = gamma_distribution.pdf(x_grid)
                pdf *= norm_func(x_grid)
                cdf = np.cumsum(pdf)
                cdf /= cdf.max()
                inverse_cdf = np.cumsum(pdf[::-1])[::-1]
                inverse_cdf /= inverse_cdf.max()
                end_grid[i, j] = x_grid[np.argmax(cdf > 0.995)]
                start_grid[i, j] = x_grid[np.argmax(inverse_cdf < 0.995)]
        start_grid = np.clip(start_grid, 1e-8, np.inf)
        end_grid = np.clip(end_grid, 1e-7, np.inf)

        alpha_grid, beta_grid = np.meshgrid(alpha_linspace, beta_linspace)

        self.start_interpolator = interpolate.bisplrep(alpha_grid.ravel(),
                                                       beta_grid.ravel(),
                                                       start_grid.T.ravel(),
                                                       kx=2, ky=2)

        self.end_interpolator = interpolate.bisplrep(alpha_grid.ravel(),
                                                     beta_grid.ravel(),
                                                     end_grid.T.ravel(),
                                                     kx=2, ky=2)

    def __call__(self, **kwargs):
        r"""
        Parameters
        ----------
        diameter : float or array, shape (N)
            cylinder (axon) diameter in meters.

        Returns
        -------
        Pgamma : float or array, shape (N)
            probability of cylinder diameter for given alpha and beta.
        """
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)

        gamma_dist = stats.gamma(alpha, scale=beta)
        start_point = interpolate.bisplev(alpha, beta, self.start_interpolator)
        end_point = interpolate.bisplev(alpha, beta, self.end_interpolator)
        start_point = max(start_point, 1e-8)
        radii = np.linspace(start_point, end_point, self.Nsteps)
        normalization = self.norm_func(radii)
        radii_pdf = gamma_dist.pdf(radii)
        radii_pdf_area = radii_pdf * normalization
        radii_pdf_normalized = (
            radii_pdf_area /
            np.trapezoid(x=radii, y=radii_pdf_area)
        )
        return radii, radii_pdf_normalized


class DD2Poisson(ModelProperties):
    r"""A Poisson-approximated diameter distribution for cylinder ensembles.

    Parameterized by a single mean diameter ``mu`` (meters). The variance
    equals the mean when both are expressed in micrometers — the defining
    Poisson property (variance = mean). Implemented as a Gamma distribution
    with ``alpha = mu / BETA_SCALING`` and ``beta = BETA_SCALING`` (1 µm),
    which gives exactly mean = mu and variance = mu × BETA_SCALING (in m),
    equivalent to variance = mu_µm (in µm²).

    Use this instead of :class:`DD1Gamma` when:

    * You want a single free parameter (mean diameter only)
    * The diameter distribution is tight and unimodal (preclinical spinal cord)
    * You want the Poisson constraint variance = mean to hold

    Parameters
    ----------
    mu : float
        Mean cylinder diameter in meters (SI). Range 0.1–20 µm.
    Nsteps : int
        Number of quadrature points for diameter integration.
    normalization : str
        Weighting normalization: 'standard' (no weight), 'cylinder' (surface
        area ∝ r²), 'sphere' (volume ∝ r³), 'plane' (length ∝ r).
    """
    _citations = {
        'definition': [
            {'key': 'desantis2018', 'authors': 'De Santis S, et al.',
             'title': 'Poisson parameterisation of axon diameter distributions in spinal cord diffusion MRI',
             'journal': 'ISMRM 2018 Abstract',
             'year': 2018, 'doi': '10.1002/mrm.27680'}
        ],
        'default_parameters': {},
    }
    _validity_constraints = []
    # Named 'mean_diameter' (not 'mu') so it never collides with a fibre
    # orientation 'mu' when this distribution wraps an anisotropic compartment;
    # the rotational-harmonics kernel keys the fibre axis off params ending in
    # 'mu', which this deliberately does not.
    _parameter_ranges = {'mean_diameter': (0.1, 20.)}
    _parameter_scales = {'mean_diameter': BETA_SCALING}   # 1 µm
    _parameter_types = {'mean_diameter': 'normal'}
    _model_type = 'SpatialDistribution'

    def __init__(self, mean_diameter=None, Nsteps=30, normalization='standard',
                 gridsize=50):
        self.mean_diameter = mean_diameter
        self.Nsteps = Nsteps

        if normalization == 'standard':
            self.norm_func = self._unity
        elif normalization == 'plane':
            self.norm_func = self._length_plane
        elif normalization == 'cylinder':
            self.norm_func = self._surface_cylinder
        elif normalization == 'sphere':
            self.norm_func = self._volume_sphere
        else:
            raise ValueError("Unknown normalization {}".format(normalization))

        self._calculate_sampling_start_and_end_points(self.norm_func,
                                                      gridsize=gridsize)

    @staticmethod
    def _unity(radius):
        return np.ones(len(radius))

    @staticmethod
    def _length_plane(radius):
        return 2 * radius

    @staticmethod
    def _surface_cylinder(radius):
        return np.pi * radius ** 2

    @staticmethod
    def _volume_sphere(radius):
        return (4. / 3.) * np.pi * radius ** 3

    def _calculate_sampling_start_and_end_points(self, norm_func,
                                                  gridsize=50):
        """Precompute integration bounds over the full mu parameter range.

        Builds two 1-D interpolators (start and end quadrature point as a
        function of mu) so that __call__ avoids the 500-point CDF scan on
        every forward pass.  Same 99.5% weighted-CDF criterion as DD1Gamma.
        """
        mu_range = (np.array(self._parameter_ranges['mean_diameter']) *
                    self._parameter_scales['mean_diameter'])
        mu_linspace = np.linspace(mu_range[0], mu_range[1], gridsize)

        start_pts = np.empty(gridsize)
        end_pts = np.empty(gridsize)

        for i, mu in enumerate(mu_linspace):
            alpha = mu / BETA_SCALING
            gd = stats.gamma(alpha, scale=BETA_SCALING)
            outer = gd.mean() + 9. * gd.std()
            x = np.linspace(1e-8, outer, 500)
            pdf_w = gd.pdf(x) * norm_func(x)
            cdf = np.cumsum(pdf_w)
            cdf /= cdf.max()
            inv_cdf = np.cumsum(pdf_w[::-1])[::-1]
            inv_cdf /= inv_cdf.max()
            end_pts[i] = x[np.argmax(cdf > 0.995)]
            start_pts[i] = max(x[np.argmax(inv_cdf < 0.995)], 1e-8)

        self._mu_grid = mu_linspace
        self._start_pts = np.clip(start_pts, 1e-8, np.inf)
        self._end_pts = np.clip(end_pts, 1e-7, np.inf)

    def __call__(self, **kwargs):
        r"""Return quadrature radii and normalized probability weights.

        Parameters
        ----------
        mu : float
            Mean diameter in meters.

        Returns
        -------
        radii : np.ndarray, shape (Nsteps,)
            Diameter quadrature points in meters.
        pdf_normalized : np.ndarray, shape (Nsteps,)
            Normalized probability weights (area under radii × pdf_normalized = 1).
        """
        mu = kwargs.get('mean_diameter', self.mean_diameter)
        # Gamma(alpha, scale=beta) with beta=1µm gives variance=mean in µm.
        alpha = mu / BETA_SCALING
        gamma_distribution = stats.gamma(alpha, scale=BETA_SCALING)

        # Look up integration bounds from precomputed interpolators (O(1)).
        start_point = float(np.interp(mu, self._mu_grid, self._start_pts))
        end_point = float(np.interp(mu, self._mu_grid, self._end_pts))

        radii = np.linspace(start_point, end_point, self.Nsteps)
        normalization = self.norm_func(radii)
        radii_pdf = gamma_distribution.pdf(radii)
        radii_pdf_area = radii_pdf * normalization
        radii_pdf_normalized = (
            radii_pdf_area /
            np.trapezoid(x=radii, y=radii_pdf_area)
        )
        return radii, radii_pdf_normalized


def _probability_bingham(kappa, beta, mu, mu_beta, n):
    "Non-normalized probability of the Bingham distribution."
    return np.exp(kappa * np.dot(n, mu) ** 2 +
                  beta * np.dot(n, mu_beta) ** 2)


def odi2kappa(odi):
    "Calculates concentration (kappa) from orientation dispersion index (odi)."
    return 1. / np.tan(odi * (np.pi / 2.0))


def kappa2odi(kappa):
    "Calculates orientation dispersion index (odi) from concentration (kappa)."
    return (2. / np.pi) * np.arctan(1. / kappa)


if have_numba:
    get_sh_order_from_odi = numba.njit()(get_sh_order_from_odi)
    _probability_bingham = numba.njit()(_probability_bingham)
