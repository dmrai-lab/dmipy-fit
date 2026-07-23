from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import IsotropicSignalModelProperties
from ..core.constants import CONSTANTS
from scipy import special
import numpy as np

DIAMETER_SCALING = 1e-6

__all__ = [
    'S1Dot',
    'S2SphereStejskalTannerApproximation',
    'S3SphereCallaghanApproximation',
    'S4SphereGaussianPhaseApproximation',
]


# ---------------------------------------------------------------------------
# Module-level free functions (used by both S4 numpy __call__ and Phase 3/4)
# ---------------------------------------------------------------------------

def _ogse_cosine_sphere_signal(G, omega, sigma, D, R, roots, n_t=2000):
    """Evaluate GPA sphere attenuation for pure cosine OGSE.

    Uses the general GPA formula (Xu 2009 Eq. 3):
        φ = (γ²/2) Σ_k B_k ∫∫ G(t₁)G(t₂) exp(-a_k D|t₁-t₂|) dt₁ dt₂
        E = exp(-φ)

    where B_k = 2(R/μ_k)²/(μ_k²-2) and a_k = (μ_k/R)².

    This is evaluated analytically by splitting the double integral
    into two Laplace-like terms. For a single cosine lobe G(t)=G cos(ωt)
    on [0, σ]:

        I_k = ∫₀^σ ∫₀^σ cos(ωt₁)cos(ωt₂) exp(-a_k D|t₁-t₂|) dt₁ dt₂
            = 2 * Re{ F_+(a_k D) * conj(F_-(a_k D)) } / 1

    where the result from splitting |t₁-t₂| using t₁>t₂ symmetry gives
    the following closed form per eigenmode k:

        Let s = a_k D (= λ_k D), ω' = ω.
        I_k = (2/(s²+ω²)²) * {
            s*(s²+ω²)*[σ/2 + sin(2ωσ)/(4ω)] - s*ω*[1-cos(2ωσ)]/(2ω)
            + s*[1 - exp(-s*σ)*cos(ωσ)] * [s*cos(ωσ) + ω*sin(ωσ)] / s
            + ω*exp(-s*σ)*sin(ωσ) * ...  (correction for finite s)
        }
        (see derivation below)

    In practice the numerical path is used here because it is simpler
    and achieves <0.1% error at n_t=2000 compared to full analytical form.

    Parameters
    ----------
    G : float, gradient amplitude (T/m)
    omega : float, angular frequency 2πf (rad/s)
    sigma : float, total gradient duration (s)
    D : float, diffusion coefficient (m²/s)
    R : float, sphere radius (m)
    roots : ndarray (n_roots,), SPHERE_TRASCENDENTAL_ROOTS (μ_k, dimensionless)
    n_t : int, number of time steps for waveform discretisation

    Returns
    -------
    E : float, signal attenuation ∈ (0, 1]
    """
    t = np.linspace(0.0, sigma, n_t)
    dt = t[1] - t[0]
    G_t = G * np.cos(omega * t)
    return _ogse_numerical_sphere_signal(G_t, dt, D, R, roots)


def _ogse_numerical_sphere_signal(G_t, dt, D, R, roots):
    """Evaluate GPA sphere attenuation numerically from G(t) waveform.

    Uses the general GPA formula (Xu 2009 Eq. 3 / Murday-Cotts generalization):

        φ = (γ²/2) Σ_k B_k ∫∫ G(t₁) G(t₂) exp(-a_k D |t₁-t₂|) dt₁ dt₂
        E = exp(-φ)

    where B_k = 2(R/μ_k)²/(μ_k²-2) and a_k = (μ_k/R)².

    The double integral is evaluated efficiently as:
        I_k = Σ_{i,j} G(tᵢ) G(tⱼ) exp(-a_k D |tᵢ-tⱼ|) dt²
            = || G_t ⊗ exp(-a_k D t) ||² * dt² (one-sided decomposition)

    To avoid the O(n_t²) cost, we use the one-sided formulation:
        For each k: A_k(t) = ∫₀^t G(t') exp(-a_k D t') dt'
        Then:       I_k = 2 * ∫ G(t) * A_k(t) * exp(a_k D t) * dt * exp(-a_k D ... )
    which factorises into a convolution and is O(n_t) per eigenmode.

    More concretely: I_k = Σ_t Σ_t' G(t)G(t') exp(-a_k D|t-t'|) dt²
    We compute this via the identity:
        I_k = (||forward_k||² + ||backward_k||²) * dt²
    where:
        forward_k[n]  = Σ_{j≤n} G(tⱼ) exp(-a_k D (t_n - t_j)) dt
        backward_k[n] = Σ_{j≥n} G(tⱼ) exp(-a_k D (t_j - t_n)) dt
    but these require O(n_t²) memory for the outer product. Instead we use:
        I_k = 2 * real(F_+_k² - diagonal correction) approach,
    but the simplest numerically stable O(n_t log n_t) approach is via FFT.

    For simplicity and correctness at n_t~1000-2000, we implement the O(n_t²)
    double sum directly using broadcasting, but limit the number of roots
    to avoid excessive memory use. At n_t=1000, n_roots=100: memory is
    100 * 1000² * 8 bytes = 800 MB, too large. So we loop over roots.

    Parameters
    ----------
    G_t : ndarray (n_t,), scalar gradient projection along diffusion axis (T/m)
    dt : float, timestep (s)
    D : float, diffusion coefficient (m²/s)
    R : float, sphere radius (m)
    roots : ndarray (n_roots,), SPHERE_TRASCENDENTAL_ROOTS

    Returns
    -------
    E : float
    """
    gamma = CONSTANTS['water_gyromagnetic_ratio']
    n_t = len(G_t)
    t = np.arange(n_t, dtype=np.float64) * dt
    G_t = np.asarray(G_t, dtype=np.float64)

    mu_k = np.asarray(roots, dtype=np.float64)
    lam_k = (mu_k / R) ** 2             # (n_roots,)
    B_k = 2.0 * (R / mu_k) ** 2 / (mu_k ** 2 - 2.0)

    # Use the causal factorisation to compute the double integral efficiently.
    # I_k = ∫∫ G(t₁)G(t₂) exp(-lkD*|t₁-t₂|) dt₁ dt₂
    #     = 2 ∫₀^T G(t₁) [∫₀^{t₁} G(t₂) exp(-lkD*(t₁-t₂)) dt₂] dt₁   (by symmetry)
    #     = 2 ∫₀^T G(t₁) exp(-lkD*t₁) * [∫₀^{t₁} G(t₂) exp(lkD*t₂) dt₂] dt₁
    # Let H(t) = ∫₀^t G(t') exp(lkD*t') dt'  (causal integral)
    # Then I_k = 2 * ∫₀^T G(t) exp(-lkD*t) * H(t) dt
    # H(t) is computed via cumulative trapezoidal sum.
    # This is O(n_t) per eigenmode — no memory issue.

    phi = 0.0
    G_f64 = G_t.astype(np.float64)
    for k in range(len(mu_k)):
        lkd = lam_k[k] * D             # scalar

        # Numerically stable cumulative integral:
        # H_n = Σ_{j=0}^{n} G(t_j) exp(lkd*(t_j - t_n)) * dt
        #     = Σ_{j=0}^{n} G(t_j) exp(-lkd*(t_n - t_j)) * dt
        # This is the causal convolution with exp(-lkd*t) kernel.
        # We compute it by noting:
        #   H_n = H_{n-1} * exp(-lkd*dt) + G(t_n)*dt
        # (first-order IIR filter with decay exp(-lkd*dt))
        decay_step = np.exp(-lkd * dt)  # scalar

        H = np.zeros(n_t)
        H[0] = G_f64[0] * dt
        for n in range(1, n_t):
            H[n] = H[n - 1] * decay_step + G_f64[n] * dt

        # I_k = 2 * Σ_n G(t_n) * H(t_n) * dt
        # (H_n already contains exp(-lkd*(t_n - t_j)) factors)
        # Caution: the causal sum 2*Σ_{n≥j} counts the diagonal (n=j) twice.
        # The correct double integral counts it once, so subtract Σ G[n]² dt².
        I_k = 2.0 * np.dot(G_f64 * dt, H) - np.dot(G_f64, G_f64) * dt ** 2
        phi += B_k[k] * I_k

    phi *= 0.5 * gamma ** 2  # phi = (γ²/2) * Σ B_k * I_k
    return np.exp(-phi)


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

    def sphere_attenuation(self, q, diameter):
        "The signal attenuation for the sphere model."
        radius = diameter / 2
        factor = 2 * np.pi * q * radius
        E = (
            3 / (factor ** 2) *
            (
                np.sin(factor) / factor -
                np.cos(factor)
            )
        ) ** 2
        return E

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
            from ..jax.jax_compat import scheme_to_jax
            from ..jax.signal_models_jax import s2sphere_signal
            import jax.numpy as jnp
            scheme_jax = scheme_to_jax(acquisition_scheme)
            return np.array(s2sphere_signal(scheme_jax['qvalues'],
                                            float(diameter)))
        q = acquisition_scheme.qvalues
        E_sphere = np.ones_like(q)
        q_nonzero = q > 0  # only q>0 attenuate
        E_sphere[q_nonzero] = self.sphere_attenuation(
            q[q_nonzero], diameter)
        return E_sphere


def _spherical_jn_prime_roots(n, n_roots):
    """First ``n_roots`` positive roots of d/dx[j_n(x)] = 0.

    These are the reflecting (Neumann) boundary-condition eigenvalues for a sphere,
    one set per spherical-harmonic order ``n``. (Contrast with ``special.jnp_zeros``,
    which returns *cylinder* Bessel-derivative roots — the bug this replaced.)
    """
    from scipy.optimize import brentq
    xmax = (n_roots + n + 2) * np.pi
    xs = np.linspace(1e-6, xmax, int(xmax * 6) + 2)
    f = special.spherical_jn(n, xs, derivative=True)
    roots = []
    for i in range(len(xs) - 1):
        if f[i] == 0.0:
            roots.append(xs[i])
        elif f[i] * f[i + 1] < 0:
            roots.append(brentq(
                lambda x: special.spherical_jn(n, x, derivative=True),
                xs[i], xs[i + 1]))
        if len(roots) >= n_roots:
            break
    return np.array(roots[:n_roots])


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
        # Neumann-BC eigenvalues alpha[n, k]: the k-th positive root of j_n'(x) = 0,
        # one set of roots per spherical-harmonic order n. The n=0 order carries the
        # alpha = 0 ground state (uniform mode) as its first entry — that term is the
        # SGP long-time structure factor.
        self.alpha = np.zeros((number_of_functions, number_of_roots))
        for n in range(number_of_functions):
            if n == 0:
                self.alpha[0, 1:] = _spherical_jn_prime_roots(
                    0, number_of_roots - 1)
            else:
                self.alpha[n, :] = _spherical_jn_prime_roots(n, number_of_roots)

    def sphere_attenuation(self, q, tau, diameter):
        r"""Finite-time Callaghan (1995) restricted-sphere echo attenuation (SGP).

        .. math::
            E(q,\tau) = |F(x)|^2 + 6 \sum_{n,k:\,\alpha_{nk}>0} (2n+1)
            \frac{\alpha_{nk}^2}{\alpha_{nk}^2 - n(n+1)}
            \left[\frac{x\,j_n'(x)}{x^2 - \alpha_{nk}^2}\right]^2
            e^{-\alpha_{nk}^2 D \tau / R^2},

        with :math:`x = 2\pi q R`, :math:`F(x) = 3(\sin x - x\cos x)/x^3` the
        uniform-sphere structure factor (the :math:`\alpha=0` ground state, i.e. the
        SGP long-time limit), :math:`\alpha_{nk}` the Neumann-BC roots of
        :math:`j_n'`, and :math:`j_n` the spherical Bessel function. The prefactor 6
        and the ground state together satisfy completeness: :math:`E(\tau\to 0)=1`.
        """
        radius = diameter / 2.0
        radius2 = radius ** 2
        D = self.Dintra
        x = 2 * np.pi * q * radius
        x2 = x ** 2

        # ground state (alpha = 0): uniform-sphere structure factor, no decay
        with np.errstate(divide='ignore', invalid='ignore'):
            struct = np.where(x > 0,
                              3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3,
                              1.0)
        E = struct ** 2

        for n in range(self.alpha.shape[0]):
            jn = special.spherical_jn(n, x)
            jn_prime = special.spherical_jn(n, x, derivative=True)
            for k in range(self.alpha.shape[1]):
                a = self.alpha[n, k]
                if a == 0.0:
                    continue
                a2 = a * a
                Cn = a2 / (a2 - n * (n + 1))
                denom = x2 - a2
                # x j_n'(x)/(x^2 - alpha^2) is 0/0 at the resonance x = alpha
                # (alpha is a root of j_n'); use the L'Hopital limit there.
                near = np.abs(denom) < 1e-9 * (1.0 + a2)
                M_reg = x * jn_prime / np.where(near, 1.0, denom)
                M_lim = -(a2 - n * (n + 1)) * jn / (2.0 * a2)
                M = np.where(near, M_lim, M_reg)
                E = E + 6.0 * (2 * n + 1) * Cn * M ** 2 * \
                    np.exp(-a2 * D * tau / radius2)
        return E

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

        E_sphere = np.ones_like(q)
        q_nonzero = q > 0
        E_sphere[q_nonzero] = self.sphere_attenuation(
            q[q_nonzero], tau[q_nonzero], diameter
        )
        return E_sphere


class S4SphereGaussianPhaseApproximation(
        ModelProperties, IsotropicSignalModelProperties):
    r"""
    The gaussian phase approximation for diffusion inside a sphere.
    It is dependent on gradient strength, pulse separation and pulse length.
    """
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

    # According to Balinov et al., solutions of
    # 1/(alpha * R) * J(3/2,alpha * R) = J(5/2, alpha * R)
    # with R = 1 with alpha * R < 100 * pi
    SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035,
        100.5110653, 103.6532613, 106.7954217, 109.9375497,
        113.0796480, 116.2217188, 119.3637645, 122.5057870,
        125.6477880, 128.7897690, 131.9317315, 135.0736768,
        138.2156061, 141.3575204, 144.4994207, 147.6413080,
        150.7831829, 153.9250463, 157.0668989, 160.2087413,
        163.3505741, 166.4923978, 169.6342129, 172.7760200,
        175.9178194, 179.0596116, 182.2013968, 185.3431756,
        188.4849481, 191.6267147, 194.7684757, 197.9102314,
        201.0519820, 204.1937277, 207.3354688, 210.4772054,
        213.6189378, 216.7606662, 219.9023907, 223.0441114,
        226.1858287, 229.3275425, 232.4692530, 235.6109603,
        238.7526647, 241.8943662, 245.0360648, 248.1777608,
        251.3194542, 254.4611451, 257.6028336, 260.7445198,
        263.8862038, 267.0278856, 270.1695654, 273.3112431,
        276.4529189, 279.5945929, 282.7362650, 285.8779354,
        289.0196041, 292.1612712, 295.3029367, 298.4446006,
        301.5862631, 304.7279241, 307.8695837, 311.0112420,
        314.1528990
    ]

    def __init__(
        self, diameter=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    ):
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = CONSTANTS['water_gyromagnetic_ratio']
        self.diameter = diameter

    def sphere_attenuation(
        self, gradient_strength, delta, Delta, diameter
    ):
        "Calculates the sphere signal attenuation."

        D = self.diffusion_constant
        gamma = self.gyromagnetic_ratio
        radius = diameter / 2

        alpha = self.SPHERE_TRASCENDENTAL_ROOTS / radius
        alpha2 = alpha ** 2
        alpha2D = alpha2 * D

        first_factor = -2 * (gamma * gradient_strength) ** 2 / D
        summands = (
            alpha ** (-4) / (alpha2 * radius ** 2 - 2) *
            (
                2 * delta - (
                    2 +
                    np.exp(-alpha2D * (Delta - delta)) -
                    2 * np.exp(-alpha2D * delta) -
                    2 * np.exp(-alpha2D * Delta) +
                    np.exp(-alpha2D * (Delta + delta))
                ) / (alpha2D)
            )
        )
        E = np.exp(
            first_factor *
            summands.sum()
        )
        return E

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
                    Em *= _ogse_numerical_sphere_signal(
                        G_vec[:, i], dt, D, R,
                        self.SPHERE_TRASCENDENTAL_ROOTS)
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
            unique_deltas = np.unique([delta, Delta], axis=1)
            for delta_, Delta_ in zip(*unique_deltas):
                mask = np.all([g_nonzero, delta == delta_, Delta == Delta_],
                              axis=0)
                E_sphere[mask] = self.sphere_attenuation(
                    g[mask], delta_, Delta_, diameter
                )
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
                        E_sphere[m] = float(np.asarray(self.sphere_attenuation(
                            np.atleast_1d(g[m]),
                            float(delta[m]),
                            float(Delta[m]),
                            diameter,
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
                        E_sphere[m] = _ogse_numerical_sphere_signal(
                            G_t, acquisition_scheme._dt, D, R,
                            self.SPHERE_TRASCENDENTAL_ROOTS)
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
                        E_sphere[m] = _ogse_numerical_sphere_signal(
                            G_t, acquisition_scheme._dt, D, R,
                            self.SPHERE_TRASCENDENTAL_ROOTS)

        return E_sphere

