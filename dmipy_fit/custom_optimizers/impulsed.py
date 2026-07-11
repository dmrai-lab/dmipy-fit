from dmipy_fit.signal_models.sphere_models import S4SphereGaussianPhaseApproximation
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.modeling_framework import MultiCompartmentModel


def IMPULSED(D_in=1.58e-9):
    """Create an IMPULSED two-compartment model (Xu et al. 2014).

    The IMPULSED model (Imaging Microstructural Parameters Using Limited
    Spectrally Edited Diffusion) combines:
    - S4SphereGaussianPhaseApproximation (intra-cellular, restricted)
    - G1Ball (extra-cellular, free Gaussian diffusion)

    Parameters
    ----------
    D_in : float, optional
        Intra-cellular diffusion constant in m^2/s.  Fixed at construction
        time and not fitted.  Default 1.58e-9 m^2/s (Xu 2014).

    Returns
    -------
    MultiCompartmentModel with free parameters:
        diameter (sphere diameter in m)
        lambda_iso (extra-cellular ADC in m^2/s)
        partial_volume_0 / partial_volume_1 (volume fractions)
    """
    sphere = S4SphereGaussianPhaseApproximation(diffusion_constant=D_in)
    ball = G1Ball()
    model = MultiCompartmentModel([sphere, ball])
    return model
