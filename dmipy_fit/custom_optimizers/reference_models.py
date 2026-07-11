"""Reference implementations of multicompartment diffusion MRI models.

Every factory function returns a **top-level** MultiCompartmentModel or
MultiCompartmentSphericalMeanModel — the only types that have a ``.fit()``
method and can be evaluated with ``model(acquisition_scheme, **params)``.

Sub-models (SD1WatsonDistributed, X0GeneralizedKarger, DD1GammaDistributed,
…) are used as building blocks *inside* the top-level model.

Lines-of-Code (LoC) counts the non-blank, non-comment, non-docstring lines
inside the function body needed to create a callable, configured model.

COMPLETE MODEL TABLE
====================
Model                               LoC  Category        Key reference
---------------------------------   ---  --------------  -----------------------------------------------
A1  ball()                            1  Isotropic       Stejskal & Tanner 1965, JCP 42
A2  zeppelin()                        1  Anisotropic     Basser et al. 1994, Biophys J 66
A3  temporal_zeppelin()               1  Time-dep        Novikov et al. 2019, NMR Biomed 32
B1  ball_and_stick()                  1  2-comp WM       Behrens et al. 2003, MRM 50
B2  ball_and_zeppelin()               1  2-comp WM       Panagiotaki et al. 2012, NeuroImage 59
B3  stick_tortuous_zeppelin()         5  Standard Model  Novikov et al. 2019, NMR Biomed 32
B4  free_water_elimination()          3  Free water      Pasternak et al. 2009, MRM 62
B5  ivim()                            3  Perfusion       Le Bihan et al. 1988, Radiology 161
C1  noddi()                           9  ODF dispersion  Zhang et al. 2012, NeuroImage 61
C2  bingham_noddi()                   9  ODF dispersion  Tariq et al. 2016, NeuroImage 133
C3  noddida()                         5  ODF dispersion  Jelescu et al. 2015, NMR Biomed 28
C4  mcsmt()                           5  Spherical mean  Kaden et al. 2016, NeuroImage 139
D1  two_fascicle_noddi()             16  Multi-fascicle  Behrens et al. 2007, NeuroImage 34
E1  charmed()                         7  Cylinder        Assaf & Basser 2005, NeuroImage 27
E2  axcaliber()                       4  Cylinder diam   Assaf et al. 2008, MRM 59
E3  active_ax()                       5  Cylinder diam   Alexander et al. 2010, NeuroImage 52
F1  verdict()                         3  Soma/sphere     Panagiotaki et al. 2014, Cancer Res 74
F2  sandi()                           4  Soma/neurite    Palombo et al. 2020, NeuroImage 215
F3  impulsed()                        3  Soma/tumour     Xu et al. 2019, Magn Reson Med
G1  nexi()                            3  Exchange        Jelescu et al. 2022, NeuroImage 256
G2  karger_two_compartment()          3  Exchange        Kärger 1985, Adv Colloid Interface Sci 23
G3  fexi()                            4  Exchange        Lasič et al. 2011, MRM 66
G4  sandix()                          4  Exchange+soma   SANDI (Palombo 2020) + Kärger 1985
G5  exchange_impulsed()               3  Exchange+soma   Shi et al. 2025, Magn Reson Imaging
H1  temporal_zeppelin_model()         3  Time-dep WM     Novikov et al. 2019, NMR Biomed 32
I1  mte_ball_stick()                  3  Relaxometry     Gong et al. 2020, NeuroImage 217
I2  mte_noddi()                       9  Relaxometry     Gong et al. 2020, NeuroImage 217
I3  mte_sandi()                       5  Relaxometry     ISMRM 2023 abstract #0766
I4  wmti()                            5  Kurtosis-WM     Fieremans et al. 2011, NeuroImage 58
I5  noddida_mte()                     8  Relaxometry     Jelescu 2015 + Gong 2020
I6  mte_impulsed()                    3  Relaxometry     Jiang et al. 2025, MRM

Total: 31 models, 8 categories


IMPORTANT — WHAT IS NOT HERE
=============================
The following model families cannot be directly expressed with the
dmipy-fit building blocks (they require tensor distributions, spectral
bases, or non-Gaussian cumulant expansions) and are therefore omitted:

- DTI / DKI / WMTI*  : tensor/cumulant phenomenology, not biophysical MCM
  (*WMTI-derived biophysical parameters can be estimated post-hoc from DKI
  fits, but the WMTI model structure itself is a simple SM — see I4 wmti())
- QTI / DIVIDE / μFA : diffusion tensor distributions; encode with b-tensor,
  not fit with compartment models
- RSI / DBSI          : basis-spectrum (continuous linear dictionary)
- MT-CSD / MT-MCM    : requires tissue response functions + SH framework
- Myelin water MRI   : multi-exponential T2 spectrum, not diffusion model
- SMEX               : finite-pulse Kärger — handled analytically in
  dmipy_fit.signal_models.exchange_models._karger_propagator_se/ste


USAGE EXAMPLE
=============
>>> from dmipy_fit.custom_optimizers.reference_models import noddi, nexi, sandi
>>> from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
>>> scheme = wu_minn_hcp_acquisition_scheme()
>>> model = noddi()
>>> print(model.parameter_names)
>>> fit = model.fit(scheme, data)
>>> print(fit.fitted_parameters)
"""

import numpy as np

from dmipy_fit.signal_models.gaussian_models import (
    G1Ball,
    G2Zeppelin,
    G3TemporalZeppelin,
)
from dmipy_fit.signal_models.cylinder_models import (
    C1Stick,
    C3CylinderCallaghanApproximation,
    C4CylinderGaussianPhaseApproximation,
)
from dmipy_fit.signal_models.sphere_models import (
    S1Dot,
    S4SphereGaussianPhaseApproximation,
)
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel,
    TransverseRelaxation,
)
from dmipy_fit.signal_models.exchange_models import (
    X0GeneralizedKarger,
)
from dmipy_fit.distributions.distribute_models import (
    SD1WatsonDistributed,
    SD2BinghamDistributed,
    DD1GammaDistributed,
)
from dmipy_fit.core.modeling_framework import (
    MultiCompartmentModel,
    MultiCompartmentSphericalMeanModel,
)
from dmipy_fit.utils.utils import T1_tortuosity

# ---------------------------------------------------------------------------
# Physical constants (published default values)
# ---------------------------------------------------------------------------
_Da   = 1.7e-9    # m²/s  intra-axonal/neurite diffusivity (NODDI, NEXI, …)
_Dcsf = 3.0e-9   # m²/s  cerebrospinal-fluid free diffusivity
_Din  = 1.58e-9   # m²/s  intra-cellular diffusivity (IMPULSED/SANDI default)


# ===========================================================================
# A. Single-compartment Gaussian models
# ===========================================================================

def ball():
    """Isotropic Gaussian (single ADC).

    Stejskal & Tanner 1965, Journal of Chemical Physics 42
    doi:10.1063/1.1695690
    """
    return MultiCompartmentModel([G1Ball()])


def zeppelin():
    """Axially symmetric Gaussian (DTI-like, single fascicle).

    Basser, Mattiello & LeBihan 1994, Biophysical Journal 66
    doi:10.1016/S0006-3495(94)80775-1
    """
    return MultiCompartmentModel([G2Zeppelin()])


def temporal_zeppelin():
    """Anisotropic compartment with structural-disorder time dependence.

    D_par(t) = D_inf + A / sqrt(t).  Sensitive to axon packing disorder.

    Novikov, Fieremans, Jespersen & Kiselev 2019, NMR in Biomedicine 32
    doi:10.1002/nbm.3998
    """
    return MultiCompartmentModel([G3TemporalZeppelin()])


# ===========================================================================
# B. Two-compartment white matter models
# ===========================================================================

def ball_and_stick():
    """Isotropic Ball + zero-radius Stick (intra-axonal).

    Behrens et al. 2003, Magnetic Resonance in Medicine 50
    doi:10.1002/mrm.10609
    """
    return MultiCompartmentModel([C1Stick(), G1Ball()])


def ball_and_zeppelin():
    """Isotropic Ball + anisotropic Zeppelin (tissue DTI tensor).

    Panagiotaki et al. 2012, NeuroImage 59
    doi:10.1016/j.neuroimage.2011.09.081
    """
    return MultiCompartmentModel([G2Zeppelin(), G1Ball()])


def stick_tortuous_zeppelin():
    """Intra-axonal Stick + tortuous extra-axonal Zeppelin.

    The 'Standard Model' two-compartment WM parametrisation.
    Tortuosity: D_e_perp = D_a × (1 − f_intra).
    Orientation and lambda_par shared between compartments.

    Novikov, Fieremans, Jespersen & Kiselev 2019, NMR in Biomedicine 32
    doi:10.1002/nbm.3998
    Szafer, Zhong & Gore 1995, MRM 33 (tortuosity), doi:10.1002/mrm.1910330702
    """
    mcm = MultiCompartmentModel([C1Stick(), G2Zeppelin()])
    mcm.set_tortuous_parameter(          # must come before set_equal_parameter
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1',
    )
    mcm.set_equal_parameter('C1Stick_1_mu', 'G2Zeppelin_1_mu')
    mcm.set_equal_parameter('C1Stick_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    return mcm


def free_water_elimination():
    """Tissue Zeppelin + fixed free-water Ball (D_iso = 3.0e-9 m²/s).

    Removes CSF partial-volume contamination from DTI maps.

    Pasternak, Sochen, Gur, Intrator & Assaf 2009, MRM 62
    doi:10.1002/mrm.22055
    """
    mcm = MultiCompartmentModel([G2Zeppelin(), G1Ball()])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', _Dcsf)
    return mcm


def ivim():
    """IVIM: tissue diffusion + vascular pseudo-diffusion.

    Blood pseudo-diffusion D* fixed at 7×10⁻⁹ m²/s;
    tissue ADC optimisation range [0.5, 6]×10⁻⁹ m²/s.

    Le Bihan et al. 1988, Radiology 161
    doi:10.1148/radiology.168.2.3393671
    """
    mcm = MultiCompartmentModel([G1Ball(), G1Ball()])
    mcm.set_fixed_parameter('G1Ball_2_lambda_iso', 7e-9)
    mcm.set_parameter_optimization_bounds('G1Ball_1_lambda_iso', [0.5e-9, 6e-9])
    return mcm


# ===========================================================================
# C. Orientation-dispersion models
# ===========================================================================

def noddi():
    """NODDI: Watson-dispersed Stick + tortuous Zeppelin + CSF Ball.

    Published defaults: D_a = 1.7×10⁻⁹ m²/s (fixed), D_csf = 3.0×10⁻⁹ m²/s
    (fixed), tortuosity D_e_perp = D_a × (1 − f_intra).

    Zhang, Schneider, Wheeler-Kingshott & Alexander 2012, NeuroImage 61
    doi:10.1016/j.neuroimage.2012.03.072
    """
    bundle = SD1WatsonDistributed(models=[C1Stick(), G2Zeppelin()])
    bundle.set_tortuous_parameter(       # tortuosity before set_equal
        'G2Zeppelin_1_lambda_perp',
        'G2Zeppelin_1_lambda_par',
        'partial_volume_0',
    )
    bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', _Da)
    mcm = MultiCompartmentModel([bundle, G1Ball()])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', _Dcsf)
    return mcm


def bingham_noddi():
    """Bingham-NODDI: Bingham-dispersed Stick + tortuous Zeppelin + CSF Ball.

    Extends NODDI with anisotropic orientation spread (ODI + beta_fraction)
    for fanning/bending fascicles.

    Tariq, Schneider, Alexander, Gandini Wheeler-Kingshott & Zhang 2016,
    NeuroImage 133, doi:10.1016/j.neuroimage.2016.01.046
    """
    bundle = SD2BinghamDistributed(models=[C1Stick(), G2Zeppelin()])
    bundle.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'G2Zeppelin_1_lambda_par',
        'partial_volume_0',
    )
    bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', _Da)
    mcm = MultiCompartmentModel([bundle, G1Ball()])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', _Dcsf)
    return mcm


def noddida():
    """NODDIDA: Stick + tortuous Zeppelin + free Ball — all diffusivities free.

    No fixed D_a; allows exploring parameter degeneracy of the Standard Model.
    CSF replaced by free-diffusing Ball with optimised D_iso.

    Jelescu, Veraart, Adisetiyo, Milla, Novikov & Fieremans 2015,
    NMR in Biomedicine 28, doi:10.1002/nbm.3450
    """
    mcm = MultiCompartmentModel([C1Stick(), G2Zeppelin(), G1Ball()])
    mcm.set_tortuous_parameter(          # tortuosity before set_equal
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1',
    )
    mcm.set_equal_parameter('C1Stick_1_mu', 'G2Zeppelin_1_mu')
    mcm.set_equal_parameter('C1Stick_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    return mcm


def mcsmt():
    """MC-SMT: Multi-Compartment Spherical Mean Technique.

    Orientation-invariant; operates on per-shell spherical mean signal.
    Stick (intra) + tortuous Zeppelin (extra), shared lambda_par.

    Kaden, Kelm, Carson, McKinstry & Rueckert 2016, NeuroImage 139
    doi:10.1016/j.neuroimage.2016.06.002
    """
    mcm = MultiCompartmentSphericalMeanModel([C1Stick(), G2Zeppelin()])
    mcm.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1',
    )
    mcm.set_equal_parameter('C1Stick_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    return mcm


# ===========================================================================
# D. Multi-fascicle models
# ===========================================================================

def two_fascicle_noddi():
    """Two independently oriented Watson-NODDI bundles + shared CSF Ball.

    Each bundle has its own mu, ODI, f_intra.  D_a = 1.7×10⁻⁹ fixed.
    Suitable for crossing-fibre voxels.

    Behrens, Berg, Jbabdi, Rushworth & Woolrich 2007, NeuroImage 34
    doi:10.1016/j.neuroimage.2006.09.018
    """
    bundle1 = SD1WatsonDistributed(models=[C1Stick(), G2Zeppelin()])
    bundle1.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp', 'G2Zeppelin_1_lambda_par', 'partial_volume_0')
    bundle1.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    bundle1.set_fixed_parameter('G2Zeppelin_1_lambda_par', _Da)

    bundle2 = SD1WatsonDistributed(models=[C1Stick(), G2Zeppelin()])
    bundle2.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp', 'G2Zeppelin_1_lambda_par', 'partial_volume_0')
    bundle2.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    bundle2.set_fixed_parameter('G2Zeppelin_1_lambda_par', _Da)

    mcm = MultiCompartmentModel([bundle1, bundle2, G1Ball()])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', _Dcsf)
    return mcm


# ===========================================================================
# E. Cylinder / axon-diameter models
# ===========================================================================

def charmed():
    """CHARMED: Callaghan cylinder (restricted) + tortuous Zeppelin (hindered).

    Orientation and lambda_par shared; uses C3 Callaghan solution
    (more accurate than Stejskal-Tanner approximation at high b).

    Assaf & Basser 2005, NeuroImage 27, doi:10.1016/j.neuroimage.2005.03.042
    """
    cyl  = C3CylinderCallaghanApproximation()
    zepp = G2Zeppelin()
    mcm  = MultiCompartmentModel([cyl, zepp])
    mcm.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C3CylinderCallaghanApproximation_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1',
    )
    mcm.set_equal_parameter(
        'C3CylinderCallaghanApproximation_1_mu', 'G2Zeppelin_1_mu')
    mcm.set_equal_parameter(
        'C3CylinderCallaghanApproximation_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    return mcm


def axcaliber():
    """AxCaliber: Gamma-distributed Callaghan cylinders + Zeppelin.

    Estimates the full axon diameter distribution (α, β of Gamma distribution)
    alongside intra-axonal density.  Requires varying diffusion times.

    Assaf, Blumenfeld-Katzir, Yovel & Basser 2008, MRM 59
    doi:10.1002/mrm.21577
    """
    gamma_cyl = DD1GammaDistributed([C3CylinderCallaghanApproximation()])
    mcm = MultiCompartmentModel([gamma_cyl, G2Zeppelin()])
    mcm.set_equal_parameter(
        'DD1GammaDistributed_1_C3CylinderCallaghanApproximation_1_mu',
        'G2Zeppelin_1_mu',
    )
    return mcm


def active_ax():
    """ActiveAx: single-diameter Callaghan cylinder + Zeppelin + free Ball.

    Orientationally invariant; optimised for clinical scanner gradient
    strengths; estimates a single representative axon diameter index.

    Alexander et al. 2010, NeuroImage 52
    doi:10.1016/j.neuroimage.2010.05.043
    """
    mcm = MultiCompartmentModel([
        C3CylinderCallaghanApproximation(), G2Zeppelin(), G1Ball()])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', _Dcsf)
    return mcm


def charmed_c4():
    """CHARMED with C4 Gaussian Phase cylinder (Van Gelderen) instead of C3 Callaghan.

    Cheaper to JIT/run than C3; valid when delta >> R^2/D (long pulses).
    Use as a fast alternative to charmed() on clinical scanners.

    Assaf & Basser 2005, NeuroImage 27, doi:10.1016/j.neuroimage.2005.03.042
    """
    cyl  = C4CylinderGaussianPhaseApproximation()
    zepp = G2Zeppelin()
    mcm  = MultiCompartmentModel([cyl, zepp])
    mcm.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C4CylinderGaussianPhaseApproximation_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1',
    )
    mcm.set_equal_parameter(
        'C4CylinderGaussianPhaseApproximation_1_mu', 'G2Zeppelin_1_mu')
    mcm.set_equal_parameter(
        'C4CylinderGaussianPhaseApproximation_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    return mcm


def axcaliber_c4():
    """AxCaliber with C4 Gaussian Phase cylinders (Van Gelderen) instead of C3 Callaghan.

    Gamma-distributed cylinder diameters + Zeppelin.  C4 is much cheaper
    to JIT-compile and avoids the CUDA graph OOM that C3×Ns=4 hits.

    Assaf et al. 2008, MRM 59 — doi:10.1002/mrm.21577
    """
    gamma_cyl = DD1GammaDistributed([C4CylinderGaussianPhaseApproximation()])
    mcm = MultiCompartmentModel([gamma_cyl, G2Zeppelin()])
    mcm.set_equal_parameter(
        'DD1GammaDistributed_1_C4CylinderGaussianPhaseApproximation_1_mu',
        'G2Zeppelin_1_mu',
    )
    return mcm


def active_ax_c4():
    """ActiveAx with C4 Gaussian Phase cylinder (Van Gelderen) instead of C3 Callaghan.

    Single representative diameter + Zeppelin + CSF Ball.  C4 is faster
    and avoids the 2 GB constant-table allocation that C3×Ns=3 hits.

    Alexander et al. 2010, NeuroImage 52 — doi:10.1016/j.neuroimage.2010.05.043
    """
    mcm = MultiCompartmentModel([
        C4CylinderGaussianPhaseApproximation(), G2Zeppelin(), G1Ball()])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', _Dcsf)
    return mcm


# ===========================================================================
# F. Soma / sphere models (gray matter and cancer)
# ===========================================================================

def verdict():
    """VERDICT: Sphere (cell body) + Stick (membrane/vascular) + Ball (EES).

    Designed for tumour microstructure; sphere diameter and intra-cellular
    fraction are the key output parameters.

    Panagiotaki et al. 2014, Cancer Research 74
    doi:10.1158/0008-5472.can-13-2511
    """
    return MultiCompartmentModel([
        S4SphereGaussianPhaseApproximation(), C1Stick(), G1Ball()])


def sandi():
    """SANDI: Sphere (soma) + Stick (neurite) + Ball (extra-cellular).

    Gray-matter analog of NODDI; targets soma radius and density.
    Requires high b-values (b ≳ 6 ms/µm²).  D_in fixed at 1.58×10⁻⁹ m²/s.

    Palombo et al. 2020, NeuroImage 215
    doi:10.1016/j.neuroimage.2020.116835
    """
    soma = S4SphereGaussianPhaseApproximation(diffusion_constant=_Din)
    mcm  = MultiCompartmentModel([soma, C1Stick(), G1Ball()])
    mcm.set_fixed_parameter('C1Stick_1_lambda_par', _Da)
    return mcm


def impulsed():
    """IMPULSED: Sphere + isotropic Ball (two-compartment, fixed D_in).

    Designed for measuring mean cell diameter in tumours via varying-Δ OGSE.

    Xu, Jiang, Li et al. 2019, Magnetic Resonance in Medicine
    doi:10.1002/mrm.28056
    """
    soma = S4SphereGaussianPhaseApproximation(diffusion_constant=_Din)
    return MultiCompartmentModel([soma, G1Ball()])


# ===========================================================================
# G. Exchange models (Kärger / NEXI / FEXI / SANDIX)
# ===========================================================================

def nexi():
    """NEXI: Neurite Exchange Imaging — Stick + Zeppelin with Kärger exchange.

    The same generic X0GeneralizedKarger used by karger_two_compartment/fexi, with an
    intra-axonal Stick, an extra-axonal Zeppelin, and the NODDI tortuosity link
    (De_perp = (1 − f)·De_par) pre-wired via parameter_links.  Free parameters:
    mu, f, kappa, C1Stick_1_lambda_par (Di), G2Zeppelin_1_lambda_par (De).

    Jelescu, de Skowronski, Geffroy, Palombo & Novikov 2022, NeuroImage 256
    doi:10.1016/j.neuroimage.2022.119277
    """
    stick, zeppelin = C1Stick(), G2Zeppelin()
    karger = X0GeneralizedKarger(
        stick, zeppelin,
        parameter_links=[(zeppelin, 'lambda_perp', T1_tortuosity(),
                          [(zeppelin, 'lambda_par'), (None, 'f')])])
    return MultiCompartmentModel([karger])


def karger_two_compartment():
    """Generic Kärger two-compartment model: Ball + Ball with exchange.

    The simplest Kärger model — two isotropic pools exchanging at rate kappa.
    Covers FEXI slow/fast pool interpretation and basic intra/extracellular
    exchange without diffusion anisotropy.

    Kärger 1985, Advances in Colloid and Interface Science 23
    doi:10.1016/0001-8686(85)80018-X
    """
    return MultiCompartmentModel([X0GeneralizedKarger(G1Ball(), G1Ball())])


def fexi():
    """FEXI: Filter EXchange Imaging — two isotropic pools with exchange.

    Measures apparent exchange rate (AXR) between slow (intra) and fast
    (extra) diffusion pools.  Requires a diffusion-weighting filter block
    followed by an exchange delay and diffusion readout.

    Lasič, Nilsson, Lätt, Ståhlberg & Topgaard 2011, Magnetic Resonance in Medicine 66
    doi:10.1002/mrm.22782
    """
    slow = G1Ball()
    fast = G1Ball()
    mcm  = MultiCompartmentModel([X0GeneralizedKarger(slow, fast)])
    mcm.set_fixed_parameter('X0GeneralizedKarger_1_G1Ball_2_lambda_iso', 2.0e-9)  # fast pool D
    return mcm


def sandix():
    """SANDIX: SANDI with exchange between soma (sphere) and extracellular (Ball).

    Extends SANDI by adding Kärger-type permeable exchange.  The neurite
    (Stick) compartment is kept as a separate non-exchanging component
    — wrapping only sphere+extracellular in the Kärger pair.

    Composed model (no single reference) — SANDI base + Kärger exchange:
    SANDI: Palombo et al. 2020, NeuroImage 215, doi:10.1016/j.neuroimage.2020.116835
    Exchange: Kärger 1985, Adv Colloid Interface Sci 23, doi:10.1016/0001-8686(85)80018-X
    """
    soma  = S4SphereGaussianPhaseApproximation(diffusion_constant=_Din)
    extra = G1Ball()
    mcm   = MultiCompartmentModel([X0GeneralizedKarger(soma, extra), C1Stick()])
    mcm.set_fixed_parameter('C1Stick_1_lambda_par', _Da)
    return mcm


def exchange_impulsed():
    """EXCHANGE: IMPULSED + transcytolemmal Kärger exchange (tumour).

    Simultaneously maps cell size, cell density, and membrane water exchange
    rate.  Validated in breast cancer patients under neoadjuvant chemotherapy.
    Intra-cellular D_in fixed at 1.58×10⁻⁹ m²/s; exchange rate κ is the
    key new biomarker vs IMPULSED.

    Shi et al. 2025, Magnetic Resonance Imaging (arXiv:2408.01918)
    doi:10.1016/j.mri.2025.110433
    """
    soma  = S4SphereGaussianPhaseApproximation(diffusion_constant=_Din)
    extra = G1Ball()
    return MultiCompartmentModel([X0GeneralizedKarger(soma, extra)])


# ===========================================================================
# H. Time-dependent diffusion models
# ===========================================================================

def temporal_zeppelin_model():
    """Temporal Zeppelin + free Ball: Standard Model with structural disorder.

    G3TemporalZeppelin captures D_par(t) = D_par_∞ + A/√t arising from
    1D disorder along axons; free Ball accounts for CSF partial volume.

    Novikov, Fieremans, Jespersen & Kiselev 2019, NMR in Biomedicine 32
    doi:10.1002/nbm.3998
    """
    mcm = MultiCompartmentModel([G3TemporalZeppelin(), G1Ball()])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', _Dcsf)
    return mcm


# ===========================================================================
# I. Relaxometry-diffusion (multi-TE) models   [2019–2024]
# ===========================================================================

def mte_ball_stick():
    """Multi-TE Ball-and-Stick with per-compartment T2 relaxation.

    T2_intra and T2_extra are free parameters; requires multi-echo acquisition.
    Serves as the simplest jointly estimated diffusion + relaxation model.

    Gong, Tong, He, Sun, Zhong & Zhang 2020, NeuroImage 217
    doi:10.1016/j.neuroimage.2020.116906
    """
    # Per-compartment T2 is an occupancy-gated factor on each compartment; bare
    # C1Stick/G1Ball carry NO T2 parameter.
    intra = OccupancyGatedModel(C1Stick(), [TransverseRelaxation()])
    extra = OccupancyGatedModel(G1Ball(), [TransverseRelaxation()])
    return MultiCompartmentModel([intra, extra])


def mte_noddi():
    """MTE-NODDI: NODDI extended with per-compartment T2 relaxation.

    Per-compartment T2 weighting (T2_intra, T2_extra) is included as free
    parameters; acquire at multiple TE to separate T2-driven volume fraction
    bias from true neurite density.

    Gong, Tong, He, Sun, Zhong & Zhang 2020, NeuroImage 217
    doi:10.1016/j.neuroimage.2020.116906
    """
    # Each compartment wrapped with a T2 factor so per-compartment T2 is fittable.
    bundle = SD1WatsonDistributed(models=[
        OccupancyGatedModel(C1Stick(), [TransverseRelaxation()]),
        OccupancyGatedModel(G2Zeppelin(), [TransverseRelaxation()])])
    bundle.set_tortuous_parameter(
        'OccupancyGatedModel_2_lambda_perp', 'OccupancyGatedModel_2_lambda_par',
        'partial_volume_0')
    bundle.set_equal_parameter(
        'OccupancyGatedModel_2_lambda_par', 'OccupancyGatedModel_1_lambda_par')
    bundle.set_fixed_parameter('OccupancyGatedModel_2_lambda_par', _Da)
    mcm = MultiCompartmentModel([bundle, OccupancyGatedModel(G1Ball(), [TransverseRelaxation()])])
    mcm.set_fixed_parameter('OccupancyGatedModel_1_lambda_iso', _Dcsf)  # CSF ball
    return mcm


def mte_sandi():
    """MTE-SANDI: SANDI with per-compartment T2 relaxation.

    Allows disentangling T2 from diffusion-based compartment fractions in grey
    matter.  All three T2 values (soma, neurite, extra) are free parameters.

    Palombo, Gong & Shemesh 2023, ISMRM abstract #0766
    """
    soma    = OccupancyGatedModel(
        S4SphereGaussianPhaseApproximation(diffusion_constant=_Din), [TransverseRelaxation()])
    neurite = OccupancyGatedModel(C1Stick(), [TransverseRelaxation()])
    extra   = OccupancyGatedModel(G1Ball(), [TransverseRelaxation()])
    mcm = MultiCompartmentModel([soma, neurite, extra])
    mcm.set_fixed_parameter('OccupancyGatedModel_2_lambda_par', _Da)  # neurite stick
    return mcm   # per-compartment T2 (soma, neurite, extra) via the T2 factors


def wmti():
    """WMTI: White Matter Tract Integrity — biophysical Standard Model structure.

    Provides the same Stick + tortuous Zeppelin compartment structure used in
    the WMTI framework.  In practice, WMTI parameters (AWF, De_par, De_perp,
    Da) are estimated analytically from DKI metrics post-hoc, but the same
    model can also be fitted via NLS.

    Fieremans, Jensen & Helpern 2011, NeuroImage 58
    doi:10.1016/j.neuroimage.2011.06.006
    """
    mcm = MultiCompartmentModel([C1Stick(), G2Zeppelin()])
    mcm.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1',
    )
    mcm.set_equal_parameter('C1Stick_1_mu', 'G2Zeppelin_1_mu')
    mcm.set_equal_parameter('C1Stick_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    return mcm


def noddida_mte():
    """NODDIDA-MTE: unconstrained NODDIDA with per-compartment T2.

    Combines the NODDIDA parametrisation (D_a free, D_iso free) with
    per-compartment T2.  Suitable for 7T multi-echo experiments.

    NODDIDA: Jelescu et al. 2015, NMR Biomed 28, doi:10.1002/nbm.3450
    MTE extension: Gong et al. 2020, NeuroImage 217, doi:10.1016/j.neuroimage.2020.116906
    """
    stick    = OccupancyGatedModel(C1Stick(), [TransverseRelaxation()])
    zeppelin = OccupancyGatedModel(G2Zeppelin(), [TransverseRelaxation()])
    extra    = OccupancyGatedModel(G1Ball(), [TransverseRelaxation()])
    mcm = MultiCompartmentModel([stick, zeppelin, extra])
    mcm.set_tortuous_parameter(
        'OccupancyGatedModel_2_lambda_perp',
        'OccupancyGatedModel_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1',
    )
    mcm.set_equal_parameter('OccupancyGatedModel_1_mu', 'OccupancyGatedModel_2_mu')
    mcm.set_equal_parameter('OccupancyGatedModel_1_lambda_par', 'OccupancyGatedModel_2_lambda_par')
    return mcm   # per-compartment T2 (all three) via the T2 factors


def mte_impulsed():
    """MTE-IMPULSED: IMPULSED with per-compartment T2 relaxation.

    Joint estimation of cell diameter, intra/extra-cellular T2, and volume
    fractions from multi-TE OGSE/PGSE data.  D_in left free (unlike single-TE
    IMPULSED) to allow joint diffusion-relaxation estimation.  Validated in
    five tumour models (brain, breast, prostate, melanoma, colon).

    Jiang, Xu, Li, Gore & Does 2025, Magnetic Resonance in Medicine
    doi:10.1002/mrm.30254
    """
    soma  = OccupancyGatedModel(S4SphereGaussianPhaseApproximation(), [TransverseRelaxation()])
    extra = OccupancyGatedModel(G1Ball(), [TransverseRelaxation()])
    return MultiCompartmentModel([soma, extra])   # per-compartment T2 via the T2 factors
