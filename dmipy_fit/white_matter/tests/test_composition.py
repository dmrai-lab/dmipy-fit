"""Decoupled diffusion-only white-matter model + the surface-relaxivity inter-compartment weighting.

The canonical WM model builds and forward-simulates as a standard MultiCompartmentModel, and
surface relaxivity introduces a b-independent signal weighting between intra and extra (the
per-compartment surface attenuation differs), which is measurable in the b0-normalised signal.
"""
import numpy as np
import numpy.testing as npt

from dmipy_fit.white_matter.composition import (
    build_white_matter_model, canonical_parameters)
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme

scheme = wu_minn_hcp_acquisition_scheme()
scheme.TE = 0.08


def test_model_builds_and_forward_simulates():
    model, params = build_white_matter_model()
    E = np.asarray(model(scheme, **params))
    assert E.shape[0] == scheme.bvalues.shape[0]
    assert np.all(E >= 0)
    # b0 signal is S0 * sum_i f_i w_i  (< M0 because surface relaxivity attenuates)
    assert E[scheme.b0_mask].mean() > 0


def test_surface_relaxivity_reweights_intra_extra():
    """Surface relaxivity (rho2>0) attenuates intra and extra differently, so the
    b0-normalised diffusion signal differs from the rho2=0 model — the inter-compartment
    weighting the susceptibility/MWF paper identified, absent from plain stick+Zeppelin."""
    b0 = scheme.b0_mask
    dwi = ~b0
    m0, p0 = build_white_matter_model(rho2=0.0)
    m1, p1 = build_white_matter_model(rho2=15e-6)
    E0 = np.asarray(m0(scheme, **p0))
    E1 = np.asarray(m1(scheme, **p1))

    # 1) absolute b0 is attenuated by surface relaxivity
    assert E1[b0].mean() < E0[b0].mean()

    # 2) the weighting is measurable: the b0-normalised DWI is not identical
    n0 = E0 / E0[b0].mean()
    n1 = E1 / E1[b0].mean()
    assert np.abs(n1[dwi] - n0[dwi]).max() > 1e-2


def test_canonical_parameters_overridable():
    p = canonical_parameters(g_ratio=0.65, rho2=20e-6)
    assert p['OccupancyGatedModel_1_g_ratio'] == 0.65
    assert p['OccupancyGatedModel_2_surface_relaxivity'] == 20e-6


def test_builder_exposes_compartment_T1():
    """Each compartment carries a longitudinal-relaxation factor, so the model exposes
    OccupancyGatedModel_<n>_T1 parameters (inert without a stimulated-echo TM)."""
    model, params = build_white_matter_model()
    t1_names = [n for n in model.parameter_names if n.endswith('_T1')]
    assert t1_names == ['OccupancyGatedModel_1_T1',
                        'OccupancyGatedModel_2_T1',
                        'OccupancyGatedModel_3_T1']
    for n in t1_names:
        assert n in params
    # T1 is overridable via keyword
    p = canonical_parameters(T1_intra=1.5)
    assert p['OccupancyGatedModel_1_T1'] == 1.5
    # with CSF the ball also carries a T1
    model_csf, params_csf = build_white_matter_model(include_csf=True)
    assert 'OccupancyGatedModel_4_T1' in model_csf.parameter_names
    assert 'OccupancyGatedModel_4_T1' in params_csf


def test_exterior_sv_derived_from_gamma():
    """S_ext/V of the extra compartment is derived from the Gamma diameter distribution
    (the sim-consistent closed form), not hand-set, and is overridable."""
    from dmipy_fit.white_matter.composition import white_matter_compartments, DEFAULTS
    from dmipy_fit.white_matter.surface import exterior_surface_to_volume
    comps = white_matter_compartments()
    sv = comps[1].factors[0].S_ext_over_V           # extra Zeppelin, ExteriorSurfaceRelaxivity
    expected = exterior_surface_to_volume(
        DEFAULTS['f_axon'], DEFAULTS['gamma_shape'], DEFAULTS['gamma_scale_outer_diameter'])
    assert abs(sv - expected) < 1e-6 * expected
    # explicit override wins
    assert white_matter_compartments(S_ext_over_V=7e5)[1].factors[0].S_ext_over_V == 7e5
