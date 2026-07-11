"""Tests for the citation graph walker and methods-section generator."""

import pytest
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import (
    C1Stick, C2CylinderStejskalTannerApproximation,
    C3CylinderCallaghanApproximation,
    C4CylinderGaussianPhaseApproximation)
from dmipy_fit.signal_models.sphere_models import (
    S1Dot, S2SphereStejskalTannerApproximation,
    _S3SphereCallaghanApproximation,
    S4SphereGaussianPhaseApproximation)
from dmipy_fit.signal_models.plane_models import (
    P2PlaneStejskalTannerApproximation,
    P3PlaneCallaghanApproximation)
from dmipy_fit.signal_models.capped_cylinder_models import (
    CC2CappedCylinderStejskalTannerApproximation,
    CC3CappedCylinderCallaghanApproximation)
from dmipy_fit.signal_models.tissue_response_models import (
    TR1IsotropicTissueResponseModel,
    TR2AnisotropicTissueResponseModel)
from dmipy_fit.distributions.distributions import (
    SD1Watson, SD2Bingham, SD3SphericalHarmonics, DD1Gamma, DD2Poisson)
from dmipy_fit.distributions.distribute_models import DistributedModel
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.core.spherical_mean_framework import MultiCompartmentSphericalMeanModel
from dmipy_fit.core.spherical_harmonics_framework import MultiCompartmentSphericalHarmonicsModel
from dmipy_fit.optimizers.brute2fine import GlobalBruteOptimizer, Brute2FineOptimizer
from dmipy_fit.optimizers.mix import MixOptimizer
from dmipy_fit.optimizers.multi_tissue_volume_fraction import MultiTissueVolumeFractionOptimizer
from dmipy_fit.optimizers_fod.csd_tournier import CsdTournierOptimizer
from dmipy_fit.optimizers_fod.csd_cvxpy import CsdCvxpyOptimizer
from dmipy_fit.optimizers_fod.csd_plus import CsdPlusOptimizer
from dmipy_fit.jax.optimizers_jax import JaxOptimizer
from dmipy_fit.jax.csd_jax import CsdOsqpOptimizer

from dmipy_fit.audit.citations import walk_citation_graph
from dmipy_fit.audit.methods_section import generate_methods_section, generate_bibtex
from dmipy_fit.audit.biophysical_constants import BIOPHYSICAL_CONSTANTS


# ---- All model classes that MUST have _citations ----

SIGNAL_MODEL_CLASSES = [
    G1Ball, G2Zeppelin,
    C1Stick, C2CylinderStejskalTannerApproximation,
    C3CylinderCallaghanApproximation,
    C4CylinderGaussianPhaseApproximation,
    S1Dot, S2SphereStejskalTannerApproximation,
    _S3SphereCallaghanApproximation,
    S4SphereGaussianPhaseApproximation,
    P2PlaneStejskalTannerApproximation,
    P3PlaneCallaghanApproximation,
    CC2CappedCylinderStejskalTannerApproximation,
    CC3CappedCylinderCallaghanApproximation,
    TR1IsotropicTissueResponseModel,
    TR2AnisotropicTissueResponseModel,
]

DISTRIBUTION_CLASSES = [
    SD1Watson, SD2Bingham, SD3SphericalHarmonics, DD1Gamma, DD2Poisson,
]

FRAMEWORK_CLASSES = [
    MultiCompartmentModel,
    MultiCompartmentSphericalMeanModel,
    MultiCompartmentSphericalHarmonicsModel,
]

OPTIMIZER_CLASSES = [
    GlobalBruteOptimizer, Brute2FineOptimizer,
    MixOptimizer, MultiTissueVolumeFractionOptimizer,
    CsdTournierOptimizer, CsdCvxpyOptimizer, CsdPlusOptimizer,
    JaxOptimizer, CsdOsqpOptimizer,
]

DISTRIBUTED_MODEL_CLASS = DistributedModel

ALL_ANNOTATED_CLASSES = (
    SIGNAL_MODEL_CLASSES + DISTRIBUTION_CLASSES +
    FRAMEWORK_CLASSES + OPTIMIZER_CLASSES + [DISTRIBUTED_MODEL_CLASS]
)


# ---- Test 1: walk a single model ----

def test_walk_single_model():
    ball = G1Ball()
    result = walk_citation_graph(ball)
    assert len(result['citations']) >= 1
    assert any(c['key'] == 'behrens2003' for c in result['citations'])


# ---- Test 2: walk multi-compartment ----

def test_walk_multi_compartment():
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[stick, ball])
    result = walk_citation_graph(mcm)
    keys = {c['key'] for c in result['citations']}
    # Must find citations from MCM itself + stick + ball
    assert 'behrens2003' in keys
    assert 'panagiotaki2012' in keys


# ---- Test 3: walk distributed model ----

def test_walk_distributed():
    from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
    stick = C1Stick()
    watson_stick = SD1WatsonDistributed(models=[stick])
    result = walk_citation_graph(watson_stick)
    keys = {c['key'] for c in result['citations']}
    # Watson citation + Stick citation + convolution framework
    assert 'kaden2007' in keys
    assert 'behrens2003' in keys


# ---- Test 4: walk full NODDI-like model ----

def test_walk_noddi():
    from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
    stick = C1Stick()
    watson_stick = SD1WatsonDistributed(models=[stick])
    zeppelin = G2Zeppelin()
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[watson_stick, zeppelin, ball])
    result = walk_citation_graph(mcm)
    keys = {c['key'] for c in result['citations']}
    # Should find Watson, Stick, Zeppelin, Ball, MCM citations
    assert 'kaden2007' in keys
    assert 'behrens2003' in keys
    assert 'panagiotaki2012' in keys
    # Should find constraints
    assert len(result['constraints']) >= 1


# ---- Test 5: deduplication ----

def test_deduplication():
    """Same DOI from two models appears only once."""
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[stick, ball])
    result = walk_citation_graph(mcm)
    dois = [c['doi'] for c in result['citations'] if 'doi' in c]
    assert len(dois) == len(set(dois)), "Duplicate DOIs found"


# ---- Test 6: all models have _citations ----

@pytest.mark.parametrize("cls", ALL_ANNOTATED_CLASSES,
                         ids=[c.__name__ for c in ALL_ANNOTATED_CLASSES])
def test_all_models_have_citations(cls):
    assert hasattr(cls, '_citations'), (
        "{} is missing _citations attribute".format(cls.__name__))
    cit = cls._citations
    assert 'definition' in cit, (
        "{} _citations missing 'definition' key".format(cls.__name__))
    assert isinstance(cit['definition'], list), (
        "{} _citations['definition'] must be a list".format(cls.__name__))
    assert len(cit['definition']) >= 1, (
        "{} must have at least one definition citation".format(cls.__name__))


# ---- Test 7: all citations have DOI ----

@pytest.mark.parametrize("cls", ALL_ANNOTATED_CLASSES,
                         ids=[c.__name__ for c in ALL_ANNOTATED_CLASSES])
def test_all_citations_have_doi(cls):
    for c in cls._citations.get('definition', []):
        assert 'doi' in c, (
            "{}: citation '{}' is missing 'doi'".format(
                cls.__name__, c.get('key', '?')))
        assert c['doi'], (
            "{}: citation '{}' has empty doi".format(
                cls.__name__, c.get('key', '?')))


# ---- Test 8: generate methods section ----

def test_generate_methods_section():
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[stick, ball])
    result = walk_citation_graph(mcm)
    md = generate_methods_section(result)
    assert '## Diffusion Model' in md
    assert '### References' in md
    assert 'Behrens' in md or 'behrens' in md.lower()


# ---- Test 9: generate BibTeX ----

def test_generate_bibtex():
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[stick, ball])
    result = walk_citation_graph(mcm)
    bib = generate_bibtex(result)
    assert '@article{' in bib
    assert 'doi' in bib
    # Check it has multiple entries
    assert bib.count('@article{') >= 2


# ---- Test 10: constraints present ----

def test_constraints_present():
    """Stejskal-Tanner models have SGP constraint, GPA models have GPA."""
    # ST models
    for cls in [C2CylinderStejskalTannerApproximation,
                S2SphereStejskalTannerApproximation,
                P2PlaneStejskalTannerApproximation,
                CC2CappedCylinderStejskalTannerApproximation]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'SGP' in ids, "{} missing SGP constraint".format(cls.__name__)

    # Callaghan models have SGP but NOT long diffusion time
    for cls in [C3CylinderCallaghanApproximation,
                P3PlaneCallaghanApproximation,
                CC3CappedCylinderCallaghanApproximation]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'SGP' in ids, "{} missing SGP constraint".format(cls.__name__)
        assert 'long_diffusion_time' not in ids, (
            "{} should NOT have long_diffusion_time constraint".format(
                cls.__name__))

    # GPA models
    for cls in [C4CylinderGaussianPhaseApproximation,
                S4SphereGaussianPhaseApproximation]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'GPA' in ids, "{} missing GPA constraint".format(cls.__name__)

    # All restricted geometry models have impermeable membrane
    restricted_classes = [
        C2CylinderStejskalTannerApproximation,
        C3CylinderCallaghanApproximation,
        C4CylinderGaussianPhaseApproximation,
        S2SphereStejskalTannerApproximation,
        _S3SphereCallaghanApproximation,
        S4SphereGaussianPhaseApproximation,
        P2PlaneStejskalTannerApproximation,
        P3PlaneCallaghanApproximation,
        CC2CappedCylinderStejskalTannerApproximation,
        CC3CappedCylinderCallaghanApproximation,
    ]
    for cls in restricted_classes:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'impermeable_membrane' in ids, (
            "{} missing impermeable_membrane constraint".format(cls.__name__))

    # Watson and Bingham have single_bundle constraint
    for cls in [SD1Watson, SD2Bingham]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'single_bundle' in ids, (
            "{} missing single_bundle constraint".format(cls.__name__))

    # MCM, SM, SH have no_exchange constraint
    for cls in [MultiCompartmentModel,
                MultiCompartmentSphericalMeanModel,
                MultiCompartmentSphericalHarmonicsModel]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'no_exchange' in ids, (
            "{} missing no_exchange constraint".format(cls.__name__))

    # Cylinder models with lambda_par have gaussian_parallel + single_axon_diffusivity
    for cls in [C1Stick,
                C2CylinderStejskalTannerApproximation,
                C3CylinderCallaghanApproximation,
                C4CylinderGaussianPhaseApproximation]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'gaussian_parallel' in ids, (
            "{} missing gaussian_parallel constraint".format(cls.__name__))
        assert 'single_axon_diffusivity' in ids, (
            "{} missing single_axon_diffusivity constraint".format(cls.__name__))

    # TR models have voxel_selection_quality and macroscopic_signal_average
    for cls in [TR1IsotropicTissueResponseModel,
                TR2AnisotropicTissueResponseModel]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'voxel_selection_quality' in ids, (
            "{} missing voxel_selection_quality constraint".format(cls.__name__))
        assert 'macroscopic_signal_average' in ids, (
            "{} missing macroscopic_signal_average constraint".format(cls.__name__))

    # All CSD optimizers have SH_convergence
    for cls in [CsdTournierOptimizer, CsdCvxpyOptimizer,
                CsdPlusOptimizer, CsdOsqpOptimizer]:
        constraints = cls._validity_constraints
        ids = [c['id'] for c in constraints]
        assert 'SH_convergence' in ids, (
            "{} missing SH_convergence constraint".format(cls.__name__))


# ---- Test 11: biophysical constants all cited ----

def test_biophysical_constants_all_cited():
    for name, entry in BIOPHYSICAL_CONSTANTS.items():
        # Must have citation
        assert 'citation' in entry, (
            "Biophysical constant '{}' missing 'citation'".format(name))
        cit = entry['citation']
        assert 'doi' in cit, (
            "Biophysical constant '{}' citation missing 'doi'".format(name))
        assert 'key' in cit, (
            "Biophysical constant '{}' citation missing 'key'".format(name))
        assert cit['doi'], (
            "Biophysical constant '{}' has empty doi".format(name))

        # Must have default with value, unit, source_key, location
        assert 'default' in entry, (
            "Biophysical constant '{}' missing 'default'".format(name))
        default = entry['default']
        assert 'value' in default, (
            "Biophysical constant '{}' default missing 'value'".format(name))
        assert 'unit' in default, (
            "Biophysical constant '{}' default missing 'unit'".format(name))
        assert 'source_key' in default, (
            "Biophysical constant '{}' default missing 'source_key'".format(name))
        assert 'location' in default, (
            "Biophysical constant '{}' default missing 'location'".format(name))
        assert default['location'], (
            "Biophysical constant '{}' has empty location".format(name))

        # Alternatives must also have location
        for i, alt in enumerate(entry.get('alternatives', [])):
            assert 'location' in alt, (
                "Biophysical constant '{}' alternative[{}] missing 'location'".format(
                    name, i))
            assert 'source_key' in alt, (
                "Biophysical constant '{}' alternative[{}] missing 'source_key'".format(
                    name, i))


# ---- Test 12: no NAA in biophysical constants ----

def test_no_naa_in_constants():
    """NAA is a metabolite, not relevant to dMRI compartment modeling."""
    for name in BIOPHYSICAL_CONSTANTS:
        assert 'naa' not in name.lower(), (
            "NAA constant '{}' should not be in BIOPHYSICAL_CONSTANTS".format(name))


# ---- Test 13: methods section includes alternatives ----

def test_methods_section_alternatives():
    """When a default parameter has alternatives, the methods section should
    include a 'see also' parenthetical."""
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[stick, ball])
    result = walk_citation_graph(mcm)
    md = generate_methods_section(result)
    # Check methods section is valid markdown
    assert '## Diffusion Model' in md


# ---- Test 14: tortuosity constraint via set_tortuous_parameter ----

def test_tortuosity_constraint():
    """When set_tortuous_parameter is called, the tortuosity citation and
    constraint should appear in the citation graph."""
    from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
    stick = C1Stick()
    zeppelin = G2Zeppelin()
    watson_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
    watson_bundle.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0')
    result = walk_citation_graph(watson_bundle)
    keys = {c['key'] for c in result['citations']}
    assert 'szafer1995' in keys, "Tortuosity citation (Szafer 1995) not found"
    constraint_ids = {c['id'] for c in result['constraints']}
    assert 'tortuosity_assumption' in constraint_ids, (
        "tortuosity_assumption constraint not found")


# ---- Test 15: MT-CSD citation when S0_tissue_responses provided ----

def test_mt_csd_citation():
    """When S0_tissue_responses is provided to
    MultiCompartmentSphericalHarmonicsModel, the MT-CSD citation and
    mt_csd_s0_scaling constraint should appear in the citation graph."""
    stick = C1Stick()
    ball = G1Ball()
    from dmipy_fit.core.spherical_harmonics_framework import (
        MultiCompartmentSphericalHarmonicsModel)
    sh_model = MultiCompartmentSphericalHarmonicsModel(
        models=[stick, ball],
        S0_tissue_responses=[1.0, 3.0])
    result = walk_citation_graph(sh_model)
    keys = {c['key'] for c in result['citations']}
    assert 'jeurissen2014' in keys, (
        "MT-CSD citation (Jeurissen 2014) not found")
    constraint_ids = {c['id'] for c in result['constraints']}
    assert 'mt_csd_s0_scaling' in constraint_ids, (
        "mt_csd_s0_scaling constraint not found")


def test_no_mt_csd_citation_without_s0():
    """Without S0_tissue_responses, the mt_csd_s0_scaling constraint should
    NOT appear (the base jeurissen2014 citation is still on the class)."""
    stick = C1Stick()
    ball = G1Ball()
    from dmipy_fit.core.spherical_harmonics_framework import (
        MultiCompartmentSphericalHarmonicsModel)
    sh_model = MultiCompartmentSphericalHarmonicsModel(
        models=[stick, ball])
    result = walk_citation_graph(sh_model)
    constraint_ids = {c['id'] for c in result['constraints']}
    assert 'mt_csd_s0_scaling' not in constraint_ids, (
        "mt_csd_s0_scaling should NOT appear without S0_tissue_responses")


# ---- Test 16: MT citation on MCM and SM with S0_tissue_responses ----

def test_mt_citation_mcm():
    """MultiCompartmentModel with S0_tissue_responses should include
    mt_csd_s0_scaling constraint."""
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(
        models=[stick, ball],
        S0_tissue_responses=[1.0, 3.0])
    result = walk_citation_graph(mcm)
    constraint_ids = {c['id'] for c in result['constraints']}
    assert 'mt_csd_s0_scaling' in constraint_ids, (
        "mt_csd_s0_scaling constraint not found on MCM with S0_tissue_responses")


def test_mt_citation_sm():
    """MultiCompartmentSphericalMeanModel with S0_tissue_responses should
    include mt_csd_s0_scaling constraint."""
    stick = C1Stick()
    ball = G1Ball()
    sm_model = MultiCompartmentSphericalMeanModel(
        models=[stick, ball],
        S0_tissue_responses=[1.0, 3.0])
    result = walk_citation_graph(sm_model)
    constraint_ids = {c['id'] for c in result['constraints']}
    assert 'mt_csd_s0_scaling' in constraint_ids, (
        "mt_csd_s0_scaling constraint not found on SM with S0_tissue_responses")


# ---- Test 17: TEdDI citation when T2 parameters are fitted ----

def test_teddi_citation_when_t2_fitted():
    """When a T2 parameter is activated for optimization, the TEdDI citation
    (Veraart et al. 2018) and per_compartment_t2 constraint should appear.

    T2 is exposed by attaching a TransverseRelaxation factor via
    OccupancyGatedModel (the current relaxation architecture), which gives the
    fittable ``OccupancyGatedModel_1_T2`` parameter.
    """
    from dmipy_fit.signal_models.attenuation import (
        OccupancyGatedModel, TransverseRelaxation)
    stick = OccupancyGatedModel(C1Stick(), factors=[TransverseRelaxation()])
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[stick, ball])
    # Activate T2 fitting by setting initial guess (makes it 'fitted')
    mcm.set_initial_guess_parameter('OccupancyGatedModel_1_T2', 0.070)
    result = walk_citation_graph(mcm)
    keys = {c['key'] for c in result['citations']}
    assert 'veraart2018' in keys, (
        "TEdDI citation (Veraart 2018) not found when T2 is fitted")
    constraint_ids = {c['id'] for c in result['constraints']}
    assert 'per_compartment_t2' in constraint_ids, (
        "per_compartment_t2 constraint not found when T2 is fitted")


def test_no_teddi_citation_when_t2_passive():
    """When T2 parameters are passive (default), the TEdDI citation should
    NOT appear."""
    stick = C1Stick()
    ball = G1Ball()
    mcm = MultiCompartmentModel(models=[stick, ball])
    result = walk_citation_graph(mcm)
    constraint_ids = {c['id'] for c in result['constraints']}
    assert 'per_compartment_t2' not in constraint_ids, (
        "per_compartment_t2 should NOT appear when T2 is passive")
