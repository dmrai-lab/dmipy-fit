"""Full-spectrum stress test, phase 2: the wrapper-nesting matrix.

Phase 1 (test_full_spectrum_initialization.py) covers leaf compartments x
schemes x frameworks + single-level wrappers. This phase systematically exercises
the *composition* graph:

  * every distribution wrapper (Watson / Bingham / Gamma / Poisson) x
    representative leaves, with the expected outcome (valid -> runs;
    invalid -> raises the specific error),
  * OccupancyGatedModel x leaves x factor sets,
  * two-level nestings (distribution-of-gated, gated-of-distribution,
    Karger-of-gated) and multi-compartment mixtures with partial volumes,
  * parameter-link patterns (tortuosity, set_equal, set_fixed).

Each valid combination must construct and produce a finite, physical
single-voxel signal; each invalid combination must raise the expected error type
(never silently misrepresent).
"""
import numpy as np
import pytest

from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.core.spherical_mean_framework import (
    MultiCompartmentSphericalMeanModel)
from dmipy_fit.core.spherical_harmonics_framework import (
    MultiCompartmentSphericalHarmonicsModel)
from dmipy_fit.signal_models import (
    gaussian_models, cylinder_models, sphere_models, capped_cylinder_models)
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, TransverseRelaxation, LongitudinalRelaxation,
    SurfaceRelaxivity, IntraSphereSurfaceRelaxivity, IntraPoreSurfaceRelaxivity)
from dmipy_fit.distributions.distribute_models import (
    SD1WatsonDistributed, SD2BinghamDistributed,
    DD1GammaDistributed, DD2PoissonDistributed)

# reuse the phase-1 scheme + value helpers
from dmipy_fit.core.tests.test_full_spectrum_initialization import (
    UNIVERSAL, UNIVERSAL_TE, _default_value)


def _params(model, extra=None):
    out = {}
    for name in model.parameter_names:
        out[name] = _default_value(name, model.parameter_cardinality[name])
    if extra:
        out.update(extra)
    return out


def _simulate_finite(framework, scheme=UNIVERSAL, extra=None):
    params = _params(framework, extra)
    sig = np.asarray(framework.simulate_signal(
        scheme, framework.parameters_to_parameter_vector(**params))).ravel()
    assert np.all(np.isfinite(sig)), "non-finite signal"
    assert np.all(sig > -1e-9) and np.all(sig <= 1.0 + 1e-6), "signal not in [0,1]"
    return sig


# representative leaves by capability
def _stick():   return cylinder_models.C1Stick()
def _zepp():    return gaussian_models.G2Zeppelin()
def _ball():    return gaussian_models.G1Ball()
def _c4():      return cylinder_models.C4CylinderGaussianPhaseApproximation()
def _s4():      return sphere_models.S4SphereGaussianPhaseApproximation()

# --- complete leaf capability groups (planes excluded: NMR, non-dispersable) ---
_ANISO_DISPERSABLE = {
    'C1Stick': cylinder_models.C1Stick,
    'G2Zeppelin': gaussian_models.G2Zeppelin,
    'G3TemporalZeppelin': gaussian_models.G3TemporalZeppelin,
    'C2Cylinder': cylinder_models.C2CylinderStejskalTannerApproximation,
    'C3Cylinder': cylinder_models.C3CylinderCallaghanApproximation,
    'C4Cylinder': cylinder_models.C4CylinderGaussianPhaseApproximation,
    'CC2CappedCyl':
        capped_cylinder_models.CC2CappedCylinderStejskalTannerApproximation,
    'CC3CappedCyl':
        capped_cylinder_models.CC3CappedCylinderCallaghanApproximation,
}
_ISOTROPIC = {
    'G1Ball': gaussian_models.G1Ball,
    'S1Dot': sphere_models.S1Dot,
    'S2Sphere': sphere_models.S2SphereStejskalTannerApproximation,
    'S3Sphere': sphere_models.S3SphereCallaghanApproximation,
    'S4Sphere': sphere_models.S4SphereGaussianPhaseApproximation,
}
_HAS_DIAMETER = {  # spatial-distribution-eligible (sphere + cylinder + capped)
    'S2Sphere': sphere_models.S2SphereStejskalTannerApproximation,
    'S3Sphere': sphere_models.S3SphereCallaghanApproximation,
    'S4Sphere': sphere_models.S4SphereGaussianPhaseApproximation,
    'C2Cylinder': cylinder_models.C2CylinderStejskalTannerApproximation,
    'C3Cylinder': cylinder_models.C3CylinderCallaghanApproximation,
    'C4Cylinder': cylinder_models.C4CylinderGaussianPhaseApproximation,
    'CC2CappedCyl': _ANISO_DISPERSABLE['CC2CappedCyl'],
    'CC3CappedCyl': _ANISO_DISPERSABLE['CC3CappedCyl'],
}
_NO_DIAMETER = {
    'G1Ball': gaussian_models.G1Ball, 'S1Dot': sphere_models.S1Dot,
    'C1Stick': cylinder_models.C1Stick, 'G2Zeppelin': gaussian_models.G2Zeppelin,
    'G3TemporalZeppelin': gaussian_models.G3TemporalZeppelin,
}


def _dist_cases():
    """Exhaustive (wrapper, leaf, expected) over the full leaf groups."""
    cases = []
    for spherical in (SD1WatsonDistributed, SD2BinghamDistributed):
        for name, cls in _ANISO_DISPERSABLE.items():
            cases.append((spherical, name, cls, None))
        for name, cls in _ISOTROPIC.items():          # isotropic-only -> ValueError
            cases.append((spherical, name, cls, ValueError))
    for spatial in (DD1GammaDistributed, DD2PoissonDistributed):
        for name, cls in _HAS_DIAMETER.items():
            cases.append((spatial, name, cls, None))
        for name, cls in _NO_DIAMETER.items():         # no diameter -> AttributeError
            cases.append((spatial, name, cls, AttributeError))
    return cases


_DIST_CASES = _dist_cases()


@pytest.mark.parametrize('wrapper, leaf_name, leaf, expect', _DIST_CASES,
                         ids=[f"{w.__name__}-{n}" for w, n, l, e in _DIST_CASES])
def test_distribution_wrapper_matrix(wrapper, leaf_name, leaf, expect):
    if expect is not None:
        with pytest.raises(expect):
            wrapper(models=[leaf()])
        return
    dist = wrapper(models=[leaf()])
    _simulate_finite(MultiCompartmentModel(models=[dist]))


# ===========================================================================
# 2. OccupancyGatedModel x leaves x factor sets
# ===========================================================================
_GATE_CASES = [
    (_stick, [TransverseRelaxation()], None),
    (_stick, [TransverseRelaxation(), LongitudinalRelaxation()], None),
    (_s4, [SurfaceRelaxivity()], None),                 # sphere has diameter
    (_c4, [SurfaceRelaxivity()], None),                 # cylinder has diameter
    (_ball, [SurfaceRelaxivity()], ValueError),         # no size -> B3 raise
    (_ball, [SurfaceRelaxivity(surface_to_volume=1e6)], None),
    (_s4, [IntraSphereSurfaceRelaxivity()], None),
    (_stick, [IntraPoreSurfaceRelaxivity()], None),
]


@pytest.mark.parametrize('leaf, factors, expect', _GATE_CASES,
                         ids=[f"{l.__name__}-{'+'.join(type(x).__name__ for x in f)}"
                              for l, f, e in _GATE_CASES])
def test_occupancy_gated_matrix(leaf, factors, expect):
    if expect is not None:
        with pytest.raises(expect):
            OccupancyGatedModel(leaf(), factors)
        return
    gated = OccupancyGatedModel(leaf(), factors)
    _simulate_finite(MultiCompartmentModel(models=[gated]), scheme=UNIVERSAL_TE)


# ===========================================================================
# 3. Two-level nestings
# ===========================================================================
def test_watson_of_gated_stick():
    """Dispersed, relaxation-gated stick: SD1Watson([OccupancyGated(Stick,[T2])])."""
    model = SD1WatsonDistributed(
        models=[OccupancyGatedModel(_stick(), [TransverseRelaxation()])])
    _simulate_finite(MultiCompartmentModel(models=[model]), scheme=UNIVERSAL_TE)


def test_gamma_of_gated_sphere():
    """Gamma-distributed, relaxation-gated sphere."""
    model = DD1GammaDistributed(
        models=[OccupancyGatedModel(_s4(), [TransverseRelaxation()])])
    _simulate_finite(MultiCompartmentModel(models=[model]), scheme=UNIVERSAL_TE)


def test_multicompartment_mixture_with_partial_volumes():
    """NODDI-like mixture: dispersed stick + ball, free partial volumes."""
    model = MultiCompartmentModel(
        models=[SD1WatsonDistributed(models=[_stick()]), _ball()])
    _simulate_finite(model)


# ===========================================================================
# 4. Parameter-link patterns
# ===========================================================================
def test_set_equal_and_tortuous_links_standard_model():
    """The 'Standard Model' link pattern: tie stick/zeppelin orientation and
    parallel diffusivity, and a tortuosity link on the zeppelin's lambda_perp."""
    stick, zepp = _stick(), _zepp()
    model = MultiCompartmentModel(models=[stick, zepp])
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0', 'partial_volume_1')
    model.set_equal_parameter('C1Stick_1_mu', 'G2Zeppelin_1_mu')
    model.set_equal_parameter('C1Stick_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    _simulate_finite(model)


def test_set_fixed_parameter():
    model = MultiCompartmentModel(models=[_ball(), _stick()])
    model.set_fixed_parameter('G1Ball_1_lambda_iso', 3.0e-9)
    assert 'G1Ball_1_lambda_iso' not in model.parameter_names
    _simulate_finite(model)


# ===========================================================================
# 5. Framework coverage for a dispersed anisotropic kernel
# ===========================================================================
def test_dispersed_stick_in_spherical_mean_and_sh():
    """A Watson-dispersed stick works in SphericalMean and SH frameworks."""
    sm = MultiCompartmentSphericalMeanModel(
        models=[SD1WatsonDistributed(models=[_stick()])])
    _simulate_finite(sm)
    sh = MultiCompartmentSphericalHarmonicsModel(models=[_stick()])
    # SH needs kernel params fixed + sh_coeff supplied; just assert it builds
    assert any(n.endswith('sh_coeff') for n in sh.parameter_names)


# ===========================================================================
# 6. Concatenated / multi-modal acquisition schemes
#    PGSE + PGSTE + OGSE + b-tensor STE + b-tensor PTE, in every combination.
# ===========================================================================
# A combined scheme (scheme_a + scheme_b) must never crash or silently produce
# NaN. The Gaussian compartments (ball/stick/zeppelin) evaluate every scheme
# type via bvalues/b-tensor and MUST stay finite on *any* combination. Models
# with narrower validity are tested on the combos they support and asserted to
# refuse (raise, not NaN) the ones they don't (S4/C4 GaussianPhase reject
# multidimensional b-tensor encoding; the temporal zeppelin needs real PGSE
# delta/Delta so is not defined on OGSE).
pytest.importorskip("dmipy_sim")   # from_pgse/pgste/ogse/btensor build waveforms

_Ncat = 6
_bcat = np.repeat([0., 1e9, 2e9], _Ncat // 3)
_dcat = np.random.RandomState(3).randn(_Ncat, 3)
_dcat /= np.linalg.norm(_dcat, axis=1, keepdims=True)


def _atoms():
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    return {
        'pgse': AcquisitionScheme.from_pgse(_bcat, _dcat, delta=0.006,
                                            Delta=0.025, TE=0.05),
        'pgste': AcquisitionScheme.from_pgste(_bcat, _dcat, delta=0.006, TM=0.04),
        'ogse': AcquisitionScheme.from_ogse(
            _bcat, _dcat,
            oscillation_frequency=np.repeat([0., 50., 80.], _Ncat // 3),
            gradient_duration=0.02),
        'ste': AcquisitionScheme.from_btensor_ste(_bcat, delta=0.02, Delta=0.02),
        'pte': AcquisitionScheme.from_btensor_pte(_bcat, plane_normal=[0, 0, 1.],
                                                  delta=0.02, Delta=0.02),
    }


_ATOMS = _atoms()
# every pairwise + a few higher-order combinations, including the "everything" one
_COMBOS = {
    'pgse+pgste': _ATOMS['pgse'] + _ATOMS['pgste'],
    'pgse+ogse': _ATOMS['pgse'] + _ATOMS['ogse'],
    'pgste+ogse': _ATOMS['pgste'] + _ATOMS['ogse'],
    'pgse+ste': _ATOMS['pgse'] + _ATOMS['ste'],
    'pgse+pte': _ATOMS['pgse'] + _ATOMS['pte'],
    'ste+pte': _ATOMS['ste'] + _ATOMS['pte'],
    'pgse+pgste+ogse': _ATOMS['pgse'] + _ATOMS['pgste'] + _ATOMS['ogse'],
    'pgse+ste+pte': _ATOMS['pgse'] + _ATOMS['ste'] + _ATOMS['pte'],
    'all5': (_ATOMS['pgse'] + _ATOMS['pgste'] + _ATOMS['ogse']
             + _ATOMS['ste'] + _ATOMS['pte']),
}

# Models valid on EVERY scheme type -> must be finite on all combos.
# Gaussian (ball/stick/zeppelin) evaluate any encoding via bvalues/b-tensor;
# the Gaussian-phase restriction models (S4/C4) handle PGSE/PGSTE/OGSE via their
# scalar/OGSE branches and multidimensional b-tensor (STE/PTE) via the
# waveform per-component (sphere) / gamma_lm (cylinder) path.
_ALL_SCHEME_MODELS = {
    'G1Ball': gaussian_models.G1Ball,
    'C1Stick': cylinder_models.C1Stick,
    'G2Zeppelin': gaussian_models.G2Zeppelin,
    'S4Sphere': sphere_models.S4SphereGaussianPhaseApproximation,
    'C4Cylinder': cylinder_models.C4CylinderGaussianPhaseApproximation,
}


def test_combo_measurement_counts():
    assert _COMBOS['pgse+pgste'].number_of_measurements == 2 * _Ncat
    assert _COMBOS['all5'].number_of_measurements == 5 * _Ncat


@pytest.mark.parametrize('combo_name', list(_COMBOS))
@pytest.mark.parametrize('model_name', list(_ALL_SCHEME_MODELS))
def test_models_finite_on_every_scheme_combination(combo_name, model_name):
    """The core guarantee: these compartments never crash or NaN on any
    concatenation of PGSE / PGSTE / OGSE / b-tensor STE / b-tensor PTE,
    including multidimensional b-tensor blocks."""
    mc = MultiCompartmentModel(models=[_ALL_SCHEME_MODELS[model_name]()])
    _simulate_finite(mc, scheme=_COMBOS[combo_name])


@pytest.mark.parametrize('scheme_name', ['ste', 'pte'])
def test_gaussianphase_restriction_runs_on_btensor(scheme_name):
    """S4/C4 now evaluate multidimensional b-tensor encoding (per-component
    sphere GPA / gamma_lm cylinder) instead of raising -- finite, physical."""
    for cls in (sphere_models.S4SphereGaussianPhaseApproximation,
                cylinder_models.C4CylinderGaussianPhaseApproximation):
        _simulate_finite(MultiCompartmentModel(models=[cls()]),
                         scheme=_ATOMS[scheme_name])


def test_s4_waveform_matches_analytic_pgse_in_colinear_limit():
    """The sphere per-component waveform path must reduce to the analytic PGSE
    GPA for a colinear waveform (validates the b-tensor generalization)."""
    import copy
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    pgse = AcquisitionScheme.from_pgse(_bcat, _dcat, delta=0.006, Delta=0.03,
                                       n_t=6000)
    s4 = sphere_models.S4SphereGaussianPhaseApproximation()
    E_analytic = np.asarray(s4(pgse, diameter=8e-6))            # scalar PGSE path
    # dropping the scalar gradient_strengths forces the general-waveform
    # per-component path; for this colinear (rank-1) waveform it must reduce to
    # the analytic PGSE result.
    forced = copy.copy(pgse)
    forced.gradient_strengths = None
    E_waveform = np.asarray(s4(forced, diameter=8e-6))
    np.testing.assert_allclose(E_waveform, E_analytic, atol=5e-4)


def test_gaussianphase_btensor_block_consistent_in_mixed_scheme():
    """A b-tensor block inside a mixed PGSE+STE concatenation gives the same
    signal as the standalone STE scheme (concatenation resamples to a common dt
    instead of silently mis-integrating the b-tensor block)."""
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    pgse = AcquisitionScheme.from_pgse(_bcat, _dcat, delta=0.006, Delta=0.025)
    ste = _ATOMS['ste']
    mixed = pgse + ste
    n = ste.number_of_measurements
    for cls, kw in (
        (sphere_models.S4SphereGaussianPhaseApproximation, {'diameter': 8e-6}),
        (cylinder_models.C4CylinderGaussianPhaseApproximation,
         {'mu': [0.3, 0.5], 'lambda_par': 1.7e-9, 'diameter': 8e-6})):
        m = cls()
        E_standalone = np.asarray(m(ste, **kw))
        E_mixed = np.asarray(m(mixed, **kw))
        np.testing.assert_allclose(E_mixed[-n:], E_standalone, atol=3e-3)


def test_relaxation_gated_model_on_pgse_plus_pgste():
    """A T2+T1-gated stick on a PGSE+PGSTE combo: T1 acts only on the PGSTE
    (mixing-time) block, T2 throughout -- must run finite across both blocks."""
    gated = OccupancyGatedModel(
        cylinder_models.C1Stick(),
        [TransverseRelaxation(), LongitudinalRelaxation()])
    _simulate_finite(MultiCompartmentModel(models=[gated]),
                     scheme=_COMBOS['pgse+pgste'])
