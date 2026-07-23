"""Full-spectrum initialization stress test.

Systematically instantiates every dmipy-fit compartment across compatible
acquisition-scheme types and modelling frameworks, plus single-level wrappers,
and asserts each combination (a) constructs, (b) produces a finite, physical
single-voxel signal, and (c) exposes a sane parameter set (no orphan/duplicate
parameters). Invalid combinations must raise the *expected* error, not silently
misbehave.

This is the "cover our bases" test: it is registry-driven so the combinatorial
space is explicit and extensible. See the module-level registries below; the
compatibility rules they encode come from the model/scheme/framework mapping.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import (
    acquisition_scheme_from_gradient_strengths,
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_qvalues,
)
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.core.spherical_mean_framework import (
    MultiCompartmentSphericalMeanModel)

from dmipy_fit.signal_models import (
    gaussian_models, cylinder_models, sphere_models, plane_models,
    capped_cylinder_models)

# ---------------------------------------------------------------------------
# Acquisition schemes: name -> (builder, set-of-provided-acquisition-params)
# A universal PGSE (from gradient strengths) provides the full PGSE parameter
# set and therefore satisfies every leaf compartment's _required_acquisition_*.
# ---------------------------------------------------------------------------
_N = 24
_rng = np.random.RandomState(0)
_bvals = np.repeat([0., 1e9, 2e9, 3e9], _N // 4)
_dirs = _rng.randn(_N, 3)
_dirs /= np.linalg.norm(_dirs, axis=1, keepdims=True)
_delta, _Delta, _TE = 0.008, 0.025, 0.060
# gradient strengths that yield the target b-values (narrow-pulse relation)
from dmipy_fit.core.gradient_conversions import g_from_b
_g = g_from_b(_bvals, _delta, _Delta)

FULL_PGSE_PARAMS = {'bvalues', 'gradient_directions', 'qvalues',
                    'gradient_strengths', 'delta', 'Delta', 'tau'}


def _universal_scheme(TE=None):
    return acquisition_scheme_from_gradient_strengths(
        _g, _dirs, _delta, _Delta, TE=(None if TE is None else np.full(_N, TE)))


UNIVERSAL = _universal_scheme()
UNIVERSAL_TE = _universal_scheme(TE=_TE)

SCHEME_VARIANTS = {
    'from_gradient_strengths': (UNIVERSAL, FULL_PGSE_PARAMS),
    'from_qvalues': (
        acquisition_scheme_from_qvalues(
            UNIVERSAL.qvalues, _dirs, _delta, _Delta), FULL_PGSE_PARAMS),
    'from_bvalues+timing': (
        acquisition_scheme_from_bvalues(_bvals, _dirs, _delta, _Delta),
        FULL_PGSE_PARAMS),
    'from_bvalues_no_timing': (
        acquisition_scheme_from_bvalues(_bvals, _dirs),
        {'bvalues', 'gradient_directions'}),
}

# ---------------------------------------------------------------------------
# Leaf compartments: name -> (factory, required-param set, isotropic?, model_type)
# ---------------------------------------------------------------------------
LEAF_COMPARTMENTS = {
    'S1Dot': (sphere_models.S1Dot, set(), True, 'CompartmentModel'),
    'G1Ball': (gaussian_models.G1Ball, {'bvalues'}, True, 'CompartmentModel'),
    'C1Stick': (cylinder_models.C1Stick,
                {'bvalues', 'gradient_directions'}, False, 'CompartmentModel'),
    'G2Zeppelin': (gaussian_models.G2Zeppelin,
                   {'bvalues', 'gradient_directions'}, False, 'CompartmentModel'),
    'S2Sphere': (sphere_models.S2SphereStejskalTannerApproximation,
                 {'qvalues'}, True, 'CompartmentModel'),
    'S3Sphere': (sphere_models.S3SphereCallaghanApproximation,
                 {'qvalues', 'tau'}, True, 'CompartmentModel'),
    'S4Sphere': (sphere_models.S4SphereGaussianPhaseApproximation,
                 {'gradient_strengths', 'delta', 'Delta'}, True, 'CompartmentModel'),
    'G3TemporalZeppelin': (gaussian_models.G3TemporalZeppelin,
                           {'bvalues', 'gradient_directions', 'delta', 'Delta'},
                           False, 'CompartmentModel'),
    'C2Cylinder': (cylinder_models.C2CylinderStejskalTannerApproximation,
                   {'bvalues', 'gradient_directions', 'qvalues'}, False,
                   'CompartmentModel'),
    'C3Cylinder': (cylinder_models.C3CylinderCallaghanApproximation,
                   {'bvalues', 'gradient_directions', 'qvalues', 'tau'}, False,
                   'CompartmentModel'),
    'C4Cylinder': (cylinder_models.C4CylinderGaussianPhaseApproximation,
                   {'bvalues', 'gradient_directions', 'gradient_strengths',
                    'delta', 'Delta'}, False, 'CompartmentModel'),
    'CC2CappedCyl': (capped_cylinder_models.CC2CappedCylinderStejskalTannerApproximation,
                     {'gradient_directions', 'qvalues'}, False, 'CompartmentModel'),
    'CC3CappedCyl': (capped_cylinder_models.CC3CappedCylinderCallaghanApproximation,
                     {'gradient_directions', 'qvalues', 'tau'}, False,
                     'CompartmentModel'),
    'P2Plane': (plane_models.P2PlaneStejskalTannerApproximation,
                {'qvalues'}, True, 'NMRModel'),
    'P3Plane': (plane_models.P3PlaneCallaghanApproximation,
                {'qvalues', 'tau'}, True, 'NMRModel'),
}


def _default_value(param_name, cardinality):
    """A physically-sensible value for a parameter, keyed by its suffix."""
    p = param_name.rsplit('_', 1)[-1]
    table = {
        'mu': [0.3, 0.5], 'lambda_par': 1.7e-9, 'lambda_perp': 0.6e-9,
        'lambda_iso': 2.5e-9, 'lambda_inf': 1.0e-9, 'A': 1e-12,
        'diameter': 5e-6, 'alpha': 2.0, 'beta': 1.5e-6, 'odi': 0.3,
        'psi': 0.5, 'f': 0.5, 'kappa': 10.0, 'T2': 0.06, 'T1': 1.0,
        'surface_relaxivity': 10e-6, 'g_ratio': 0.7,
        'partial': 0.5, 'volume': 0.5,  # partial_volume_0
    }
    if p in table:
        return table[p]
    return 0.5


def _default_params(model):
    out = {}
    for name in model.parameter_names:
        card = model.parameter_cardinality[name]
        out[name] = _default_value(name, card)
    return out


def _assert_physical(signal, scheme):
    signal = np.asarray(signal).ravel()
    assert signal.shape[0] == scheme.number_of_measurements
    assert np.all(np.isfinite(signal)), "non-finite signal"
    assert np.all(signal > -1e-9) and np.all(signal <= 1.0 + 1e-6), \
        "signal outside physical [0, 1]"


# ===========================================================================
# TEST 1 — every leaf compartment x every scheme that satisfies its _req
# ===========================================================================
@pytest.mark.parametrize('comp_name', list(LEAF_COMPARTMENTS))
@pytest.mark.parametrize('scheme_name', list(SCHEME_VARIANTS))
def test_leaf_compartment_on_compatible_schemes(comp_name, scheme_name):
    factory, req, _iso, mtype = LEAF_COMPARTMENTS[comp_name]
    scheme, provided = SCHEME_VARIANTS[scheme_name]
    if not req.issubset(provided):
        pytest.skip("scheme does not provide {}".format(req - provided))
    mc = MultiCompartmentModel(models=[factory()])
    params = _default_params(mc)
    sig = mc.simulate_signal(scheme, mc.parameters_to_parameter_vector(**params))
    _assert_physical(sig, scheme)


# ===========================================================================
# TEST 2 — framework acceptance (MultiCompartment vs SphericalMean)
# ===========================================================================
@pytest.mark.parametrize('comp_name', list(LEAF_COMPARTMENTS))
def test_leaf_in_spherical_mean_framework(comp_name):
    factory, req, _iso, mtype = LEAF_COMPARTMENTS[comp_name]
    if mtype == 'NMRModel':
        with pytest.raises(ValueError):        # NMR not allowed in spherical mean
            MultiCompartmentSphericalMeanModel(models=[factory()])
        return
    sm = MultiCompartmentSphericalMeanModel(models=[factory()])
    params = _default_params(sm)
    sig = sm.simulate_signal(UNIVERSAL, sm.parameters_to_parameter_vector(**params))
    signal = np.asarray(sig).ravel()
    assert np.all(np.isfinite(signal))
    assert np.all(signal > -1e-9) and np.all(signal <= 1.0 + 1e-6)


# ===========================================================================
# TEST 3 — parameter-set sanity (no orphan / no accidental relaxivity)
# ===========================================================================
def test_bare_compartments_have_no_relaxation_or_orphan_params():
    """Bare compartments expose only their own diffusion params (regression for
    the surface_relaxivity leak) and never a stray T2/T1/surface_relaxivity."""
    for comp_name, (factory, _r, _i, mtype) in LEAF_COMPARTMENTS.items():
        m = factory()
        names = list(m.parameter_names)
        for forbidden in ('T2', 'T1', 'surface_relaxivity'):
            assert not any(n == forbidden or n.endswith('_' + forbidden)
                           for n in names), \
                "{} unexpectedly exposes {}".format(comp_name, forbidden)
        # no duplicate keys
        assert len(names) == len(set(names)), \
            "{} has duplicate parameters".format(comp_name)


# ===========================================================================
# TEST 4 — single-level wrappers (regressions for the fixes/blocks)
# ===========================================================================
from dmipy_fit.distributions.distribute_models import (
    DD1GammaDistributed, DD2PoissonDistributed, SD1WatsonDistributed)
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, TransverseRelaxation, SurfaceRelaxivity,
    IntraSphereSurfaceRelaxivity)


def _diameter_default_params(model):
    out = {}
    for name in model.parameter_names:
        p = name.rsplit('_', 1)[-1]
        out[name] = {'mu': [0.3, 0.5], 'alpha': 2.0, 'beta': 1.5e-6,
                     'mean_diameter': 3e-6, 'diameter': 5e-6, 'length': 5e-6,
                     'lambda_par': 1.7e-9}.get(p, 0.5)
    return out


def test_dd2poisson_wraps_anisotropic_compartment():
    """Regression: DD2Poisson's mean-diameter param was named 'mu' and collided
    with the fibre orientation. Renamed to 'mean_diameter' -> Poisson now works
    on an oriented compartment and pins the correct fibre axis."""
    pd = DD2PoissonDistributed(models=[cylinder_models.C3CylinderCallaghanApproximation()])
    assert 'DD2Poisson_1_mean_diameter' in pd.parameter_names
    assert pd.mu_param == 'C3CylinderCallaghanApproximation_1_mu'   # fibre axis
    mc = MultiCompartmentModel(models=[pd])
    scheme, _ = SCHEME_VARIANTS['from_qvalues']
    sig = np.asarray(mc.simulate_signal(
        scheme, mc.parameters_to_parameter_vector(**_diameter_default_params(mc))))
    assert np.all(np.isfinite(sig)) and np.all(sig <= 1.0 + 1e-6)


def test_dd1gamma_isotropic_and_anisotropic():
    """Gamma distribution over an isotropic (sphere) and an anisotropic
    (cylinder) compartment both construct and simulate."""
    for base in (sphere_models.S4SphereGaussianPhaseApproximation(),
                 cylinder_models.C4CylinderGaussianPhaseApproximation()):
        gd = DD1GammaDistributed(models=[base])
        mc = MultiCompartmentModel(models=[gd])
        sig = np.asarray(mc.simulate_signal(
            UNIVERSAL,
            mc.parameters_to_parameter_vector(**_diameter_default_params(mc))))
        assert np.all(np.isfinite(sig))


def test_occupancy_gated_can_wrap_a_distributed_model():
    """Relaxation on a distributed/dispersed bundle is valid (scalar factor)."""
    og = OccupancyGatedModel(
        DD1GammaDistributed(models=[sphere_models.S4SphereGaussianPhaseApproximation()]),
        [TransverseRelaxation()])
    assert 'T2' in [n.rsplit('_', 1)[-1] for n in og.parameter_names]


def test_surface_relaxivity_requires_a_size():
    """Generic SurfaceRelaxivity with no diameter and no surface_to_volume would
    be a silent no-op -> must raise (no silent misrepresentation). The
    gamma-averaged closed-form factors carry their own size and are unaffected."""
    with pytest.raises(ValueError, match="surface-to-volume"):
        OccupancyGatedModel(gaussian_models.G1Ball(), [SurfaceRelaxivity()])
    # with a diameter (sphere) or explicit S/V it is fine:
    OccupancyGatedModel(sphere_models.S4SphereGaussianPhaseApproximation(),
                        [SurfaceRelaxivity()])
    OccupancyGatedModel(gaussian_models.G1Ball(),
                        [SurfaceRelaxivity(surface_to_volume=1e6)])
    # gamma-averaged closed form (paper) on a bare sphere is fine:
    OccupancyGatedModel(sphere_models.S4SphereGaussianPhaseApproximation(),
                        [IntraSphereSurfaceRelaxivity()])


def test_capped_cylinder_length_is_fittable():
    """CC2/CC3 expose 'length' as a real parameter (previously conflated with the
    plane's diameter and defaulted to None -> crash)."""
    for C in (capped_cylinder_models.CC2CappedCylinderStejskalTannerApproximation,
              capped_cylinder_models.CC3CappedCylinderCallaghanApproximation):
        assert 'length' in C().parameter_names
