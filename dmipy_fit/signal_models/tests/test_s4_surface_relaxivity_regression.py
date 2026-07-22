"""Regression tests for surface-relaxivity handling on the bare sphere/cylinder
compartments, and the S4 spherical-mean crash.

Reported by a user against dmipy-fit 2.1.0. Two distinct problems:

1.  ``surface_relaxivity`` had been baked into the bare S2/S3/S4 spheres and
    C2/C3/C4 cylinders as a fittable parameter. That is wrong by design: bare
    signal models expose only their own diffusion parameters. Relaxivity is a
    composable, occupancy-gated *factor* (``attenuation.SurfaceRelaxivity``)
    layered on via :class:`OccupancyGatedModel` -- the only place a
    ``surface_relaxivity`` parameter should appear. (On S4 the baked-in block
    was additionally unreachable behind an early ``return`` -- a copy-paste
    artifact -- but the fix is to remove it from the bare model entirely, not
    to make it reachable.)

2.  The S4 PGSE branch fetched ``acquisition_scheme.shell_delta``/``shell_Delta``,
    attributes that ``SphericalMeanAcquisitionScheme`` never defines, so any
    spherical-mean use of S4 raised ``AttributeError``. Fixed by reusing the
    per-measurement ``delta``/``Delta`` arrays (present on every scheme type).
"""
import types

import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.core.spherical_mean_framework import (
    MultiCompartmentSphericalMeanModel,
)
from dmipy_fit.signal_models.sphere_models import (
    S2SphereStejskalTannerApproximation,
    S3SphereCallaghanApproximation,
    S4SphereGaussianPhaseApproximation,
)
from dmipy_fit.signal_models.cylinder_models import (
    C2CylinderStejskalTannerApproximation,
    C3CylinderCallaghanApproximation,
    C4CylinderGaussianPhaseApproximation,
)
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, SurfaceRelaxivity,
)

BARE_MODELS = [
    S2SphereStejskalTannerApproximation,
    S3SphereCallaghanApproximation,
    S4SphereGaussianPhaseApproximation,
    C2CylinderStejskalTannerApproximation,
    C3CylinderCallaghanApproximation,
    C4CylinderGaussianPhaseApproximation,
]


def _scheme(TE=0.05):
    b = np.array([0., 2e9, 2e9, 2e9])
    g = np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    delta = np.full(4, 0.0035)
    Delta = np.full(4, 0.015)
    return acquisition_scheme_from_bvalues(b, g, delta, Delta, TE=TE)


@pytest.mark.parametrize('model_cls', BARE_MODELS)
def test_bare_compartment_has_no_surface_relaxivity(model_cls):
    """Bug 1: bare compartments must not expose surface_relaxivity."""
    m = model_cls()
    assert 'surface_relaxivity' not in m.parameter_names
    assert 'surface_relaxivity' not in m._parameter_ranges
    assert not hasattr(m, 'surface_relaxivity')


def test_surface_relaxivity_available_via_occupancy_gated_model():
    """The supported route: relaxivity is a factor on OccupancyGatedModel, and
    it actually attenuates the signal when TE is present."""
    acq = _scheme(TE=0.05)
    gated = OccupancyGatedModel(S4SphereGaussianPhaseApproximation(),
                                [SurfaceRelaxivity()])
    assert 'surface_relaxivity' in gated.parameter_names
    sig0 = np.asarray(gated(acq, diameter=8e-6, surface_relaxivity=0.0))
    sig1 = np.asarray(gated(acq, diameter=8e-6, surface_relaxivity=20e-6))
    assert not np.allclose(sig0, sig1)
    assert np.all(sig1[1:] < sig0[1:])


def test_s4_spherical_mean_simulate_signal():
    """Bug 2: spherical-mean simulation with the bare S4 sphere must not raise."""
    acq = _scheme(TE=0.05)
    sm = MultiCompartmentSphericalMeanModel(
        models=[S4SphereGaussianPhaseApproximation()])
    out = sm.simulate_signal(acq, {
        'S4SphereGaussianPhaseApproximation_1_diameter': 6e-6,
    })
    out = np.asarray(out).ravel()
    assert out.shape[0] == acq.spherical_mean_scheme.number_of_measurements
    assert np.all(out > 0) and np.all(out <= 1.0 + 1e-9)


def test_c4_cylinder_pgse_independent_of_shell_delta():
    """C4's PGSE branch had the same shell_delta/shell_Delta reach as S4. It was
    latent (C4 is anisotropic, so it never sees a SphericalMeanAcquisitionScheme,
    and the rotational-harmonics scheme happens to carry shell_delta). Assert the
    PGSE branch now depends only on the per-measurement delta/Delta arrays: a
    scheme stub that exposes everything C4 reads *except* shell_delta must give
    the same signal as the full scheme."""
    acq = _scheme(TE=None)
    c4 = C4CylinderGaussianPhaseApproximation()
    kw = dict(mu=[0., 0.], lambda_par=1.7e-9, diameter=6e-6)
    E_full = np.asarray(c4(acq, **kw))

    # Duck-typed scheme with NO shell_delta/shell_Delta attributes.
    stub = types.SimpleNamespace(
        bvalues=acq.bvalues,
        gradient_directions=acq.gradient_directions,
        gradient_strengths=acq.gradient_strengths,
        delta=acq.delta,
        Delta=acq.Delta,
        number_of_measurements=acq.number_of_measurements,
    )
    assert not hasattr(stub, 'shell_delta')
    E_stub = np.asarray(c4(stub, **kw))
    np.testing.assert_allclose(E_stub, E_full, rtol=1e-12, atol=1e-12)
