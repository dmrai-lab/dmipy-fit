"""Regression tests for two bugs in S4SphereGaussianPhaseApproximation.

Both were reported by a user against dmipy-fit 2.1.0 and are copy-paste
artifacts isolated to the S4 sphere (S2 and S3 were correct):

1.  A stray ``return E_sphere`` sat before the surface-relaxivity block in
    ``__call__``, making the block unreachable. ``surface_relaxivity`` was
    exposed as a fittable parameter but had zero effect on the signal.

2.  The PGSE branch of ``__call__`` fetched ``acquisition_scheme.shell_delta`` /
    ``shell_Delta``, attributes that ``SphericalMeanAcquisitionScheme`` never
    defines. Because the spherical-mean framework always calls the model with a
    ``SphericalMeanAcquisitionScheme``, any spherical-mean use of S4 raised
    ``AttributeError``. The fix reuses the per-measurement ``delta``/``Delta``
    arrays fetched two lines earlier (present on every scheme type).
"""
import numpy as np

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.core.spherical_mean_framework import (
    MultiCompartmentSphericalMeanModel,
)
from dmipy_fit.signal_models.sphere_models import (
    S4SphereGaussianPhaseApproximation,
)


def _scheme(TE=0.05):
    b = np.array([0., 2e9, 2e9, 2e9])
    g = np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    delta = np.full(4, 0.0035)
    Delta = np.full(4, 0.015)
    return acquisition_scheme_from_bvalues(b, g, delta, Delta, TE=TE)


def test_s4_surface_relaxivity_affects_signal():
    """Bug 1: changing surface_relaxivity must change the simulated signal."""
    acq = _scheme(TE=0.05)
    m = S4SphereGaussianPhaseApproximation()
    sig0 = m(acq, diameter=8e-6, surface_relaxivity=0.0)
    sig1 = m(acq, diameter=8e-6, surface_relaxivity=1e-5)
    assert not np.allclose(sig0, sig1)
    # relaxivity attenuates, so the non-b0 signal must drop
    assert np.all(sig1[1:] < sig0[1:])


def test_s4_surface_relaxivity_noop_without_TE():
    """No TE on the scheme -> relaxivity is silently ignored (unchanged)."""
    acq = _scheme(TE=None)
    m = S4SphereGaussianPhaseApproximation()
    sig0 = m(acq, diameter=8e-6, surface_relaxivity=0.0)
    sig1 = m(acq, diameter=8e-6, surface_relaxivity=1e-5)
    assert np.allclose(sig0, sig1)


def test_s4_spherical_mean_simulate_signal():
    """Bug 2: spherical-mean simulation with S4 must not raise."""
    acq = _scheme(TE=0.05)
    sph = S4SphereGaussianPhaseApproximation()
    sm = MultiCompartmentSphericalMeanModel(models=[sph])
    out = sm.simulate_signal(acq, {
        'S4SphereGaussianPhaseApproximation_1_diameter': 6e-6,
        'S4SphereGaussianPhaseApproximation_1_surface_relaxivity': 0.0,
    })
    out = np.asarray(out).ravel()
    assert out.shape[0] == acq.spherical_mean_scheme.number_of_measurements
    assert np.all(out > 0) and np.all(out <= 1.0 + 1e-9)
