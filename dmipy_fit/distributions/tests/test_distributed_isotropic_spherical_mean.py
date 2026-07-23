"""Isotropic distributed compartments in the spherical-mean framework.

A distribution wrapper (Gamma / Poisson diameter distribution, ...) around an
*isotropic* compartment (sphere/dot) has no orientation parameter, so its
`rotational_harmonics_representation` must not try to pin a (nonexistent) `mu`.
Regression for `AttributeError: 'DD1GammaDistributed' object has no attribute
'mu_param'` when a Gamma-distributed sphere is used in a
MultiCompartmentSphericalMeanModel (soma-size use case).
"""
import numpy as np

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.core.spherical_mean_framework import (
    MultiCompartmentSphericalMeanModel)
from dmipy_fit.distributions.distribute_models import DD1GammaDistributed
from dmipy_fit.signal_models import sphere_models


def _scheme():
    b = np.array([0., 1e9, 2e9, 3e9])
    g = np.tile([1., 0, 0], (4, 1))
    return acquisition_scheme_from_bvalues(b, g, np.full(4, 0.0035),
                                           np.full(4, 0.02))


def test_gamma_distributed_sphere_spherical_mean_runs():
    acq = _scheme()
    gd = DD1GammaDistributed(
        models=[sphere_models.S4SphereGaussianPhaseApproximation()])
    assert not hasattr(gd, 'mu_param')          # isotropic -> no orientation
    sm = MultiCompartmentSphericalMeanModel(models=[gd])
    params = {p: (2.0 if p.endswith('alpha') else 2e-6)
              for p in sm.parameter_names}
    out = np.asarray(sm.simulate_signal(acq, params)).ravel()
    assert out.shape[0] == acq.number_of_measurements
    assert np.all(np.isfinite(out))
    assert np.all(out > 0) and np.all(out <= 1.0 + 1e-9)
    assert out[0] == np.float64(out[0]) and np.isclose(out[0], 1.0, atol=1e-6)
