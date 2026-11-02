"""The capped cylinders are even in the gradient direction: the signal of ``n`` and of ``-n`` is the same.

The attenuation along the axis is a function of the squared projection of the wave-vector on the axis (the plates'
diffraction pattern, and the Callaghan series, are even in it), so a gradient and its reverse measure the same
displacement distribution. A projection ``q (n . mu)`` that is negative must attenuate like its magnitude."""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models.capped_cylinder_models import (
    CC2CappedCylinderStejskalTannerApproximation, CC3CappedCylinderCallaghanApproximation)

MU = np.array([0.7, 1.1])
DIAMETER, LENGTH = 4e-6, 20e-6


def _directions():
    rng = np.random.default_rng(5)
    d = rng.normal(size=(12, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return np.vstack([d, [[0., 0., 1.], [1., 0., 0.]]])


def _scheme(directions):
    return acquisition_scheme_from_bvalues(np.full(len(directions), 2e9), directions, delta=0.012, Delta=0.032)


@pytest.mark.parametrize("model", [
    CC2CappedCylinderStejskalTannerApproximation(mu=MU, diameter=DIAMETER, length=LENGTH),
    CC3CappedCylinderCallaghanApproximation(mu=MU, diameter=DIAMETER, length=LENGTH),
])
def test_the_signal_of_a_gradient_and_of_its_reverse_is_the_same(model):
    directions = _directions()
    forward = model(_scheme(directions))
    reverse = model(_scheme(-directions))
    assert np.array_equal(forward, reverse)
    cos = directions @ np.array([np.sin(MU[0]) * np.cos(MU[1]), np.sin(MU[0]) * np.sin(MU[1]), np.cos(MU[0])])
    assert np.any(cos < 0) and np.any(cos > 0)                    # both signs of the axial projection are covered
