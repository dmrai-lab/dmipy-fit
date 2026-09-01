"""P4PlaneGaussianPhaseApproximation — the Gaussian-phase (finite-pulse) plane model. Validated against
the exact matrix-method plane P5 (its infinite-order limit): they agree where the GPA holds (small
slab / low b) and P4 departs where the GPA breaks down (large slab / high b)."""
import numpy as np
import numpy.testing as npt

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.constants import CONSTANTS
from dmipy_fit.signal_models.plane_models import (
    P4PlaneGaussianPhaseApproximation, P5PlaneMatrixMethod)

D = CONSTANTS['water_in_axons_diffusion_constant']
GAMMA = CONSTANTS['water_gyromagnetic_ratio']
DELTA, DELTAB = 10e-3, 40e-3


def _scheme(bvals):
    return AcquisitionScheme.from_pgse(np.asarray(bvals, float),
                                       np.tile([1., 0, 0], (len(bvals), 1)), DELTA, DELTAB)


def test_p4_matches_p5_in_gpa_regime():
    "Small slab (L^2/D << delta): GPA is exact, so P4 == the exact matrix method P5 at all b."
    b = np.array([0.0, 5e8, 1e9, 2e9, 5e9])
    scheme = _scheme(b)
    p4 = P4PlaneGaussianPhaseApproximation(diameter=2e-6)(scheme)   # L^2/D = 2 ms << 10 ms
    p5 = P5PlaneMatrixMethod(diameter=2e-6)(scheme)
    npt.assert_allclose(p4, p5, atol=1e-3)


def test_p4_low_b_matches_p5_large_slab():
    "Even for a large slab (GPA marginal), P4 and P5 agree at low b (leading cumulant)."
    b = np.array([0.0, 5e8, 1e9])
    scheme = _scheme(b)
    p4 = P4PlaneGaussianPhaseApproximation(diameter=8e-6)(scheme)
    p5 = P5PlaneMatrixMethod(diameter=8e-6)(scheme)
    npt.assert_allclose(p4, p5, atol=2e-3)


def test_p4_departs_from_p5_at_high_b_large_slab():
    "Large slab + high b: the GPA over-estimates the signal; P5 (exact) attenuates more."
    scheme = _scheme([0.0, 1e10])
    p4 = P4PlaneGaussianPhaseApproximation(diameter=8e-6)(scheme)
    p5 = P5PlaneMatrixMethod(diameter=8e-6)(scheme)
    assert p4[1] > p5[1]                                  # GPA breakdown: exact model lower
    assert abs(p4[1] - p5[1]) > 1e-2                      # a real departure


def test_p4_physical_and_monotonic():
    "b0 = 1, signal in (0,1], and a thinner slab is more restricted (higher signal)."
    scheme = _scheme([0.0, 1e9, 2e9])
    thin = P4PlaneGaussianPhaseApproximation(diameter=3e-6)(scheme)
    thick = P4PlaneGaussianPhaseApproximation(diameter=10e-6)(scheme)
    npt.assert_allclose(thin[0], 1.0)
    assert np.all((thin > 0) & (thin <= 1.0 + 1e-9))
    assert np.all(thin[1:] > thick[1:])                   # smaller pore -> higher signal


def test_p4_free_limit_large_slab():
    "A very thick slab approaches free diffusion exp(-b D) at low-moderate b."
    b = np.array([0.0, 1e9])
    scheme = _scheme(b)
    E = P4PlaneGaussianPhaseApproximation(diameter=200e-6)(scheme)
    npt.assert_allclose(E, np.exp(-b * D), atol=0.03)
