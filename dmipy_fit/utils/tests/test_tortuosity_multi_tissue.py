"""Tests of the Szafer tortuosity link lambda_perp = (1 - f) * lambda_par.

With intra and extra volume fractions vi, ve and S0 responses s0i, s0e of the two compartments, the intra fraction of
the intra+extra bundle is f = vi s0e / (vi s0e + ve s0i), and f does not depend on the scale of (vi, ve) (a third
compartment takes the rest of the volume). Each expected value below is written out from that formula by hand. Values
are compared with a relative tolerance: the diffusivities are of order 1e-9 m^2/s, below any absolute tolerance of the
decimal-places kind.
"""
import numpy as np
import pytest

from dmipy_fit.utils.utils import T1_tortuosity

LPAR = 1.7e-9


def test_nested_fraction():
    # lambda_perp = (1 - 0.7) lambda_par
    np.testing.assert_allclose(T1_tortuosity()(LPAR, 0.7), 0.3 * LPAR, rtol=1e-12)


def test_two_fractions_summing_to_one():
    # f = 0.7 / (0.7 + 0.3) = 0.7
    np.testing.assert_allclose(T1_tortuosity()(LPAR, 0.7, 0.3), 0.3 * LPAR, rtol=1e-12)


@pytest.mark.parametrize("vi, ve, expected_ratio", [
    (0.7, 0.2, 2.0 / 9.0),      # 1 - 0.7/0.9 = 0.2/0.9
    (0.4, 0.35, 7.0 / 15.0),    # 0.35/0.75
    (0.1, 0.3, 0.75),           # 0.3/0.4
])
def test_fractions_summing_below_one(vi, ve, expected_ratio):
    np.testing.assert_allclose(T1_tortuosity()(LPAR, vi, ve), expected_ratio * LPAR, rtol=1e-12)


def test_fractions_scale_invariant():
    np.testing.assert_allclose(
        T1_tortuosity()(LPAR, 0.2, 0.1), T1_tortuosity()(LPAR, 0.4, 0.2), rtol=1e-12)


def test_unequal_s0_nested():
    # s0i = 3000, s0e = 4000, vi = 0.7, ve = 0.3:
    # f = 0.7*4000 / (0.7*4000 + 0.3*3000) = 2800 / 3700 = 28/37; 1 - f = 9/37
    tort = T1_tortuosity(3000., 4000.)
    np.testing.assert_allclose(tort(LPAR, 0.7, 0.3), 9.0 / 37.0 * LPAR, rtol=1e-12)
    np.testing.assert_allclose(tort(LPAR, 0.7), 9.0 / 37.0 * LPAR, rtol=1e-12)


def test_unequal_s0_with_a_third_compartment():
    # s0i = 3000, s0e = 4000, vi = 0.4, ve = 0.2 (third compartment 0.4):
    # f = 0.4*4000 / (0.4*4000 + 0.2*3000) = 1600 / 2200 = 8/11; 1 - f = 3/11
    tort = T1_tortuosity(3000., 4000.)
    np.testing.assert_allclose(tort(LPAR, 0.4, 0.2), 3.0 / 11.0 * LPAR, rtol=1e-12)


def test_unequal_s0_swapped():
    # s0i = 4000, s0e = 3000, vi = 0.25, ve = 0.25:
    # f = 0.25*3000 / (0.25*3000 + 0.25*4000) = 3/7; 1 - f = 4/7
    tort = T1_tortuosity(4000., 3000.)
    np.testing.assert_allclose(tort(LPAR, 0.25, 0.25), 4.0 / 7.0 * LPAR, rtol=1e-12)
