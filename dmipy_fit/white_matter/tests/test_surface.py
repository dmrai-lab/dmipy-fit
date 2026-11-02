"""Unit tests for the analytical surface-relaxivity factors (dmipy_analytic.relaxation) as dmipy-fit uses them.

Covers the intra Bessel-K Gamma average and its weak limit against dmipy-sim's substrate, the number-vs-volume
weighting (the MC-validated choice), and the long-time extra-axonal form.
"""
import numpy as np

from dmipy_analytic.relaxation import (
    surface_relaxation as b_hat_ea_long,
    surface_relaxation_gamma_cylinders as b_hat_ia)
from dmipy_sim.substrate.substrate import mean_inv_diameter_4


def test_weak_limit_recovers_exp_rate():
    """For small c, B_IA -> exp(-rho <4/d> tau) with the right <4/d>."""
    alpha, beta_d, rho, tau = 2.0, 0.304e-6, 1.16e-6, 1e-4   # tiny tau -> weak
    for vw in (True, False):
        B = float(b_hat_ia(alpha, beta_d, rho, tau, volume_weighted=vw))
        sv = mean_inv_diameter_4(alpha, beta_d, volume_weighted=vw)
        assert np.isclose(B, np.exp(-rho * sv * tau), rtol=2e-3)


def test_volume_vs_number_weighting():
    """Volume weighting (alpha+2) gives 4b/(a+1); number gives 4b/(a-1)."""
    alpha, beta_d = 2.0, 0.304e-6
    beta = 1.0 / beta_d
    assert np.isclose(mean_inv_diameter_4(alpha, beta_d, False), 4*beta/(alpha-1))
    assert np.isclose(mean_inv_diameter_4(alpha, beta_d, True), 4*beta/(alpha+1))
    # volume-weighted rate is the MC-validated one (~3x below number form)
    assert (mean_inv_diameter_4(alpha, beta_d, True)
            < mean_inv_diameter_4(alpha, beta_d, False))


def test_b_hat_ia_monotone_and_bounded():
    alpha, beta_d, rho = 2.0, 0.304e-6, 1.16e-6
    taus = np.array([0.0, 0.02, 0.04, 0.08])
    B = b_hat_ia(alpha, beta_d, rho, taus)
    assert np.isclose(B[0], 1.0)                 # no relaxation at tau=0
    assert np.all(np.diff(B) < 0)                # decreases with tau
    assert np.all((B > 0) & (B <= 1.0))


def test_b_hat_ea_long_is_exponential():
    B = b_hat_ea_long(1.16e-6, 4e6, 0.04)
    assert np.isclose(B, np.exp(-1.16e-6 * 4e6 * 0.04))

