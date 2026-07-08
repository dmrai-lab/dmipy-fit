"""Unit tests for the paper's analytical surface-relaxivity factors (surface.py).

Covers eq:b_hat_int_dist (intra Bessel-K Gamma average) and its weak limit, the
number-vs-volume weighting (the MC-validated choice), and the extra-axonal
long/short forms.
"""
import numpy as np

from dmipy_fit.white_matter.surface import (
    b_hat_ia, mean_inv_diameter_4, b_hat_ea_long, b_hat_ea_short)


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
    B = b_hat_ea_long(rho_ext=1.16e-6, S_ext_over_V_EA=4e6, tau_perp=0.04)
    assert np.isclose(B, np.exp(-1.16e-6 * 4e6 * 0.04))


def test_b_hat_ea_short_sqrt_TE():
    # short-time Mitra: exponent ~ sqrt(TE)
    B1 = b_hat_ea_short(1.16e-6, 4e6, 1.0e-9, 0.02)
    B2 = b_hat_ea_short(1.16e-6, 4e6, 1.0e-9, 0.08)
    r1, r2 = -np.log(B1), -np.log(B2)
    assert np.isclose(r2 / r1, np.sqrt(0.08 / 0.02), rtol=1e-6)   # 4x TE -> 2x exponent
