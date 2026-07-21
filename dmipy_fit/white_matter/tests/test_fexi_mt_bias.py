"""MT biases FEXI's apparent exchange rate; the qMT Z-spectrum removes it (paper 2).

FEXI reads the apparent exchange rate (AXR) from the mixing-time (t_m) recovery of a
diffusion filter -- and t_m is a LONGITUDINAL storage period, so (as with NEXI's PGSTE
storage) MT is active during it.  For a 3-pool longitudinal model (intra, extra free
water + a shared bound pool) the filter imbalance (i-e) is an exact eigenmode with
decay rate AXR_water + k_f: the free water leaks to the bound sink at k_f and the bound
returns symmetrically, so it cannot restore the imbalance.  Hence FEXI measures
AXR_water + k_f; fitting AXR without MT is biased, and subtracting the qMT-measured k_f
recovers the true water-exchange rate.  (Verified analytically + numerically:
imbalance-decay rate = AXR_water + k_f exactly.)
"""
import numpy as np
import pytest
from scipy.optimize import least_squares

from dmipy_fit.white_matter.magnetization_transfer import z_spectrum

TM = np.linspace(0.02, 1.0, 12)                 # mixing times (s)
OFFS = np.array([1000., 2000., 4000., 8000., 16000., 32000.])   # Z-spectrum offsets (Hz)
AXR_TRUE, KF_TRUE, K_R = 1.0, 1.5, 8.0          # water AXR, MT forward, MT backward (s^-1)
T1A, T1B, T2A, T2B, W1 = 1.0, 1.0, 0.05, 1e-5, 300.0
ADCEQ, SIGMA = 0.8, 0.5                          # FEXI equilibrium ADC + filter efficiency (known)


def fexi(axr_water, k_f):
    # filter-imbalance recovery decays at the exact 3-pool eigenrate AXR_water + k_f
    return ADCEQ * (1.0 - SIGMA * np.exp(-(axr_water + k_f) * TM))


def zspec(k_f):
    return z_spectrum(OFFS, W1, k_f=k_f, k_r=K_R, T1a=T1A, T1b=T1B, T2a=T2A, T2b=T2B)


def _fit_biased(f):                              # AXR from FEXI alone (no MT term)
    return least_squares(lambda a: fexi(a[0], 0.0) - f, [1.0], bounds=(0.0, 50.0)).x[0]


def _fit_joint(f, z):                            # (AXR_water, k_f) over FEXI + Z-spectrum
    def resid(x):
        return np.concatenate([fexi(x[0], x[1]) - f, zspec(x[1]) - z])
    return least_squares(resid, [0.5, 0.5], bounds=([0., 0.], [50., 10.])).x[0]


def test_mt_biases_axr_and_zspectrum_corrects_it():
    f_true, z_true = fexi(AXR_TRUE, KF_TRUE), zspec(KF_TRUE)
    rng = np.random.RandomState(0)
    snr, N = 80.0, 12
    kb, kj = [], []
    for _ in range(N):
        f = f_true + rng.randn(len(TM)) * ADCEQ / snr
        z = z_true + rng.randn(len(OFFS)) / snr
        kb.append(_fit_biased(f))
        kj.append(_fit_joint(f, z))
    kb, kj = np.array(kb), np.array(kj)
    # FEXI alone recovers AXR_water + k_f (biased up by ~k_f), joint recovers AXR_water
    assert kb.mean() == pytest.approx(AXR_TRUE + KF_TRUE, rel=0.10)
    assert abs(kj.mean() - AXR_TRUE) < 0.20 * AXR_TRUE
    assert abs(kj.mean() - AXR_TRUE) < abs(kb.mean() - AXR_TRUE)
