"""Joint NEXI + MT + Z-spectrum fit de-biases PGSTE permeability under noise (paper 2).

A single synthetic dataset combines PGSTE mixing-time measurements (Karger exchange x
the longitudinal MT factor exp(-k_f*TM)) with a qMT Z-spectrum (which constrains k_f).
Fitting the exchange rate on the PGSTE data ALONE (no MT term) is biased; the JOINT fit
of (kappa_exchange, k_f) over PGSTE + Z-spectrum recovers the true exchange rate -- with
noise, averaged over realisations.
"""
import numpy as np
import pytest
from scipy.optimize import least_squares

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
from dmipy_fit.white_matter.magnetization_transfer import z_spectrum

MODEL = X0GeneralizedKarger(C1Stick(), G2Zeppelin())
TMS = np.linspace(0.02, 0.30, 8)
OFFS = np.array([1000., 2000., 4000., 8000., 16000., 32000.])
KAPPA_TRUE, KF_TRUE, K_R = 20.0, 0.4, 8.0
T1A, T1B, T2A, T2B, W1 = 1.0, 1.0, 0.05, 1e-5, 300.0
_KW = dict(f=0.5, mu=[0., 0.], C1Stick_1_lambda_par=1.7e-9,
           G2Zeppelin_1_lambda_par=1.7e-9, G2Zeppelin_1_lambda_perp=0.6e-9)
_SCHEMES = [AcquisitionScheme.from_pgste(np.array([2e9]), np.array([[1., 0., 0.]]),
                                         delta=5e-3, TM=float(t), TE=0.06, n_t=6000)
            for t in TMS]
_S0 = np.array([float(MODEL(s, kappa=0.0, **_KW)[0]) for s in _SCHEMES])


def _exch(kappa):
    return np.array([float(MODEL(s, kappa=kappa, **_KW)[0]) for s in _SCHEMES]) / _S0


def _zspec(k_f):
    return z_spectrum(OFFS, W1, k_f=k_f, k_r=K_R, T1a=T1A, T1b=T1B, T2a=T2A, T2b=T2B)


def _fit_biased(pg):                     # exchange only, PGSTE alone, no MT term
    return least_squares(lambda k: _exch(k[0]) - pg, [10.0], bounds=(0.0, 200.0)).x[0]


def _fit_joint(pg, z):                   # (kappa, k_f) over PGSTE + Z-spectrum
    def resid(x):
        return np.concatenate([_exch(x[0]) * np.exp(-x[1] * TMS) - pg, _zspec(x[1]) - z])
    return least_squares(resid, [10.0, 0.1], bounds=([0.0, 0.0], [200.0, 5.0])).x[0]


def test_joint_fit_debiases_permeability_under_noise():
    pg_true = _exch(KAPPA_TRUE) * np.exp(-KF_TRUE * TMS)
    z_true = _zspec(KF_TRUE)
    rng = np.random.RandomState(0)
    snr, N = 60.0, 12
    kb, kj = [], []
    for _ in range(N):
        pg = pg_true + rng.randn(len(TMS)) / snr
        z = z_true + rng.randn(len(OFFS)) / snr
        kb.append(_fit_biased(pg))
        kj.append(_fit_joint(pg, z))
    kb, kj = np.array(kb), np.array(kj)
    assert kb.mean() > 1.2 * KAPPA_TRUE                       # PGSTE-alone is biased high
    assert abs(kj.mean() - KAPPA_TRUE) < 0.15 * KAPPA_TRUE    # joint recovers the truth
    assert abs(kj.mean() - KAPPA_TRUE) < abs(kb.mean() - KAPPA_TRUE)   # and beats the biased fit
