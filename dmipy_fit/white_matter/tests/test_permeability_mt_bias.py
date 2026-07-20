"""MT biases PGSTE permeability, and the qMT-measured k_f removes the bias (paper 2).

A Karger two-compartment exchange (X0GeneralizedKarger) on a PGSTE acquisition encodes
membrane permeability in the mixing-time (TM) exchange-attenuation S(kappa)/S(0).  MT
adds an extra exp(-k_f*TM) over the SAME TM (the one effect still active during
longitudinal storage), so fitting the exchange rate WITHOUT an MT term absorbs k_f into
kappa and over-estimates permeability.  Supplying k_f (measured independently by the
qMT Z-spectrum) removes the bias -- the paper-2 headline.
"""
import numpy as np
import pytest
from scipy.optimize import least_squares

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin

MODEL = X0GeneralizedKarger(C1Stick(), G2Zeppelin())
TMS = np.linspace(0.02, 0.30, 10)
F, DPAR, DPERP, B = 0.5, 1.7e-9, 0.6e-9, 2e9
KAPPA_TRUE, K_F = 20.0, 0.4          # exchange rate (s^-1); MT forward rate (s^-1)

# one PGSTE scheme per TM (n_t high -> smooth gradient calibration)
_SCHEMES = [AcquisitionScheme.from_pgste(np.array([B]), np.array([[1., 0., 0.]]),
                                         delta=5e-3, TM=float(TM), TE=0.06, n_t=6000)
            for TM in TMS]


def _karger(sch, kappa):
    return float(MODEL(sch, kappa=kappa, f=F, mu=[0., 0.], C1Stick_1_lambda_par=DPAR,
                       G2Zeppelin_1_lambda_par=DPAR, G2Zeppelin_1_lambda_perp=DPERP)[0])


_S0 = np.array([_karger(s, 0.0) for s in _SCHEMES])


def _exchange_ratio(kappa):
    return np.array([_karger(s, kappa) for s in _SCHEMES]) / _S0


def _fit_kappa(resid):
    return float(least_squares(resid, x0=[10.0], bounds=(0.0, 200.0)).x[0])


def test_mt_biases_permeability_and_correction_recovers_it():
    mt_factor = np.exp(-K_F * TMS)
    measured = _exchange_ratio(KAPPA_TRUE) * mt_factor       # PGSTE data with the MT confound

    kappa_biased = _fit_kappa(lambda k: _exchange_ratio(k[0]) - measured)             # ignore MT
    kappa_corr   = _fit_kappa(lambda k: _exchange_ratio(k[0]) * mt_factor - measured)  # k_f from qMT

    assert kappa_biased > 1.3 * KAPPA_TRUE                   # MT inflates the exchange estimate
    assert abs(kappa_corr - KAPPA_TRUE) < 0.05 * KAPPA_TRUE  # correction recovers truth
