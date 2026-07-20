"""Longitudinal MT during a PGSTE mixing time (fit-E of the MT ladder).

MT is the one wall effect still active while magnetisation is stored along z: the
free water exchanges with the bound pool longitudinally, so the stimulated-echo
magnetisation carries an extra exp(-k_f * TM), k_f = kappa_MT*(S/V) -- the residual
confound for PGSTE-based permeability.  Susceptibility and surface relaxivity (and
the transverse MT factor) are gated OFF during storage, which is exactly why PGSTE is
a cleaner permeability probe -- and why the longitudinal MT term must be modelled.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.attenuation import LongitudinalMT
from dmipy_fit.white_matter.composition import build_white_matter_model
from dmipy_fit.white_matter.surface import mean_inv_diameter_4


def _pgste(TM):
    b = np.array([0.0, 1e9])
    dirs = np.tile([1.0, 0.0, 0.0], (2, 1))
    return AcquisitionScheme.from_pgste(b, dirs, delta=0.01, TM=TM, TE=0.06)


def _spin_echo():
    b = np.array([0.0, 1e9])
    dirs = np.tile([1.0, 0.0, 0.0], (2, 1))
    return AcquisitionScheme.from_pgse(b, dirs, delta=0.01, Delta=0.03, TE=0.06)


def test_factor_is_exp_kf_tm_on_pgste():
    kappa_MT, sv, TM = 1.5e-5, 4.0e5, 0.2
    val = LongitudinalMT(S_over_V=sv).factor(_pgste(TM), None, {}, kappa_MT=kappa_MT)
    assert np.allclose(val, np.exp(-kappa_MT * sv * TM))


def test_factor_inert_on_spin_echo():
    # no mixing time -> transverse-gated storage effects are off; longitudinal MT too
    val = LongitudinalMT(S_over_V=4.0e5).factor(_spin_echo(), None, {}, kappa_MT=1.5e-5)
    assert val == 1.0


def test_interior_factor_uses_inner_wall_geometry():
    kappa_MT, alpha, scale, g, TM = 1.5e-5, 2.0, 0.304e-6, 0.7, 0.15
    f = LongitudinalMT(gamma_shape=alpha, gamma_scale_outer_diameter=scale)
    val = f.factor(_pgste(TM), None, {}, kappa_MT=kappa_MT, g_ratio=g)
    sv = mean_inv_diameter_4(alpha, g * scale)            # inner diameter = g * outer
    assert np.allclose(val, np.exp(-kappa_MT * sv * TM))


def test_model_mt_attenuates_stored_signal_and_grows_with_tm():
    m0, p0 = build_white_matter_model(magnetization_transfer=True, kappa_MT=0.0)
    m1, p1 = build_white_matter_model(magnetization_transfer=True, kappa_MT=2e-5)
    r_short = m1(_pgste(0.05), **p1)[0] / m0(_pgste(0.05), **p0)[0]
    r_long = m1(_pgste(0.30), **p1)[0] / m0(_pgste(0.30), **p0)[0]
    assert r_short < 1.0 and r_long < 1.0        # MT attenuates the stored signal
    assert r_long < r_short                        # more storage time -> more MT loss
