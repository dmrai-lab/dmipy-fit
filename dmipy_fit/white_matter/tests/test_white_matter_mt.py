"""MT in the canonical white-matter composition (fit-B of the MT staging ladder).

``build_white_matter_model(magnetization_transfer=True)`` layers the MT reactivity
onto the SAME wall as surface relaxivity, so the free-water transverse factor becomes
``b_hat(rho + kappa_MT)``.  This is the DEGENERATE observable: a (rho only) wall and a
(rho' + kappa) wall with the same total reactivity produce the identical diffusion
signal -- exactly the sim-side "Panel A".  The qMT Z-spectrum (fit-C) lifts it.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.white_matter.composition import build_white_matter_model


def _scheme():
    b = np.array([0.0, 1e9, 2e9, 3e9])
    dirs = np.tile([1.0, 0.0, 0.0], (4, 1))
    return AcquisitionScheme.from_pgse(b, dirs, delta=0.01, Delta=0.03, TE=0.06)


def test_mt_model_builds_and_exposes_kappa():
    model, params = build_white_matter_model(magnetization_transfer=True, kappa_MT=1e-5)
    assert 'OccupancyGatedModel_1_kappa_MT' in params
    assert 'OccupancyGatedModel_2_kappa_MT' in params
    assert 'OccupancyGatedModel_1_kappa_MT' in model.parameter_names
    S = model(_scheme(), **params)
    assert np.all(np.isfinite(S)) and S[0] > 0.0


def test_mt_degenerate_with_surface_relaxivity():
    """b_hat(rho + kappa_MT): a rho-only wall and a (rho', kappa) wall with the same
    total reactivity give the IDENTICAL diffusion signal (the degeneracy)."""
    sch = _scheme()
    R, K = 1.5e-5, 5e-6
    m_rho, p_rho = build_white_matter_model(magnetization_transfer=True, rho2=R, kappa_MT=0.0)
    m_mt, p_mt = build_white_matter_model(magnetization_transfer=True, rho2=R - K, kappa_MT=K)
    assert np.allclose(m_rho(sch, **p_rho), m_mt(sch, **p_mt), rtol=1e-6)


def test_mt_adds_wall_attenuation():
    sch = _scheme()
    m0, p0 = build_white_matter_model(magnetization_transfer=True, rho2=1.16e-6, kappa_MT=0.0)
    m1, p1 = build_white_matter_model(magnetization_transfer=True, rho2=1.16e-6, kappa_MT=2e-5)
    S0, S1 = m0(sch, **p0), m1(sch, **p1)
    assert np.all(S1 <= S0 + 1e-12)          # MT only removes signal
    assert S1[0] < S0[0]                       # a real extra wall loss at b0


def test_no_mt_matches_plain_surface_model():
    """With kappa_MT unset the MT-enabled build reproduces the plain surface model."""
    sch = _scheme()
    m_plain, p_plain = build_white_matter_model()                       # surface only
    m_mt0, p_mt0 = build_white_matter_model(magnetization_transfer=True, kappa_MT=0.0)
    assert np.allclose(m_plain(sch, **p_plain), m_mt0(sch, **p_mt0), rtol=1e-6)
