"""qMT off-resonance saturation in the WM composition (fit-C of the MT ladder).

A saturation block attached to the scheme (attach_mt_saturation) makes the free-water
compartments carry the two-pool steady-state Mz prefactor (MTSaturation).  This is the
degeneracy-LIFTING observable: with a bound pool (kappa_MT > 0) an off-resonance pulse
reduces the free-water signal (MTR > 0); a pure surface-relaxivity wall (kappa_MT = 0)
shows no off-resonance MTR.  Together with fit-B this is the fit-side "three
observables": the SAME (rho' + kappa) and (rho only) walls that give the identical
diffusion signal are separated by the Z-spectrum.

Modest B1 (qMT regime): offsets well outside the narrow free-pool line, so free-pool
direct saturation is negligible and any MTR is the bound pool.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.white_matter.composition import (
    build_white_matter_model, attach_mt_saturation)

B1, OFFSET = 50.0, 6000.0        # qMT: modest B1, far off the 2 Hz free-water line


def _scheme():
    b = np.array([0.0, 1e9])
    dirs = np.tile([1.0, 0.0, 0.0], (2, 1))
    return AcquisitionScheme.from_pgse(b, dirs, delta=0.01, Delta=0.03, TE=0.06)


def test_no_block_leaves_signal_unchanged():
    """Without a saturation block the MT model equals its no-saturation self."""
    sch = _scheme()
    model, params = build_white_matter_model(magnetization_transfer=True, kappa_MT=1.5e-5)
    S_plain = model(sch, **params)
    attach_mt_saturation(sch, offset_hz=OFFSET, b1_hz=0.0)   # b1=0 -> no saturation
    assert np.allclose(model(sch, **params), S_plain, rtol=1e-6)


def test_offresonance_mtr_needs_bound_pool():
    """Off-resonance MTR is present with a bound pool and ~absent without one."""
    sch_ref = _scheme()
    sch_sat = attach_mt_saturation(_scheme(), offset_hz=OFFSET, b1_hz=B1)

    def mtr(kappa_MT):
        m, p = build_white_matter_model(magnetization_transfer=True, kappa_MT=kappa_MT)
        return 1.0 - m(sch_sat, **p)[0] / m(sch_ref, **p)[0]     # b0 free-water MTR

    mtr_rho = mtr(0.0)                     # pure surface-relaxivity wall (no bound pool)
    mtr_mt = mtr(1.5e-5)                   # rho + MT bound pool
    assert abs(mtr_rho) < 0.01            # narrow free line spared in the qMT regime
    assert mtr_mt > 0.02                  # a real off-resonance MT dip
    assert mtr_mt > 3.0 * max(abs(mtr_rho), 1e-3)


def test_zspectrum_recovers_off_resonance():
    """Model Z-spectrum: deepest on resonance, recovers far off-resonance."""
    model, params = build_white_matter_model(magnetization_transfer=True, kappa_MT=1.5e-5)
    ref = model(attach_mt_saturation(_scheme(), OFFSET, 0.0), **params)[0]
    Z = []
    # the bound line is broad (~1/(2 pi T2b) ~ 16 kHz) with a heavy Lorentzian tail, so
    # full recovery needs a FAR offset (1 MHz) -- as in the kernel's own shape test.
    for off in (0.0, 2000.0, 6000.0, 1e6):
        s = model(attach_mt_saturation(_scheme(), off, B1), **params)[0]
        Z.append(s / ref)
    assert Z[0] < Z[1] < Z[2] < Z[3]
    assert Z[-1] == pytest.approx(1.0, abs=0.02)


def test_full_deconfounding():
    """Three observables: (rho only) and (rho'+kappa) walls with the SAME total
    reactivity are DEGENERATE in diffusion but SEPARATED by the Z-spectrum."""
    R, K = 1.5e-5, 1.0e-5
    sch_diff = _scheme()                                  # no saturation block
    m_rho, p_rho = build_white_matter_model(magnetization_transfer=True, rho2=R, kappa_MT=0.0)
    m_mt, p_mt = build_white_matter_model(magnetization_transfer=True, rho2=R - K, kappa_MT=K)
    # Panel A: identical diffusion/relaxation signal
    assert np.allclose(m_rho(sch_diff, **p_rho), m_mt(sch_diff, **p_mt), rtol=1e-6)
    # Panel B: the Z-spectrum splits them
    sch_ref = attach_mt_saturation(_scheme(), OFFSET, 0.0)
    sch_sat = attach_mt_saturation(_scheme(), OFFSET, B1)
    mtr_rho = 1.0 - m_rho(sch_sat, **p_rho)[0] / m_rho(sch_ref, **p_rho)[0]
    mtr_mt = 1.0 - m_mt(sch_sat, **p_mt)[0] / m_mt(sch_ref, **p_mt)[0]
    assert abs(mtr_rho) < 0.01 and mtr_mt > 0.02
