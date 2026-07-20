"""qMT saturation in the spherical-mean path (first-class block + SM propagation).

with_mt_saturation makes the saturation block first-class: measurements differing only
in (offset_hz, b1_hz) become distinct shells, and the block propagates to
spherical_mean_scheme, so MTSaturation applies in the spherical-mean (powder-average)
path -- not just the full/SH path.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.attenuation import OccupancyGatedModel, MTSaturation
from dmipy_fit.signal_models.cylinder_models import C1Stick

# two off-resonance offsets: 6 kHz is inside the broad bound line (MT saturates) while
# 100 kHz is beyond it (nothing saturates) -- both spare the narrow free line, so any
# difference is MT-specific (not direct free-pool saturation, which dominates on-resonance).
OFFSETS = np.array([6000.0, 100000.0])
B1, B = 50.0, 2e9
MT = dict(kappa_MT=1.5e-5, dwell_time=0.025, T2_bound=1e-5, T1_bound=1.0, T1=1.0, T2=0.05)


def _zspectrum_scheme(n_dir=6):
    """One b0 + n_dir DWI per saturation offset (b0 per shell keeps the scheme valid)."""
    rng = np.random.RandomState(0)
    d = rng.randn(n_dir, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    per_b = np.concatenate([[0.0], np.full(n_dir, B)])
    per_dir = np.vstack([[1.0, 0.0, 0.0], d])
    bvals = np.tile(per_b, len(OFFSETS))
    dirs = np.vstack([per_dir] * len(OFFSETS))
    offs = np.repeat(OFFSETS, n_dir + 1)
    sch = AcquisitionScheme.from_pgse(bvals, dirs, delta=0.005, Delta=0.025)
    sch.with_mt_saturation(offset_hz=offs, b1_hz=B1)             # first-class, chainable
    return sch


def test_block_is_first_class_and_propagates_to_spherical_mean():
    sch = _zspectrum_scheme()
    sms = sch.spherical_mean_scheme
    # distinct offsets split the shells (block entered the fingerprint): b0+DWI per offset
    assert sch.N_shells == 2 * len(OFFSETS)
    assert sms.mt_offset_hz is not None and sms.mt_b1_hz is not None
    assert set(np.round(np.unique(sms.mt_offset_hz))) == set(OFFSETS)   # per-shell offsets


def test_mt_saturation_applies_in_spherical_mean_path():
    sch = _zspectrum_scheme()
    og = OccupancyGatedModel(C1Stick(), [MTSaturation(S_over_V=4e5)])
    sms = sch.spherical_mean_scheme
    b0 = np.asarray(sms.bvalues) <= 1e6
    off = np.asarray(sms.mt_offset_hz)

    def b0_at(o):                        # the b0 shell at offset o (pure MT, no diffusion)
        return int(np.where(b0 & (np.abs(off - o) < 1.0))[0][0])

    sm = og.spherical_mean(sch, mu=[0., 0.], lambda_par=1.7e-9, **MT)
    assert float(sm[b0_at(6000.0)]) < 0.98 * float(sm[b0_at(100000.0)])   # MT dip in the SM path
    # control: kappa_MT=0 -> no bound pool -> both off-resonance offsets spare the free
    # pool equally (the dip is MT-specific, not direct free-pool saturation)
    sm0 = og.spherical_mean(sch, mu=[0., 0.], lambda_par=1.7e-9,
                            **{**MT, 'kappa_MT': 0.0})
    assert float(sm0[b0_at(6000.0)]) == pytest.approx(float(sm0[b0_at(100000.0)]), rel=0.02)
