"""Cross-engine parity: the fit MTSaturation factor <-> the dmipy-sim two-pool oracle
(fit-D of the MT staging ladder).

The wired occupancy-gated ``MTSaturation`` factor (k_f = kappa_MT*(S/V), per-measurement
W from the scheme's saturation block, two-pool steady state) must reproduce the
dmipy-sim two-pool Bloch--McConnell steady state -- the same oracle the sim's emergent
Monte-Carlo Z-spectrum is validated against (dmipy_sim tests/test_mt_zspectrum.py).  So
transitively the FIT closed form agrees with the SIM Monte-Carlo forward: the three
observables close across both public engines.

Needs a ``dmipy_sim`` carrying the MT kernel on the path (the sim mt-staging branch):
    PYTHONPATH=/path/to/dmipy-sim-mt-staging pytest .../test_mt_sim_parity.py
Skips gracefully otherwise.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.attenuation import MTSaturation
from dmipy_fit.white_matter.composition import attach_mt_saturation


def test_mtsaturation_factor_matches_sim_oracle():
    sim_mt = pytest.importorskip("dmipy_sim.mt")   # needs the sim mt-staging on path

    kappa_MT, S_over_V = 1.5e-5, 4.0e5
    k_f, k_r = kappa_MT * S_over_V, 40.0
    T1a, T1b, T2a, T2b, w1 = 1.0, 1.0, 0.08, 1e-5, 50.0
    offs = np.array([2000.0, 6000.0, 20000.0])

    # a scheme of len(offs) measurements, each with its own saturation offset
    dirs = np.tile([1.0, 0.0, 0.0], (len(offs), 1))
    sch = AcquisitionScheme.from_pgse(np.zeros(len(offs)), dirs,
                                      delta=0.01, Delta=0.03, TE=0.06)
    attach_mt_saturation(sch, offset_hz=offs, b1_hz=w1)

    # fit forward component (the actual wired factor)
    z_fit = MTSaturation(S_over_V=S_over_V).factor(
        sch, None, {}, kappa_MT=kappa_MT, T1=T1a, T2=T2a,
        dwell_time=1.0 / k_r, T2_bound=T2b, T1_bound=T1b)

    # sim two-pool oracle steady state (t >> T1 so the transient is gone)
    M0b = k_f / k_r
    z_sim = []
    for off in offs:
        dw = 2.0 * np.pi * off
        A = sim_mt.two_pool_generator(R1a=1 / T1a, R2a=1 / T2a, R1b=1 / T1b,
                                      R2b=1 / T2b, k_f=k_f, k_r=k_r, M0a=1.0,
                                      M0b=M0b, dw_a=dw, dw_b=dw, w1=2 * np.pi * w1)
        s0 = np.array([0, 0, 1.0, 0, 0, M0b, 1.0])
        z_sim.append(sim_mt.evolve_two_pool(s0, 10.0, A)[2])
    z_sim = np.array(z_sim)

    assert np.allclose(z_fit, z_sim, atol=0.02), f"fit {z_fit} vs sim {z_sim}"
