"""Standard NNLS T2-spectrum MWF recovers a synthetic two-pool myelin fraction."""
import numpy as np
import numpy.testing as npt

from dmipy_fit.white_matter.mwf import t2_spectrum_mwf


def _two_pool_decay(echo_times, mwf_true, T2_myelin=15e-3, T2_ie=80e-3):
    """Ideal 2-pool multi-echo decay: myelin water (short T2) + IE water (long T2)."""
    return (mwf_true * np.exp(-echo_times / T2_myelin)
            + (1.0 - mwf_true) * np.exp(-echo_times / T2_ie))


def test_mwf_recovers_two_pool_fraction():
    echo_times = np.arange(1, 33) * 10e-3          # 32 echoes, TE = 10 ms
    for mwf_true in (0.05, 0.10, 0.15, 0.20):
        signal = _two_pool_decay(echo_times, mwf_true)
        mwf, T2_grid, spectrum = t2_spectrum_mwf(signal, echo_times)
        npt.assert_allclose(mwf, mwf_true, atol=0.02,
                            err_msg=f"MWF {mwf} vs true {mwf_true}")
        assert spectrum.min() >= 0.0                 # non-negative spectrum


def test_mwf_zero_when_no_short_t2():
    echo_times = np.arange(1, 33) * 10e-3
    signal = np.exp(-echo_times / 80e-3)             # single long-T2 pool
    mwf, _, _ = t2_spectrum_mwf(signal, echo_times)
    assert mwf < 0.02
