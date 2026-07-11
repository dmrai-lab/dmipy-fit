import pytest

nibabel = pytest.importorskip("nibabel", reason="nibabel optional; install dmipy[data]")

import numpy as np

from dmipy_fit.data.saved_data import (
    panagiotaki_verdict,
    hcp_191841_coronal_slice,
    mc_reference,
)


def test_panagiotaki_verdict_raises():
    with pytest.raises(NotImplementedError):
        panagiotaki_verdict()


@pytest.mark.parametrize("field,n_meas", [("3T", 288), ("7T", 143)])
def test_hcp_191841_coronal_slice(field, n_meas):
    scheme, data, mask = hcp_191841_coronal_slice(field=field)
    assert data.ndim == 4 and data.shape[1] == 1          # (X, 1, Z, N)
    assert data.shape[-1] == n_meas == len(scheme.bvalues)
    assert mask.shape == data.shape[:3]
    assert np.allclose(np.unique(scheme.TE), scheme.TE[0])  # single TE per field
    assert mask.sum() > 1000


def test_mc_reference_arrays():
    for name in ("surface_relaxivity", "crossterm", "parity_overlap"):
        ref = mc_reference(name)
        assert len(ref.files) > 0
    with pytest.raises(ValueError):
        mc_reference("does_not_exist")
