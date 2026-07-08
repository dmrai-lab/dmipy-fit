import numpy as np
from os.path import join
import os
import os as _os
from ..core.acquisition_scheme import (
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_gradient_strengths,
    acquisition_scheme_from_schemefile)

_GRADIENT_TABLES_PATH = _os.path.join(_os.path.dirname(__file__), 'gradient_tables')
DATA_PATH = _os.path.dirname(__file__)


__all__ = [
    'wu_minn_hcp_acquisition_scheme',
    'hcp_191841_acquisition_scheme',
    'mgh_1010_acquisition_scheme'
]

# Per-field acquisition for the bundled HCP 191841 example slices: a within-subject
# 3T/7T pair, both resampled to the subject ACPC/T1w frame.  TE from the WU-Minn HCP
# protocol; delta/Delta are the 3T protocol values (used for both fields -- they only
# enter time-dependent models, not the diffusion/relaxation tutorials).
HCP_191841_FIELDS = {
    '3T': dict(delta=10.6e-3, Delta=43.1e-3, TE=89.5e-3),
    '7T': dict(delta=10.6e-3, Delta=43.1e-3, TE=71.2e-3),
}


def hcp_191841_acquisition_scheme(field='3T'):
    """PGSE scheme for the bundled HCP 191841 example slice (``field`` 3T or 7T).

    The returned scheme carries per-measurement ``TE`` so the transverse-relaxation
    and surface-relaxivity occupancy-gated factors work out of the box.
    """
    if field not in HCP_191841_FIELDS:
        raise ValueError(f"field must be one of {list(HCP_191841_FIELDS)}")
    cfg = HCP_191841_FIELDS[field]
    data_dir = _os.path.join(DATA_PATH, 'hcp_191841')
    bvals = np.loadtxt(_os.path.join(data_dir, f'bvals_{field}')).ravel() * 1e6
    bvecs = np.loadtxt(_os.path.join(data_dir, f'bvecs_{field}'))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    TE = np.full(bvals.shape, cfg['TE'])
    scheme = acquisition_scheme_from_bvalues(
        bvals, bvecs, delta=cfg['delta'], Delta=cfg['Delta'], TE=TE,
        b0_threshold=150e6)
    return scheme


def wu_minn_hcp_acquisition_scheme():
    "Returns PGSEAcquisitionScheme of Wu-Minn HCP project."
    _bvals = np.loadtxt(
        join(_GRADIENT_TABLES_PATH,
             'bvals_hcp_wu_minn.txt')
    ) * 1e6
    _gradient_directions = np.loadtxt(
        join(_GRADIENT_TABLES_PATH,
             'bvecs_hcp_wu_minn.txt')
    )
    _delta = 0.0106
    _Delta = 0.0431
    return acquisition_scheme_from_bvalues(
        _bvals, _gradient_directions, _delta, _Delta)


def mgh_1010_acquisition_scheme():
    """Returns PGSEAcquisitionScheme for MGH-USC HCP subject 1010.

    Four-shell acquisition (b=1000/3000/5000 s/mm^2) acquired on the MGH-USC
    Connectom scanner (300 mT/m); the original b=10000 shell is not bundled.
    Acquisition parameters from Fan et al. 2016, NeuroImage 124:1108-1114.

    delta = 12.9 ms, Delta = 21.8 ms (constant across all shells).
    """
    mgh_dir = _os.path.join(DATA_PATH, 'mgh_1010')
    _bvals = np.loadtxt(_os.path.join(mgh_dir, 'bvals.txt')) * 1e6  # s/mm^2 -> s/m^2
    _bvecs = np.loadtxt(_os.path.join(mgh_dir, 'bvecs.txt'))
    _delta = 0.0129  # s  (Fan et al. 2016)
    _Delta = 0.0218  # s
    return acquisition_scheme_from_bvalues(_bvals, _bvecs, _delta, _Delta)


def panagiotaki_verdict_acquisition_scheme():
    """Returns acquisition scheme for VERDICT tumor characterization.

    .. deprecated::
        This function previously downloaded the scheme file from an external
        Camino URL which is no longer reliable. The data is not bundled with
        dmipy-core. To use this scheme, download VC_DTIDW.scheme manually and
        load it with ``acquisition_scheme_from_schemefile(path, b0_threshold=5e6)``.
    """
    raise NotImplementedError(
        "panagiotaki_verdict_acquisition_scheme() requires an external download "
        "that is no longer supported. Download VC_DTIDW.scheme from the Camino "
        "tutorials page and load it with "
        "acquisition_scheme_from_schemefile(path, b0_threshold=5e6)."
    )
