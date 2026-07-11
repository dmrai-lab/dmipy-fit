from os.path import join
import os
import os as _os
import hashlib
import shutil
import numpy as np
from scipy.stats import pearsonr
import zipfile
from . import saved_acquisition_schemes

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

DATA_PATH = _os.path.dirname(__file__)

# Example slices too large for the pip wheel are hosted as GitHub Release assets and
# fetched on first use into a local cache (the dipy pattern). This keeps the wheel small
# and makes downloading the HCP-derived data an explicit user action. Override the cache
# location with the DMIPY_DATA_DIR environment variable.
_RELEASE_BASE = "https://github.com/dmrai-lab/dmipy-fit/releases/download/data-v1"
_REMOTE_FILES = {
    # relpath under DATA_PATH: (release asset name, sha256, approx MB)
    "hcp_191841/coronal_3T.nii.gz":
        ("hcp_191841_coronal_3T.nii.gz",
         "0146551082fa22876a6defcd30b8ccd06f561e5b527f7da149390266155c9f79", 11),
    "hcp_191841/coronal_7T.nii.gz":
        ("hcp_191841_coronal_7T.nii.gz",
         "25894714a1093f17697a7b82769af2135913e02b49811a4d84a620086b398b8e", 7),
    "mgh_1010/coronal_slice.nii.gz":
        ("mgh_1010_coronal_slice.nii.gz",
         "5303341d434275bdb608228c3c9695acda5df304c2f03e5df455e0f696aab16c", 14),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_dir():
    return _os.environ.get("DMIPY_DATA_DIR") or join(
        _os.path.expanduser("~"), ".dmipy", "data")


def _fetch(relpath):
    """Local path to a data file, downloaded from the release on first use.

    Resolution order: (1) bundled in the package (dev checkout / older wheels),
    (2) local cache, (3) download from the GitHub Release into the cache and verify
    its sha256. Returns the (possibly still-missing) bundled path for unknown files so
    the caller raises a clear error.
    """
    bundled = join(DATA_PATH, relpath)
    if _os.path.exists(bundled):
        return bundled
    if relpath not in _REMOTE_FILES:
        return bundled
    asset, sha, mb = _REMOTE_FILES[relpath]
    cached = join(_cache_dir(), relpath)
    if _os.path.exists(cached) and _sha256(cached) == sha:
        return cached
    _os.makedirs(_os.path.dirname(cached), exist_ok=True)
    url = f"{_RELEASE_BASE}/{asset}"
    print(f"dmipy-fit: downloading {asset} (~{mb} MB) from the data release -> {cached}")
    tmp = cached + ".part"
    with urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    got = _sha256(tmp)
    if got != sha:
        _os.remove(tmp)
        raise IOError(
            f"checksum mismatch for {asset}: expected {sha[:12]}..., got {got[:12]}...")
    _os.replace(tmp, cached)
    return cached

__all__ = [
    'hcp_191841_coronal_slice',
    'mc_reference',
    'mgh_1010_coronal_slice',
]

# Human Connectome Project data-use acknowledgement (required for redistribution
# of HCP-derived data).  Printed on load and stored in hcp_191841/NOTICE.
_HCP_ACK = (
    "Data were provided in part by the Human Connectome Project, WU-Minn "
    "Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; "
    "1U54MH091657) funded by the 16 NIH Institutes and Centers that support the "
    "NIH Blueprint for Neuroscience Research; and by the McDonnell Center for "
    "Systems Neuroscience at Washington University.")


def hcp_191841_coronal_slice(field='3T'):
    """Bundled coronal slice of HCP subject 191841 at 3T or 7T.

    A single matched coronal slice (ACPC world frame) through the body of the
    corpus callosum / corona radiata -- corpus callosum, cortico-spinal tract and
    superior-longitudinal fasciculus cross here, so it exercises crossing-fibre
    fitting and the white-matter kernel. The 3T and 7T volumes share the subject's
    anatomical frame and differ in TE.

    Parameters
    ----------
    field : {'3T', '7T'}
        Which field-strength acquisition to load.

    Returns
    -------
    scheme : PGSEAcquisitionScheme
        Carries per-measurement TE.
    data : ndarray, shape (X, 1, Z, N)
        The diffusion-weighted coronal slice (float32).
    mask : ndarray, shape (X, 1, Z)
        Brain mask for the slice (bool).
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required to load NIfTI data files. "
            "Install it with: pip install 'dmipy-fit[data]'")
    if field not in ('3T', '7T'):
        raise ValueError("field must be '3T' or '7T'")
    data_dir = join(DATA_PATH, 'hcp_191841')
    data = nib.load(_fetch(f'hcp_191841/coronal_{field}.nii.gz')).get_fdata(
        dtype=np.float32)
    mask = nib.load(
        join(data_dir, f'coronal_{field}_mask.nii.gz')).get_fdata() > 0
    scheme = saved_acquisition_schemes.hcp_191841_acquisition_scheme(field)
    print(_HCP_ACK)
    return scheme, data, mask


def mc_reference(name):
    """Load a bundled Monte Carlo reference dataset (a dict-like ``.npz``).

    These are small, pre-reduced ground-truth arrays from the dmipy-sim Monte
    Carlo engine (validated, paper-grade), shipped so the tutorials and CI can
    overlay the numerical ground truth on the analytical model **without
    re-running any walk**.  The arrays are GPU-generated from dmipy-sim in the
    development repo and shipped here as cached ground truth.

    Parameters
    ----------
    name : {'surface_relaxivity', 'crossterm', 'parity_overlap'}

    Returns
    -------
    numpy ``NpzFile`` (index by key, e.g. ``ref['B_ia_mc']``).
    """
    allowed = ('surface_relaxivity', 'crossterm', 'parity_overlap')
    if name not in allowed:
        raise ValueError(f"name must be one of {allowed}")
    return np.load(join(DATA_PATH, 'mc_reference', f'{name}.npz'),
                   allow_pickle=True)


def mgh_1010_coronal_slice():
    """Returns coronal slice of MGH-USC HCP subject 1010.

    Four-shell acquisition (b=0/1000/3000/5000 s/mm^2), 296 volumes, acquired on
    the MGH-USC Connectom scanner (300 mT/m).  The original b=10000 shell is not
    bundled (it doubled the file size and is rarely needed for the tutorials).
    Slice y=80 (mid-brain coronal), shape (140, 1, 96, 296).

    Reference: Fan et al. 2016, NeuroImage 124:1108-1114.
    https://doi.org/10.1016/j.neuroimage.2015.08.004
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required to load NIfTI data files. "
            "Install it with: pip install dmipy-fit[data]"
        )
    data_path = _fetch('mgh_1010/coronal_slice.nii.gz')
    data = nib.load(data_path).get_fdata()
    scheme = saved_acquisition_schemes.mgh_1010_acquisition_scheme()
    print("MGH-USC HCP subject 1010. Reference: Fan et al. 2016, NeuroImage.")
    return scheme, data


def panagiotaki_verdict():
    """
    Downloads and returns the example VERDICT acquisition scheme and data that
    is available at the UCL website. The data is an example of [1]_.

    Returns
    -------
    scheme: PGSEAcquisitionScheme instance,
        acquisition scheme of the challenge data.
    data_verdict: array,
        contains the DWIs for a single tumor voxel.

    References
    ----------
    .. [1] Panagiotaki, Eletheria, et al. "Noninvasive quantification of solid
        tumor microstructure using VERDICT MRI." Cancer research 74.7 (2014):
        1902-1912.
    """
    raise NotImplementedError(
        "panagiotaki_verdict() requires an external download from the Camino "
        "tutorials page that is no longer supported. The data is not bundled "
        "with dmipy-fit."
    )
