"""dipy is the oracle for the pieces dmipy-fit used to import from it: the real spherical-harmonic basis, the 724-point
sphere, the peak finder and the fod projection. The package itself never imports dipy (see
``tests/test_no_dipy_import.py``)."""
import numpy as np
import pytest

from dmipy_fit.utils import sh_basis

pytest.importorskip('dipy')
from dipy.core.sphere import HemiSphere  # noqa: E402
from dipy.data import get_sphere  # noqa: E402
from dipy.reconst.shm import real_sh_tournier as dipy_tournier  # noqa: E402
from dipy.reconst.shm import sph_harm_ind_list  # noqa: E402


def _directions(n=500, seed=1):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # the poles and the axes, where the azimuth is degenerate
    return np.concatenate([d, np.array([[0, 0, 1.], [0, 0, -1.], [1, 0, 0.], [0, 1, 0.], [-1, 0, 0.]])])


@pytest.mark.parametrize('sh_order', [0, 2, 4, 8, 14])
def test_basis_is_dipys_real_sh_tournier(sh_order):
    """Tournier/MRtrix convention (orthonormal, even degrees, m = -l..l with the sine at negative m), dipy's
    ``legacy=False``: the same numbers to rounding."""
    d = _directions()
    theta, phi = np.arccos(np.clip(d[:, 2], -1, 1)), np.arctan2(d[:, 1], d[:, 0])
    ref, _, ell = dipy_tournier(sh_order, theta, phi, legacy=False)
    np.testing.assert_allclose(sh_basis.real_sh_tournier(sh_order, d), ref, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(sh_basis.sh_degrees(sh_order), ell)
    np.testing.assert_array_equal(sh_basis.sh_degrees(sh_order), sph_harm_ind_list(sh_order)[1])


def test_stored_sphere_is_dipys_symmetric724():
    sph = get_sphere(name='symmetric724')
    hemi = HemiSphere(phi=sph.phi, theta=sph.theta)
    v = sh_basis.positivity_hemisphere()
    assert v.shape == (362, 3)
    np.testing.assert_array_equal(v, hemi.vertices)
    full = sh_basis.symmetric_sphere()
    assert full.shape == sph.vertices.shape
    # the same 724 points; only the order differs
    key = lambda a: a[np.lexsort(a.round(9).T)]
    np.testing.assert_allclose(key(full), key(sph.vertices), atol=1e-12)


def test_peak_directions_are_dipys():
    from dipy.direction import peak_directions as dipy_peaks
    sph = get_sphere(name='symmetric724')
    verts = np.asarray(sph.vertices, np.float64)
    edges = sh_basis.sphere_edges(verts)
    np.testing.assert_array_equal(edges, np.unique(np.sort(sph.edges, axis=1), axis=0))
    rng = np.random.default_rng(3)
    for n_fibres in (1, 2, 3):
        mus = rng.normal(size=(n_fibres, 3))
        mus /= np.linalg.norm(mus, axis=1, keepdims=True)
        for rel in (0.0, 0.25, 0.5):
            for _ in range(20):
                w = rng.uniform(0.3, 1.0, n_fibres)
                odf = sum(wi * np.exp(12 * (verts @ m) ** 2) for wi, m in zip(w, mus)) + rng.normal(0, 0.05, len(verts))
                dirs, vals, idx = sh_basis.peak_directions(odf, verts, edges, rel, 25)
                rdirs, rvals, ridx = dipy_peaks(odf, sph, relative_peak_threshold=rel, min_separation_angle=25)
                np.testing.assert_array_equal(idx, ridx)
                np.testing.assert_allclose(vals, rvals, rtol=0, atol=1e-12)
                np.testing.assert_allclose(dirs, rdirs, rtol=0, atol=1e-12)


def test_fod_projection_is_dipys_sh_to_sf_matrix():
    """``fod_sh`` projects with the pseudo-inverse of the basis on the symmetric sphere: dipy's ``sh_to_sf_matrix``
    inverse in the same basis (``tournier07``, non-legacy) on the same points."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix
    v = sh_basis.symmetric_sphere()
    _, inv = sh_to_sf_matrix(Sphere(xyz=v), sh_order_max=8, basis_type='tournier07', legacy=False, return_inv=True)
    np.testing.assert_allclose(np.linalg.pinv(sh_basis.real_sh_tournier(8, v)), inv.T, rtol=0, atol=1e-10)
