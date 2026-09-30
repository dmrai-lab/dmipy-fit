"""The batched FOD peak values are dipy's ``peak_directions`` (the oracle) on the hemisphere of ``symmetric724``,
voxel for voxel, on noisy one-to-three-fibre FODs; the neighbour table is the hemisphere's convex-hull adjacency."""
import numpy as np
import pytest

from dmipy_fit.utils.sh_basis import (fod_peak_values, positivity_hemisphere, positivity_hemisphere_neighbours,
                                      real_sh_tournier, sh_degrees)

dipy_peaks = pytest.importorskip('dipy.direction')


def _fods(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    w = rng.dirichlet([1, 1, 1], size=n) * (rng.uniform(size=(n, 3)) < [1, 0.6, 0.3])
    smooth = np.exp(-0.05 * sh_degrees(8) ** 2)
    sh = np.einsum('nk,nkc->nc', w, real_sh_tournier(8, dirs.reshape(-1, 3)).reshape(n, 3, -1)) * smooth
    return sh + rng.normal(0, 0.01, sh.shape)


def test_it_is_dipys_peak_directions():
    from dipy.data import get_sphere, HemiSphere
    from dipy.reconst.shm import real_sh_tournier as dipy_basis
    sph = get_sphere(name='symmetric724')
    hemi = HemiSphere(theta=sph.theta, phi=sph.phi)
    sh = _fods()
    got = fod_peak_values(sh, 8, max_peaks=2)
    B = dipy_basis(8, hemi.theta, hemi.phi, legacy=False)[0]
    ref = np.zeros_like(got)
    for i, c in enumerate(sh):
        _, vals, _ = dipy_peaks.peak_directions(B @ c, hemi, relative_peak_threshold=0., min_separation_angle=25)
        ref[i, :min(2, len(vals))] = vals[:2]
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)          # measured 1e-15


def test_the_neighbours_wrap_across_the_equator():
    v = positivity_hemisphere()
    nb = positivity_hemisphere_neighbours()
    cos = np.abs(np.einsum('ij,ikj->ik', v, v[nb]))
    assert cos.min() > np.cos(np.deg2rad(15))            # every neighbour is a near vertex (measured: 12.4 deg at most)
    sym = {(i, j) for i in range(len(v)) for j in nb[i]}
    assert all((j, i) in sym for i, j in sym)                             # the adjacency is symmetric


def test_torch_tensors_give_the_same_peaks():
    torch = pytest.importorskip('torch')
    sh = _fods(300)
    np.testing.assert_allclose(fod_peak_values(torch.as_tensor(sh), 8), fod_peak_values(sh, 8), rtol=1e-12)
