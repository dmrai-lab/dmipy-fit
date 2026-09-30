"""The torch tensor fit is dipy's ``TensorModel`` (the oracle, float64): FA to 1e-6 relative and the principal
eigenvector to 1e-4 rad on a synthetic noisy volume, for both WLS and OLS; float32 within its measured floor;
``b_max`` selects the measurements; the setup cache is keyed by value. Measured against dipy 1.12 (float64): FA
to 9e-13 (WLS) / 5e-12 (OLS) relative, the eigenvector to 5e-8 rad."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'      # the GPU when the host has one
dti = pytest.importorskip('dipy.reconst.dti')

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues, dti_gradient_table
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.torch.dti_torch import _setup, build_dti_fitter_torch


@pytest.fixture(scope="module")
def scheme():
    return wu_minn_hcp_acquisition_scheme()


def _volume(scheme, n=2000, seed=0, sigma=0.02):
    """Tensors with random frames and eigenvalues (FA from ~0 to ~0.9), S0 = 1, Gaussian noise, positive."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    ev = np.sort(rng.uniform(0.1e-9, 2.5e-9, size=(n, 3)), axis=1)[:, ::-1]
    D = np.einsum('nij,nj,nkj->nik', q, ev, q)
    b, g = scheme.bvalues, scheme.gradient_directions
    S = np.exp(-b[None] * np.einsum('mi,nij,mj->nm', g, D, g))
    return np.abs(S + rng.normal(0, sigma, S.shape))


def _angle(u, v):
    return np.arccos(np.clip(np.abs(np.sum(u * v, -1)), 0, 1))


@pytest.mark.parametrize("method", ['WLS', 'OLS'])
def test_it_is_dipys_tensor_model(scheme, method):
    data = _volume(scheme)
    gtab, pgse = dti_gradient_table(scheme)
    ref = dti.TensorModel(gtab, fit_method=method).fit(data[:, pgse])
    got = build_dti_fitter_torch(scheme, method=method, dtype=torch.float64, device=DEVICE)(data)
    fa = got.fa.cpu().numpy()
    np.testing.assert_allclose(fa, ref.fa, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got.evals.cpu().numpy(), ref.evals * 1e-6, rtol=1e-6, atol=1e-18)
    np.testing.assert_allclose(got.md.cpu().numpy(), ref.md * 1e-6, rtol=1e-6)
    ok = ref.evals[:, 0] - ref.evals[:, 1] > 1e-5                   # a principal direction exists
    assert _angle(got.principal_direction.cpu().numpy()[ok], ref.evecs[ok, :, 0]).max() < 1e-4


def test_float32_within_its_floor(scheme):
    data = _volume(scheme)
    f64 = build_dti_fitter_torch(scheme, dtype=torch.float64, device=DEVICE)(data)
    f32 = build_dti_fitter_torch(scheme, device=DEVICE)(data.astype(np.float32))
    assert f32.fa.dtype == torch.float32
    err = np.abs(f32.fa.double().cpu().numpy() - f64.fa.cpu().numpy()).max()
    assert err < 1e-4, err                       # measured 1.8e-6 on this volume (CPU)
    ok = (f64.evals[:, 0] - f64.evals[:, 1]).cpu().numpy() > 1e-11
    ang = _angle(f32.principal_direction.double().cpu().numpy()[ok], f64.principal_direction.cpu().numpy()[ok]).max()
    assert ang < 1e-2, ang                       # measured 7.8e-4 rad (the most nearly degenerate voxel)


def test_b_max_selects_the_measurements(scheme):
    data = _volume(scheme, n=50)
    low = scheme.bvalues <= 1.1e9
    sub = acquisition_scheme_from_bvalues(scheme.bvalues[low], scheme.gradient_directions[low],
                                          scheme.delta[low], scheme.Delta[low])
    a = build_dti_fitter_torch(scheme, b_max=1.1e9, dtype=torch.float64, device=DEVICE)(data)
    b = build_dti_fitter_torch(sub, dtype=torch.float64, device=DEVICE)(data[:, low])
    np.testing.assert_allclose(a.fa.cpu().numpy(), b.fa.cpu().numpy(), rtol=1e-12)


def test_the_setup_is_cached_by_value(scheme):
    _setup.cache_clear()
    build_dti_fitter_torch(scheme, device=DEVICE)
    twin = acquisition_scheme_from_bvalues(scheme.bvalues.copy(), scheme.gradient_directions.copy(),
                                           scheme.delta, scheme.Delta)
    build_dti_fitter_torch(twin, device=DEVICE)
    info = _setup.cache_info()
    assert (info.hits, info.misses) == (1, 1)
    with pytest.raises(ValueError, match="method"):
        build_dti_fitter_torch(scheme, method='NLLS')
