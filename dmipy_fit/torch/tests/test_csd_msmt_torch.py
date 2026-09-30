"""The batched multi-tissue CSD (``solver='csd_msmt_torch'``) solves the problem of ``CsdCvxpyOptimizer`` (the oracle,
CLARABEL at tight tolerances): 500 synthetic multi-shell voxels (zeppelin + ball + free water, b = 1000 / 2000 / 3000,
96 directions each) agree at 1e-5 on the coefficients and the fractions wherever cvxpy reached the optimum;
with the unity constraint too. ``fit()`` dispatches it and the fractions of noise-free voxels are the truth; the
float32 warm phase changes the solution within its accuracy; the refusals; the device problem is cached by value."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'      # the GPU when the host has one
cvxpy = pytest.importorskip('cvxpy')

from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
from dmipy_fit.optimizers_fod.csd_cvxpy import CsdCvxpyOptimizer
from dmipy_fit.signal_models.tissue_response_models import (
    estimate_TR1_isotropic_tissue_response_model, estimate_TR2_anisotropic_tissue_response_model)
from dmipy_fit.tissue_response.tests import three_tissue_phantom as ph
from dmipy_fit.torch.csd_msmt_torch import CsdMsmtTorchOptimizer, _device_problem

TIGHT = dict(solver='CLARABEL', tol_gap_abs=1e-14, tol_gap_rel=1e-14, tol_feas=1e-14, tol_ktratio=1e-10,
             max_iter=1000)


@pytest.fixture(scope="module")
def scheme():
    return ph.multishell_scheme()


@pytest.fixture(scope="module")
def responses(scheme):
    s_wm, wm = estimate_TR2_anisotropic_tissue_response_model(
        scheme, ph.S0['wm'] * ph.zeppelin(scheme, np.array([[0., 0., 1.]])))
    s_gm, gm = estimate_TR1_isotropic_tissue_response_model(scheme, ph.S0['gm'] * ph.ball(scheme, ph.D_GM)[None])
    s_csf, csf = estimate_TR1_isotropic_tissue_response_model(scheme, ph.S0['csf'] * ph.ball(scheme, ph.D_CSF)[None])
    return [s_wm, s_gm, s_csf], [wm, gm, csf]


def _prepared(scheme, responses, unity=False):
    S0s, models = responses
    model = MultiCompartmentSphericalHarmonicsModel(models, S0_tissue_responses=S0s)
    model.fit(scheme, ph.voxels(scheme, n=2)[0], solver='csd_msmt_torch', verbose=False)   # sets S0_responses
    x0 = model.parameter_initial_guess_to_parameter_vector()[None]
    return model, x0


def _unknowns(opt, P):
    """The solver's unknowns from fitted parameter vectors: the FOD coefficients, the first of each kernel being its
    fraction over 2 sqrt(pi)."""
    u = np.zeros((len(P), opt.A.shape[1]))
    u[:, opt.sh_start:opt.sh_start + opt.Ncoef] = P[:, opt._sh_slice]
    for pos, k in zip(opt._pv_positions, opt.vf_indices):
        u[:, k] = P[:, pos] / (2 * np.sqrt(np.pi))
    return u


@pytest.mark.parametrize("unity,n", [(False, 500), (True, 100)])
def test_it_is_the_cvxpy_problem(scheme, responses, unity, n):
    """Our solution is feasible and at most 1e-10 above cvxpy's objective; where cvxpy reached the same optimum
    (its objective at most 1e-11 above ours) the coefficients and fractions agree at 1e-5. The problem is flat: an
    objective gap of 5e-9 left coefficient gaps of 5e-5 where CLARABEL stopped that far above the optimum."""
    model, x0 = _prepared(scheme, responses)
    data = ph.voxels(scheme, n=n)[0] / model.max_S0_response
    x0s = np.repeat(x0, n, 0)
    opt = CsdMsmtTorchOptimizer(scheme, model, x0, unity_constraint=unity, lambda_lb=1e-5, device=DEVICE)
    got, diag = opt.fit_batch(data, x0s, diagnostics=True)
    cv = CsdCvxpyOptimizer(scheme, model, x0, unity_constraint=unity, lambda_lb=1e-5, solve_kwargs=TIGHT)
    ref, ok = [], []
    for d in data:
        ref.append(cv(d, x0[0]))
        ok.append(cv.last_status == 'optimal')
    ref, ok = np.array(ref), np.array(ok)
    Q = opt._problem['float64']['Q'].cpu().numpy()
    G = opt._problem['float64']['G'].cpu().numpy()
    ug, ur = _unknowns(opt, got), _unknowns(opt, ref)
    objective = lambda u: 0.5 * np.einsum('ni,ij,nj->n', u, Q, u) - np.einsum('ni,ni->n', u, data @ opt.A)
    gap = objective(ur) - objective(ug)
    assert (ug @ G.T).min() > -1e-12                   # feasible
    assert (-gap[ok]).max() < 1e-10, (-gap[ok]).max()  # at most 1e-10 above cvxpy: measured 1.5e-11 / 7.5e-12
    agree = ok & (gap < 1e-11)
    assert agree.mean() > 0.9, agree.mean()            # measured 0.996 / 1.0
    err = np.abs(got - ref)[agree]
    assert err.max() < 1e-5, err.max()                 # measured 5.8e-6 (500 voxels) / 2.2e-6 (unity, 100)
    pv = [model.parameter_names.index(p) for p in model.partial_volume_names]
    assert err[:, pv].max() < 1e-5                     # measured 4.9e-6 / 6.1e-7
    if unity:
        np.testing.assert_allclose(got[:, pv].sum(1), 1.0, atol=1e-10)
    assert diag['converged'].mean() > 0.9              # measured 0.954 / 0.92 within the default iterations


def _representable_voxels(model, opt, n, seed=5):
    """Noise-free voxels the model represents exactly: the kernel times a non-negative order-8 FOD (one or two
    ``(n . mu)^8`` lobes, unit integral) scaled by the WM fraction, plus the isotropic fractions; ``(data, fr)``."""
    from dmipy_fit.utils.sh_basis import real_sh_tournier
    rng = np.random.default_rng(seed)
    grid = rng.normal(size=(4000, 3))
    grid /= np.linalg.norm(grid, axis=1, keepdims=True)
    B = real_sh_tournier(8, grid)
    fr = rng.dirichlet([2., 1., 1.], size=n)
    X = np.zeros((n, opt.A.shape[1]))
    for i in range(n):
        mu = rng.normal(size=(2, 3))
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        w = rng.uniform(0.3, 0.7)
        fod = w * (grid @ mu[0]) ** 8 + (1 - w) * (grid @ mu[1]) ** 8
        c = np.linalg.lstsq(B, fod, rcond=None)[0]
        X[i, opt.sh_start:opt.sh_start + opt.Ncoef] = fr[i, 0] * c / (c[0] * 2 * np.sqrt(np.pi))
        X[i, list(opt.vf_indices[1:])] = fr[i, 1:] / (2 * np.sqrt(np.pi))
    return (X @ opt.A.T) * model.max_S0_response, fr


def test_fit_dispatches_it_and_recovers_the_fractions(scheme, responses):
    model, x0 = _prepared(scheme, responses)
    data, fr = _representable_voxels(model, model.optimizer, 40)
    fit = model.fit(scheme, data, solver='csd_msmt_torch', lambda_lb=0., verbose=False)
    got = np.stack([fit.fitted_parameters['partial_volume_%d' % i] for i in range(3)], -1)
    np.testing.assert_allclose(got, fr, atol=1e-4)      # measured 2.1e-6 on noise-free voxels
    np.testing.assert_allclose(fit.fitted_parameters['sh_coeff'][:, 0], 1 / (2 * np.sqrt(np.pi)))
    with pytest.raises(ValueError, match='maxiter'):
        model.fit(scheme, data, solver='csd', maxiter=3, verbose=False)


def test_the_float32_warm_phase_stays_within_the_accuracy(scheme, responses):
    model, x0 = _prepared(scheme, responses)
    data = ph.voxels(scheme, n=100, seed=3)[0] / model.max_S0_response
    x0s = np.repeat(x0, len(data), 0)
    cold = CsdMsmtTorchOptimizer(scheme, model, x0, warm_iter=0, device=DEVICE).fit_batch(data, x0s)
    warm = CsdMsmtTorchOptimizer(scheme, model, x0, device=DEVICE).fit_batch(data, x0s)
    assert np.abs(cold - warm).max() < 1e-5             # the solutions' accuracy; measured 1.8e-6


def test_the_refusals(scheme, responses):
    S0s, (wm, gm, csf) = responses
    model, x0 = _prepared(scheme, responses)
    model.set_fixed_parameter('partial_volume_1', 0.2)
    with pytest.raises(ValueError, match='every volume fraction'):
        model.fit(scheme, ph.voxels(scheme, n=2)[0], solver='csd_msmt_torch', verbose=False)
    single = MultiCompartmentSphericalHarmonicsModel([wm])
    with pytest.raises(ValueError, match='csd_tournier07_torch'):
        single.fit(scheme, ph.voxels(scheme, n=2)[0], solver='csd_msmt_torch', verbose=False)


def test_the_device_problem_is_cached_by_value(scheme, responses):
    model, x0 = _prepared(scheme, responses)
    _device_problem.cache_clear()
    CsdMsmtTorchOptimizer(scheme, model, x0, device=DEVICE)
    twin = ph.multishell_scheme()
    CsdMsmtTorchOptimizer(twin, model, x0, device=DEVICE)
    info = _device_problem.cache_info()
    assert (info.hits, info.misses) == (1, 1)
