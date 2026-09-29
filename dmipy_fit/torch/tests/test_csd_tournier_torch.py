"""The torch Tournier solver (dmipy-fit#37) is the JAX one: the same coefficients to the float32 arithmetic and the
same active-set path (identical iteration counts) on dispersed sticks; the numpy reference voxel for voxel;
chunking and padding change nothing; a zero signal; max_iter; fit() dispatches it; TF32 flags are restored."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

torch = pytest.importorskip('torch')
jax = pytest.importorskip('jax')

from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.distributions import distribute_models
from dmipy_fit.jax.csd_tournier_jax import CsdTournierJaxOptimizer
from dmipy_fit.optimizers_fod.csd_tournier import CsdTournierOptimizer
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.torch.csd_tournier_torch import CsdTournierTorchOptimizer, full_precision

LAMBDA_PAR = 1.7e-9


def _model(scheme):
    mc = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick()])
    mc.set_fixed_parameter('C1Stick_1_lambda_par', LAMBDA_PAR)
    mc.scheme = scheme
    mc._check_if_kernel_parameters_are_fixed()
    mc.S0_responses = np.ones(1)
    return mc


def _signals(scheme, n=12, seed=0):
    rng = np.random.default_rng(seed)
    stick = distribute_models.SD1WatsonDistributed([C1Stick()])
    out = []
    for i in range(n):
        mu = rng.uniform([0, 0], [np.pi, 2 * np.pi])
        s = stick(scheme, SD1Watson_1_mu=mu, SD1Watson_1_odi=rng.uniform(0.05, 0.5), C1Stick_1_lambda_par=LAMBDA_PAR)
        out.append(s + rng.normal(0, 0.01, s.shape))
    return np.array(out)


@pytest.fixture(scope="module")
def scheme():
    return wu_minn_hcp_acquisition_scheme()


@pytest.mark.parametrize("unity", [True, False])
def test_it_is_the_jax_solver_and_the_numpy_reference(scheme, unity):
    mc = _model(scheme)
    data = _signals(scheme)
    x0 = np.tile(mc.parameter_initial_guess_to_parameter_vector(), (len(data), 1))
    ref = CsdTournierOptimizer(scheme, mc, x0, sh_order=8, unity_constraint=unity, lambda_lb=1e-5)
    jx = CsdTournierJaxOptimizer(scheme, mc, x0, sh_order=8, unity_constraint=unity, lambda_lb=1e-5)
    tc = CsdTournierTorchOptimizer(scheme, mc, x0, sh_order=8, unity_constraint=unity, lambda_lb=1e-5, device="cpu")
    expected = np.array([ref(d, x) for d, x in zip(data, x0)])
    got_j, diag_j = jx.fit_batch(data, x0, diagnostics=True)
    got_t, diag_t = tc.fit_batch(data, x0, diagnostics=True)
    assert_allclose(got_t, expected, rtol=1e-4, atol=2e-5)           # the reference's float64 against float32 solves
    np.testing.assert_array_equal(diag_t['iter_num'], diag_j['iter_num'])   # the same active-set path, voxel for voxel
    err = np.abs(got_t - got_j).max()
    assert err < 5e-6, err                                             # two float32 solvers of the same systems: measured 3.4e-7 (CPU)
    if unity:
        assert_allclose(got_t[:, tc._sh_slice][:, 0], 1.0 / (2.0 * np.sqrt(np.pi)))


def test_chunking_padding_and_one_voxel_change_nothing(scheme, monkeypatch):
    mc = _model(scheme)
    data = _signals(scheme, n=7)
    x0 = np.tile(mc.parameter_initial_guess_to_parameter_vector(), (7, 1))
    tc = CsdTournierTorchOptimizer(scheme, mc, x0, sh_order=8, lambda_lb=1e-5, device="cpu")
    whole = tc.fit_batch(data, x0)
    monkeypatch.setenv("DMIPY_CSD_TOURNIER_BATCH", "3")
    assert_allclose(tc.fit_batch(data, x0), whole, rtol=1e-5, atol=1e-6)
    for i in range(3):
        assert_allclose(tc(data[i], x0[i]), whole[i], rtol=1e-5, atol=1e-6)


def test_a_zero_signal_and_max_iter(scheme):
    mc = _model(scheme)
    x0 = mc.parameter_initial_guess_to_parameter_vector()[None]
    tc = CsdTournierTorchOptimizer(scheme, mc, x0, sh_order=8, unity_constraint=False, lambda_lb=1e-5, device="cpu")
    out, diag = tc.fit_batch(np.zeros((1, scheme.number_of_measurements)), x0, diagnostics=True)
    assert np.all(out[0, tc._sh_slice] == 0.0) and diag['iter_num'][0] == 1
    data = _signals(scheme, n=4); x0 = np.tile(x0, (4, 1))
    capped = CsdTournierTorchOptimizer(scheme, mc, x0, sh_order=8, lambda_lb=1e-5, max_iter=1, device="cpu")
    _, diag = capped.fit_batch(data, x0, diagnostics=True)
    assert (diag['iter_num'] == 1).all()


def test_fit_dispatches_the_torch_solver_and_restores_the_precision_flags(scheme):
    data = _signals(scheme, n=5)
    mc_j = MultiCompartmentSphericalHarmonicsModel([C1Stick()]); mc_j.set_fixed_parameter('C1Stick_1_lambda_par', LAMBDA_PAR)
    mc_t = MultiCompartmentSphericalHarmonicsModel([C1Stick()]); mc_t.set_fixed_parameter('C1Stick_1_lambda_par', LAMBDA_PAR)
    ref = mc_j.fit(scheme, data, solver='csd_tournier07_jax', verbose=False)
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    got = mc_t.fit(scheme, data, solver='csd_tournier07_torch', verbose=False)
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
    assert_allclose(got.fitted_parameters_vector, ref.fitted_parameters_vector, rtol=1e-4, atol=5e-6)
    with pytest.raises(ValueError, match="tol is solver='csd_jax'-only"):
        mc_t.fit(scheme, data, solver='csd_tournier07_torch', tol=1e-3, verbose=False)
    with full_precision():
        assert not torch.backends.cuda.matmul.allow_tf32
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
