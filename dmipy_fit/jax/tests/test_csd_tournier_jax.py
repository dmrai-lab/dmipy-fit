"""The batched JAX Tournier CSD is the numpy CsdTournierOptimizer, voxel for voxel."""
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')

from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.distributions import distribute_models
from dmipy_fit.jax.csd_tournier_jax import CsdTournierJaxOptimizer
from dmipy_fit.optimizers_fod.csd_tournier import CsdTournierOptimizer
from dmipy_fit.signal_models.cylinder_models import C1Stick

LAMBDA_PAR = 1.7e-9


def _model(scheme):
    mc = MultiCompartmentSphericalHarmonicsModel(models=[C1Stick()])
    mc.set_fixed_parameter('C1Stick_1_lambda_par', LAMBDA_PAR)
    mc.scheme = scheme
    mc._check_if_kernel_parameters_are_fixed()
    mc.S0_responses = np.ones(1)
    return mc


def _signals(scheme, n=12, seed=0):
    """Dispersed sticks at several orientations and dispersions, with a little noise: the fits differ per voxel."""
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
def test_it_is_the_numpy_tournier_solver_voxel_for_voxel(scheme, unity):
    mc = _model(scheme)
    data = _signals(scheme)
    x0 = np.tile(mc.parameter_initial_guess_to_parameter_vector(), (len(data), 1))
    ref = CsdTournierOptimizer(scheme, mc, x0, sh_order=8, unity_constraint=unity, lambda_lb=1e-5)
    jx = CsdTournierJaxOptimizer(scheme, mc, x0, sh_order=8, unity_constraint=unity, lambda_lb=1e-5)
    expected = np.array([ref(d, x) for d, x in zip(data, x0)])
    got, diag = jx.fit_batch(data, x0, diagnostics=True)
    assert got.shape == expected.shape
    # float32 on the device against float64 numpy: coefficients agree to the float32 rounding of the solves
    assert_allclose(got, expected, rtol=1e-4, atol=2e-5)
    assert (diag['iter_num'] < jx.max_iter).all(), "an active set did not settle within max_iter"
    assert (diag['iter_num'] >= 1).all()
    if unity:
        assert_allclose(got[:, jx._sh_slice][:, 0], 1.0 / (2.0 * np.sqrt(np.pi)))


def test_one_voxel_call_matches_the_batch(scheme):
    mc = _model(scheme)
    data = _signals(scheme, n=3)
    x0 = np.tile(mc.parameter_initial_guess_to_parameter_vector(), (3, 1))
    jx = CsdTournierJaxOptimizer(scheme, mc, x0, sh_order=8, lambda_lb=1e-5)
    batch = jx.fit_batch(data, x0)
    for i in range(3):
        assert_allclose(jx(data[i], x0[i]), batch[i], rtol=1e-5, atol=1e-6)   # float32 reduction order differs by batch width


def test_chunking_and_padding_change_nothing(scheme, monkeypatch):
    mc = _model(scheme)
    data = _signals(scheme, n=7)
    x0 = np.tile(mc.parameter_initial_guess_to_parameter_vector(), (7, 1))
    jx = CsdTournierJaxOptimizer(scheme, mc, x0, sh_order=8, lambda_lb=1e-5)
    whole = jx.fit_batch(data, x0)
    monkeypatch.setenv("DMIPY_CSD_TOURNIER_BATCH", "3")             # 3 + 3 + 1, the last chunk zero-padded
    assert_allclose(jx.fit_batch(data, x0), whole, rtol=1e-5, atol=1e-6)


def test_a_zero_signal_gives_a_zero_fod_in_one_iteration(scheme):
    mc = _model(scheme)
    x0 = mc.parameter_initial_guess_to_parameter_vector()[None]
    jx = CsdTournierJaxOptimizer(scheme, mc, x0, sh_order=8, unity_constraint=False, lambda_lb=1e-5)
    out, diag = jx.fit_batch(np.zeros((1, scheme.number_of_measurements)), x0, diagnostics=True)
    assert np.all(out[0, jx._sh_slice] == 0.0) and diag['iter_num'][0] == 1


def test_the_kernel_must_be_fixed(scheme):
    mc = _model(scheme)
    x0 = mc.parameter_initial_guess_to_parameter_vector()[None].copy()
    x0[:] = 0.0                                                    # a value, not NaN: a voxel-varying request
    with pytest.raises(ValueError, match="fixed"):
        CsdTournierJaxOptimizer(scheme, mc, x0, sh_order=8)


def test_fit_with_the_jax_solver_is_fit_with_the_numpy_solver(scheme):
    data = _signals(scheme, n=5)
    mc_np = MultiCompartmentSphericalHarmonicsModel([C1Stick()]); mc_np.set_fixed_parameter('C1Stick_1_lambda_par', LAMBDA_PAR)
    mc_jx = MultiCompartmentSphericalHarmonicsModel([C1Stick()]); mc_jx.set_fixed_parameter('C1Stick_1_lambda_par', LAMBDA_PAR)
    ref = mc_np.fit(scheme, data, solver='csd_tournier07', verbose=False)
    got = mc_jx.fit(scheme, data, solver='csd_tournier07_jax', verbose=False)
    assert_allclose(got.fitted_parameters_vector, ref.fitted_parameters_vector, rtol=1e-4, atol=2e-5)
    with pytest.raises(ValueError, match="tol is solver='csd_jax'-only"):
        mc_jx.fit(scheme, data, solver='csd_tournier07_jax', tol=1e-3, verbose=False)
    with pytest.raises(ValueError, match="maxiter is for"):
        mc_jx.fit(scheme, data, solver='csd_tournier07', maxiter=10, verbose=False)


def test_max_iter_caps_the_solves(scheme):
    mc = _model(scheme)
    data = _signals(scheme, n=4)
    x0 = np.tile(mc.parameter_initial_guess_to_parameter_vector(), (4, 1))
    jx = CsdTournierJaxOptimizer(scheme, mc, x0, sh_order=8, lambda_lb=1e-5, max_iter=1)
    _, diag = jx.fit_batch(data, x0, diagnostics=True)
    assert (diag['iter_num'] == 1).all()
