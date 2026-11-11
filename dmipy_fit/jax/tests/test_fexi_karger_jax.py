"""The JAX route of the Karger models on a filter-exchange protocol against the NumPy model: the same generalised
Karger steps (dmipy_analytic.jax.exchange.karger_waveform) along each row's effective gradient, in float32.

The tolerance is the float32 rounding of the route: the pools' exponents (``OPS_D`` operations to a diffusivity along
the row), and per step of the walk the Magnus exponent and its closed 2 x 2 exponential (``OPS_STEP``), whose errors add
over the ``n`` steps of a column-substochastic product (the tree of products adding ``3 log2 n``)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.custom_optimizers.reference_models import stick_zeppelin_karger
from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.gaussian_models import G1Ball

U32 = 2.0 ** -24
OPS_D, OPS_STEP = 16, 36


@pytest.fixture(autouse=True, params=[True, False])
def precision(request):
    with jax.enable_x64(request.param):
        yield


@pytest.fixture(scope="module")
def scheme():
    return AcquisitionScheme.from_fexi([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], 4e-3, 10e-3, [0.02, 0.1], b_f=1e9,
                                       bvalues=[0.0, 0.6e9], dt=2e-4, slew_rate=np.inf)


def _compare(model, scheme, values):
    want = np.asarray(model(scheme, **values))
    got = np.asarray(build_mc_forward_fn(model, scheme)(model.parameters_to_parameter_vector(**values)), dtype=float)
    n = max(s.n_t for s in scheme.protocol) - 1
    u = np.finfo(np.float64 if jax.config.jax_enable_x64 else np.float32).eps / 2
    rtol = (n * OPS_STEP + 3 * np.log2(n) + OPS_D * max(1.0, float(-np.log(want.min())))) * u
    assert got == pytest.approx(want, rel=rtol, abs=0)
    return want


def test_two_balls(scheme):
    model = MultiCompartmentModel([X0GeneralizedKarger(G1Ball(), G1Ball())])
    p = "X0GeneralizedKarger_1_"
    want = _compare(model, scheme, {p + "G1Ball_1_lambda_iso": 0.3e-9, p + "G1Ball_2_lambda_iso": 1.8e-9,
                                    p + "f": 0.45, p + "kappa": 6.0})
    assert want.min() < 0.5


def test_stick_zeppelin_karger(scheme):
    p = "X0GeneralizedKarger_1_"
    _compare(stick_zeppelin_karger(), scheme, {p + "C1Stick_1_lambda_par": 2.1e-9, p + "G2Zeppelin_1_lambda_par": 1.7e-9,
                              p + "mu": [0.5, 0.3], p + "f": 0.6, p + "kappa": 8.0})
