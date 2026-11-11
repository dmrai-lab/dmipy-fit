"""High attenuation without a compartment-signal floor, in both JAX precisions."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.jax.jax_compat import scheme_to_jax
from dmipy_fit.jax.multicompartment_jax import _make_x1karger_jax_fn, checked_compartment_exponent
from dmipy_fit.signal_models.exchange_models import (
    X0GeneralizedKarger, CompartmentSignalDomainError, compartment_exponent,
)
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin


@pytest.mark.parametrize('x64', [True, False])
def test_no_exchange_high_attenuation_has_no_signal_floor(x64):
    scheme = acquisition_scheme_from_bvalues(np.array([0., 1e11]), np.array([[1., 0, 0]]*2), .004, .04)
    model = X0GeneralizedKarger(G1Ball(), G1Ball())
    parameters = dict(f=.37, kappa=0., G1Ball_1_lambda_iso=.3e-9, G1Ball_2_lambda_iso=2.e-9)
    expected = np.exp(-scheme.bvalues[:, None]*[.3e-9, 2.e-9]) @ [.37, .63]
    want = model(scheme, **parameters)
    assert_allclose(want, expected, rtol=2e-12, atol=0)
    with jax.enable_x64(x64):
        fn = jax.jit(_make_x1karger_jax_fn(model, scheme))
        got = fn(scheme_to_jax(scheme), {k: jnp.asarray(v) for k, v in parameters.items()})
        assert_allclose(got, expected, rtol=3e-11 if x64 else 3e-4, atol=0)


@pytest.mark.parametrize('x64', [True, False])
def test_oriented_pair_has_no_signal_floor(x64):
    scheme = acquisition_scheme_from_bvalues(np.array([0., 2e10]), np.array([[0., 0, 1]]*2), .004, .04)
    model = X0GeneralizedKarger(C1Stick(), G2Zeppelin())
    parameters = dict(f=.37, kappa=0., mu=[0., 0.], C1Stick_1_lambda_par=2.1e-9,
                      G2Zeppelin_1_lambda_par=1.7e-9, G2Zeppelin_1_lambda_perp=.6e-9)
    expected = np.exp(-scheme.bvalues[:, None]*[2.1e-9, 1.7e-9]) @ [.37, .63]
    assert_allclose(model(scheme, **parameters), expected, rtol=2e-12, atol=0)
    with jax.enable_x64(x64):
        fn = jax.jit(_make_x1karger_jax_fn(model, scheme))
        got = fn(scheme_to_jax(scheme), {k: jnp.asarray(v) for k, v in parameters.items()})
        assert_allclose(got, expected, rtol=3e-11 if x64 else 3e-4, atol=0)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_effective_medium_log_domain_raises_by_name(value):
    class RestrictedSignal:
        def __call__(self, scheme, **parameters):
            return np.array([value])
    with pytest.raises(CompartmentSignalDomainError, match='finite E > 0'):
        compartment_exponent(RestrictedSignal(), None, {})
    with pytest.raises(Exception, match='CompartmentSignalDomainError'):
        jax.jit(lambda x: checked_compartment_exponent(x, 'RestrictedSignal'))(jnp.array([value])).block_until_ready()


def test_gaussian_exponents_survive_compartment_underflow():
    scheme = acquisition_scheme_from_bvalues(np.array([0., 1e13]), np.array([[1., 0, 0]]*2), .004, .04)
    with np.errstate(under='ignore'):
        assert G1Ball()(scheme, lambda_iso=2.e-9)[1] == 0
    assert compartment_exponent(G1Ball(), scheme, {'lambda_iso': 2.e-9})[1] == 2.e4
