"""The JAX route of the oriented Karger pair (a Stick and a Zeppelin exchanging at the rate kappa) against the NumPy
model: Stick–Zeppelin Karger, whose zeppelin's perpendicular diffusivity follows from its parallel one and the intra-axonal fraction by
the tortuosity link, and the same pair with the perpendicular diffusivity free. The pairs the closed Karger form is not
wired for are refused by name."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.custom_optimizers.reference_models import nexi, stick_zeppelin_karger
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn, build_mc_sm_forward_fn
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
from dmipy_fit.signal_models.nexi_reference import nexi_jelescu2022

U32 = 2.0 ** -24            # the unit roundoff of the float32 forward
OPS = 64                    # the rounding operations between the parameters and the signal: the two diffusivities
#                             along the gradient (6), the exponents (4), the Karger rates (30) and eigenvalues (10)
#                             and the two exponentials with their weights (14)
PREFIX = "X0GeneralizedKarger_1_"


@pytest.fixture(scope="module")
def scheme():
    return wu_minn_hcp_acquisition_scheme()


def _compare(model, scheme, values, waveform=False):
    expected = np.asarray(model(scheme, **values))
    forward = build_mc_forward_fn(model, scheme)
    got = np.asarray(forward(model.parameters_to_parameter_vector(**values)), dtype=float)
    rtol = OPS * U32 * max(1.0, float(-np.log(expected.min())))     # the exponent's rounding reaches the signal
    if waveform:
        # the finite-pulse waveform propagator multiplies a few hundred float32 steps; measured 3e-5
        rtol = max(rtol, 1e-4)
    assert got == pytest.approx(expected, rel=rtol, abs=0)
    return expected


@pytest.mark.parametrize("f", [0.3, 0.7])
def test_stick_zeppelin_karger_matches_the_numpy_model(scheme, f):
    values = {PREFIX + "C1Stick_1_lambda_par": 2.0e-9, PREFIX + "G2Zeppelin_1_lambda_par": 1.5e-9,
              PREFIX + "mu": [0.7, 1.1], PREFIX + "f": f, PREFIX + "kappa": 30.0}
    expected = _compare(stick_zeppelin_karger(), scheme, values, waveform=True)
    assert expected.min() < 0.5                                  # a real attenuation, not a flat signal


def test_the_tortuosity_link_sets_the_perpendicular_diffusivity_of_the_zeppelin(scheme):
    """Two fractions at the same parallel diffusivities attenuate differently through the link alone: the model
    exposes no perpendicular diffusivity of the zeppelin to set."""
    model = stick_zeppelin_karger()
    assert PREFIX + "G2Zeppelin_1_lambda_perp" not in model.parameter_names
    base = {PREFIX + "C1Stick_1_lambda_par": 2.0e-9, PREFIX + "G2Zeppelin_1_lambda_par": 1.5e-9,
            PREFIX + "mu": [0.7, 1.1], PREFIX + "kappa": 30.0}
    forward = build_mc_forward_fn(model, scheme)
    low, high = (np.asarray(forward(model.parameters_to_parameter_vector(**base, **{PREFIX + "f": f})))
                 for f in (0.3, 0.7))
    assert not np.allclose(low, high, rtol=1e-3)


def test_the_oriented_pair_with_a_free_perpendicular_diffusivity_matches_the_numpy_model(scheme):
    model = MultiCompartmentModel([X0GeneralizedKarger(C1Stick(), G2Zeppelin())])
    values = {PREFIX + "C1Stick_1_lambda_par": 2.0e-9, PREFIX + "G2Zeppelin_1_lambda_par": 1.5e-9,
              PREFIX + "G2Zeppelin_1_lambda_perp": 0.6e-9, PREFIX + "mu": [0.7, 1.1],
              PREFIX + "f": 0.45, PREFIX + "kappa": 30.0}
    _compare(model, scheme, values, waveform=True)


def test_a_pair_the_closed_form_is_not_wired_for_is_refused_by_name(scheme):
    model = MultiCompartmentModel([X0GeneralizedKarger(C1Stick(), C1Stick())])
    with pytest.raises(NotImplementedError, match="C1Stick and C1Stick"):
        build_mc_forward_fn(model, scheme)


@pytest.fixture(scope='module')
def published_scheme():
    b = np.tile([0., 0.2e9, 1.e9, 3.e9, 6.e9], 3)
    # Duplicate perpendicular/parallel rows also exercise shell broadcasting.
    return acquisition_scheme_from_bvalues(
        np.repeat(b, 2), np.tile([[1., 0., 0.], [0., 0., 1.]], (len(b), 1)),
        delta=0.004, Delta=np.repeat([0.015, 0.04, 0.08], 10))


@pytest.mark.parametrize('x64', [False, True])
@pytest.mark.parametrize('f', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('di', [1.2e-9, 2.8e-9])
@pytest.mark.parametrize('de', [0.6e-9, 1.8e-9])
@pytest.mark.parametrize('t_ex', [0.005, 0.03, 0.2])
def test_published_nexi_twin_against_reference(published_scheme, x64, f, di, de, t_ex,
                                              record_property):
    """36 tissues x 15 b/Delta pairs in both precisions, against the oracle.

    float32: rtol=2e-5, atol=2e-7; float64: rtol=2e-12, atol=2e-14.
    Also check broadcast measurements and finite gradients of every parameter.
    """
    jax.config.update('jax_enable_x64', x64)
    scheme = published_scheme
    model = nexi()
    values = {'X1NEXI_1_f': f, 'X1NEXI_1_D_i': di,
              'X1NEXI_1_D_e': de, 'X1NEXI_1_t_ex': t_ex}
    vector = jnp.asarray(model.parameters_to_parameter_vector(**values))
    sm = scheme.spherical_mean_scheme
    expected = nexi_jelescu2022(sm.bvalues, sm.Delta-sm.delta/3,
                               f=f, di=di, de=de, t_ex=t_ex)
    forward = build_mc_sm_forward_fn(model, scheme, broadcast=False)
    got = np.asarray(forward(vector))
    record_property('max_absolute_error', float(np.max(np.abs(got-expected))))
    rtol, atol = (2e-12, 2e-14) if x64 else (2e-5, 2e-7)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)
    broadcast = build_mc_sm_forward_fn(model, scheme, broadcast=True)
    np.testing.assert_allclose(broadcast(vector), expected[scheme.shell_indices],
                               rtol=rtol, atol=atol)
    jacobian = np.asarray(jax.jacfwd(forward)(vector))
    assert np.isfinite(jacobian).all()
    assert np.all(np.any(jacobian != 0., axis=0))


def test_published_nexi_jax_fit(published_scheme):
    """Independent multi-Delta signal, four free parameters; recovered within
    0.2% from a 5% perturbed guess through the package's JAX fitting path.
    """
    pytest.importorskip('jaxopt')
    jax.config.update('jax_enable_x64', True)
    model = nexi()
    truth = {'X1NEXI_1_f': 0.6, 'X1NEXI_1_D_i': 2.4e-9,
             'X1NEXI_1_D_e': 0.9e-9, 'X1NEXI_1_t_ex': 0.035}
    for name, value in truth.items():
        model.set_initial_guess_parameter(name, value * 0.95)
    sm = published_scheme.spherical_mean_scheme
    signal = nexi_jelescu2022(sm.bvalues, sm.Delta-sm.delta/3,
                             f=0.6, di=2.4e-9, de=0.9e-9, t_ex=0.035)
    fitted = model.fit(published_scheme, signal, solver='jax')
    for name, value in truth.items():
        np.testing.assert_allclose(np.squeeze(fitted.fitted_parameters[name]),
                                   value, rtol=0.002, atol=0.)
