"""The JAX route of the oriented Karger pair (a Stick and a Zeppelin exchanging at the rate kappa) against the NumPy
model: NEXI, whose zeppelin's perpendicular diffusivity follows from its parallel one and the intra-axonal fraction by
the tortuosity link, and the same pair with the perpendicular diffusivity free. The pairs the closed Karger form is not
wired for are refused by name."""
import numpy as np
import pytest

pytest.importorskip("jax")

from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.custom_optimizers.reference_models import nexi
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.jax.multicompartment_jax import build_mc_forward_fn
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin

U32 = 2.0 ** -24            # the unit roundoff of the float32 forward
OPS = 64                    # the rounding operations between the parameters and the signal: the two diffusivities
#                             along the gradient (6), the exponents (4), the Karger rates (30) and eigenvalues (10)
#                             and the two exponentials with their weights (14)
PREFIX = "X0GeneralizedKarger_1_"


@pytest.fixture(scope="module")
def scheme():
    return wu_minn_hcp_acquisition_scheme()


def _compare(model, scheme, values):
    expected = np.asarray(model(scheme, **values))
    forward = build_mc_forward_fn(model, scheme)
    got = np.asarray(forward(model.parameters_to_parameter_vector(**values)), dtype=float)
    rtol = OPS * U32 * max(1.0, float(-np.log(expected.min())))     # the exponent's rounding reaches the signal
    assert got == pytest.approx(expected, rel=rtol, abs=0)
    return expected


@pytest.mark.parametrize("f", [0.3, 0.7])
def test_nexi_matches_the_numpy_model(scheme, f):
    values = {PREFIX + "C1Stick_1_lambda_par": 2.0e-9, PREFIX + "G2Zeppelin_1_lambda_par": 1.5e-9,
              PREFIX + "mu": [0.7, 1.1], PREFIX + "f": f, PREFIX + "kappa": 30.0}
    expected = _compare(nexi(), scheme, values)
    assert expected.min() < 0.5                                  # a real attenuation, not a flat signal


def test_the_tortuosity_link_sets_the_perpendicular_diffusivity_of_the_zeppelin(scheme):
    """Two fractions at the same parallel diffusivities attenuate differently through the link alone: the model
    exposes no perpendicular diffusivity of the zeppelin to set."""
    model = nexi()
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
    _compare(model, scheme, values)


def test_a_pair_the_closed_form_is_not_wired_for_is_refused_by_name(scheme):
    model = MultiCompartmentModel([X0GeneralizedKarger(C1Stick(), C1Stick())])
    with pytest.raises(NotImplementedError, match="C1Stick and C1Stick"):
        build_mc_forward_fn(model, scheme)
