"""The JAX spherical mean of the zeppelin covers the whole parameter range: the prolate tensor
(``lambda_par > lambda_perp``), the ball and the oblate one (``lambda_par < lambda_perp``), with the default ranges of
the parameters (both 0.1-3e-9 m^2/s), narrowed bounds, a fixed ``lambda_perp`` and the tortuosity link of MC-SMT."""
import numpy as np
import pytest

pytest.importorskip("jax")

from dmipy_analytic import gaussian
from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalMeanModel
from dmipy_fit.custom_optimizers.reference_models import mcsmt
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.jax.multicompartment_jax import build_mc_sm_forward_fn
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin

U32 = 2.0 ** -24           # the unit roundoff of the float32 forward


def _rtol(scheme):
    """The rounding of the float32 forward against the float64 numpy forms, relative: the 29 terms of the oblate
    zeppelin's quadrature are summed (29 U32); each term has the rounding of its weight and of ``exp`` (3 U32 and 4 U32
    for the 2 ulps of the exponential) and an exponent formed from three rounded inputs, the rounded squares and
    complements of the nodes and four operations (9 U32 x_max, x_max = b D_max <= b_max 3e-9); the stick's error
    function has 22 U32 (dmipy-analytic's budget) and the weighted sum 3 U32."""
    x_max = scheme.shell_bvalues.max() * 3e-9
    return (29 + 3 + 4 + 22 + 3 + 9 * x_max) * U32


@pytest.fixture(scope="module")
def scheme():
    return wu_minn_hcp_acquisition_scheme()


def _stick_zeppelin():
    return MultiCompartmentSphericalMeanModel([C1Stick(), G2Zeppelin()])


def _forward_and_expected(model, scheme, values, stick_par, zeppelin_par, zeppelin_perp, f_stick):
    shell_b = scheme.shell_bvalues[~scheme.shell_b0_mask]
    expected = (f_stick * gaussian.stick_powder_average(shell_b, stick_par)
                + (1.0 - f_stick) * gaussian.zeppelin_powder_average(shell_b, zeppelin_par, zeppelin_perp))
    forward = build_mc_sm_forward_fn(model, scheme, broadcast=False)
    got = np.asarray(forward(model.parameters_to_parameter_vector(**values)), dtype=float)[~scheme.shell_b0_mask]
    return got, expected


@pytest.mark.parametrize("zeppelin_par, zeppelin_perp", [(1.6e-9, 0.6e-9), (0.6e-9, 1.6e-9), (0.2e-9, 2.9e-9),
                                                         (1.2e-9, 1.2e-9)])
def test_the_default_ranges_cover_the_prolate_the_spherical_and_the_oblate_zeppelin(scheme, zeppelin_par,
                                                                                    zeppelin_perp):
    model = _stick_zeppelin()
    values = {"C1Stick_1_lambda_par": 2.0e-9, "G2Zeppelin_1_lambda_par": zeppelin_par,
              "G2Zeppelin_1_lambda_perp": zeppelin_perp, "partial_volume_0": 0.4, "partial_volume_1": 0.6}
    got, expected = _forward_and_expected(model, scheme, values, 2.0e-9, zeppelin_par, zeppelin_perp, 0.4)
    assert np.isfinite(got).all()
    assert got == pytest.approx(expected, rel=_rtol(scheme), abs=0)


def test_the_tortuosity_link_matches_the_numpy_forms(scheme):
    model = mcsmt()
    f_stick, f_zeppelin, d_par = 0.45, 0.55, 1.8e-9
    values = {"C1Stick_1_lambda_par": d_par, "partial_volume_0": f_stick, "partial_volume_1": f_zeppelin}
    shell_b = scheme.shell_bvalues[~scheme.shell_b0_mask]
    expected = (f_stick * gaussian.stick_powder_average(shell_b, d_par)
                + f_zeppelin * gaussian.zeppelin_powder_average(shell_b, d_par, (1.0 - f_stick / (f_stick + f_zeppelin)) * d_par))
    forward = build_mc_sm_forward_fn(model, scheme, broadcast=False)
    got = np.asarray(forward(model.parameters_to_parameter_vector(**values)), dtype=float)[~scheme.shell_b0_mask]
    assert got == pytest.approx(expected, rel=_rtol(scheme), abs=0)


def test_bounds_that_admit_only_an_oblate_tensor_are_evaluated(scheme):
    model = _stick_zeppelin()
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_par", [[0.1e-9, 1.0e-9]])
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_perp", [[1.0e-9, 3.0e-9]])
    values = {"C1Stick_1_lambda_par": 2.0e-9, "G2Zeppelin_1_lambda_par": 0.4e-9, "G2Zeppelin_1_lambda_perp": 2.2e-9,
              "partial_volume_0": 0.4, "partial_volume_1": 0.6}
    got, expected = _forward_and_expected(model, scheme, values, 2.0e-9, 0.4e-9, 2.2e-9, 0.4)
    assert got == pytest.approx(expected, rel=_rtol(scheme), abs=0)


def test_a_fixed_perpendicular_diffusivity_above_the_parallel_one_is_evaluated(scheme):
    model = _stick_zeppelin()
    model.set_fixed_parameter("G2Zeppelin_1_lambda_perp", 2.5e-9)
    values = {"C1Stick_1_lambda_par": 2.0e-9, "G2Zeppelin_1_lambda_par": 0.9e-9, "partial_volume_0": 0.4,
              "partial_volume_1": 0.6}
    got, expected = _forward_and_expected(model, scheme, values, 2.0e-9, 0.9e-9, 2.5e-9, 0.4)
    assert got == pytest.approx(expected, rel=_rtol(scheme), abs=0)
