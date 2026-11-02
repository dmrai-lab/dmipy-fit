"""The JAX spherical mean of the zeppelin is defined for a prolate tensor (``lambda_par >= lambda_perp``): the fit
refuses, by name, a model whose parameter ranges admit an oblate one, and accepts the models that keep it prolate (the
tortuosity link of MC-SMT, bounds with ``lambda_perp`` below ``lambda_par``, a fixed ``lambda_perp``)."""
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
OPS = 24                   # rounding operations between the parameters and the shell signal: the tortuosity (3), the
#                            exponent and the error function of the average (12) and the weighted sum (9)


@pytest.fixture(scope="module")
def scheme():
    return wu_minn_hcp_acquisition_scheme()


def _stick_zeppelin():
    return MultiCompartmentSphericalMeanModel([C1Stick(), G2Zeppelin()])


def test_the_default_ranges_admit_an_oblate_zeppelin_and_are_refused_by_name(scheme):
    with pytest.raises(ValueError, match="G2Zeppelin_1_lambda_perp"):
        build_mc_sm_forward_fn(_stick_zeppelin(), scheme)


def test_the_tortuosity_link_keeps_the_zeppelin_prolate_and_matches_the_numpy_forms(scheme):
    model = mcsmt()
    shell_b = scheme.shell_bvalues[~scheme.shell_b0_mask]
    f_stick, f_zeppelin, d_par = 0.45, 0.55, 1.8e-9
    values = {"C1Stick_1_lambda_par": d_par, "partial_volume_0": f_stick, "partial_volume_1": f_zeppelin}
    expected = (f_stick * gaussian.stick_powder_average(shell_b, d_par)
                + f_zeppelin * gaussian.zeppelin_powder_average(shell_b, d_par, (1.0 - f_stick / (f_stick + f_zeppelin)) * d_par))
    forward = build_mc_sm_forward_fn(model, scheme, broadcast=False)
    got = np.asarray(forward(model.parameters_to_parameter_vector(**values)), dtype=float)[~scheme.shell_b0_mask]
    rtol = OPS * U32 * max(1.0, float(-np.log(expected.min())))
    assert got == pytest.approx(expected, rel=rtol, abs=0)
    assert np.isfinite(got).all()


def test_bounds_with_the_perpendicular_diffusivity_below_the_parallel_one_are_accepted(scheme):
    model = _stick_zeppelin()
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_par", [[1.0e-9, 3.0e-9]])
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_perp", [[0.1e-9, 1.0e-9]])
    forward = build_mc_sm_forward_fn(model, scheme, broadcast=False)
    values = {"C1Stick_1_lambda_par": 2.0e-9, "G2Zeppelin_1_lambda_par": 1.6e-9, "G2Zeppelin_1_lambda_perp": 0.6e-9,
              "partial_volume_0": 0.4, "partial_volume_1": 0.6}
    shell_b = scheme.shell_bvalues[~scheme.shell_b0_mask]
    expected = (0.4 * gaussian.stick_powder_average(shell_b, 2.0e-9)
                + 0.6 * gaussian.zeppelin_powder_average(shell_b, 1.6e-9, 0.6e-9))
    got = np.asarray(forward(model.parameters_to_parameter_vector(**values)), dtype=float)[~scheme.shell_b0_mask]
    assert got == pytest.approx(expected, rel=OPS * U32 * max(1.0, float(-np.log(expected.min()))), abs=0)


def test_ranges_that_overlap_by_a_little_are_refused(scheme):
    model = _stick_zeppelin()
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_par", [[0.9e-9, 3.0e-9]])
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_perp", [[0.1e-9, 1.0e-9]])
    with pytest.raises(ValueError, match="lambda_perp <= lambda_par"):
        build_mc_sm_forward_fn(model, scheme)


def test_a_fixed_perpendicular_diffusivity_below_the_parallel_range_is_accepted(scheme):
    model = _stick_zeppelin()
    model.set_fixed_parameter("G2Zeppelin_1_lambda_perp", 0.5e-9)
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_par", [[0.5e-9, 3.0e-9]])
    build_mc_sm_forward_fn(model, scheme, broadcast=False)
    model.set_parameter_optimization_bounds("G2Zeppelin_1_lambda_par", [[0.4e-9, 3.0e-9]])
    with pytest.raises(ValueError, match="G2Zeppelin_1_lambda_perp"):
        build_mc_sm_forward_fn(model, scheme, broadcast=False)
