"""Tests for Rician eta pre-processing bias correction in CSD.

Phase 5 of the S0/PD gap fill: verify that eta parameter in CsdOsqpOptimizer
correctly removes Rician bias before the QP solve.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
jaxopt = pytest.importorskip('jaxopt')

from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)
from dmipy_fit.signal_models.tissue_response_models import (
    estimate_TR1_isotropic_tissue_response_model,
    estimate_TR2_anisotropic_tissue_response_model)
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.jax.csd_jax import CsdOsqpOptimizer

scheme = wu_minn_hcp_acquisition_scheme()


def test_eta_zero_is_identity():
    """eta=0 should not change the fit result."""
    stick = C1Stick()
    ball = G1Ball()
    mc = MultiCompartmentSphericalHarmonicsModel(
        models=[stick, ball])
    mc.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
    mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
    mc.scheme = scheme
    mc._check_if_kernel_parameters_are_fixed()
    mc.S0_responses = np.ones(len(mc.models), dtype=float)

    x0 = mc.parameter_initial_guess_to_parameter_vector()
    x0_2d = np.reshape(x0, (1, -1))

    # Create synthetic signal
    stick_sig = C1Stick()(scheme, lambda_par=1.7e-9,
                           mu=[np.pi / 2, np.pi / 2])
    ball_sig = G1Ball()(scheme, lambda_iso=3e-9)
    data = 0.7 * stick_sig + 0.3 * ball_sig
    S0 = np.mean(data[scheme.b0_mask])
    data_norm = data / S0

    jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d)

    # fit without eta
    result_no_eta = jax_opt.fit_batch(data_norm[None], x0_2d)
    # fit with eta=0
    result_eta_0 = jax_opt.fit_batch(data_norm[None], x0_2d, eta=0.0)

    assert_allclose(result_no_eta, result_eta_0, rtol=1e-10)


def test_eta_correction_applied():
    """eta > 0 should modify the data via sqrt(d^2 - eta^2)."""
    stick = C1Stick()
    ball = G1Ball()
    mc = MultiCompartmentSphericalHarmonicsModel(
        models=[stick, ball])
    mc.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
    mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
    mc.scheme = scheme
    mc._check_if_kernel_parameters_are_fixed()
    mc.S0_responses = np.ones(len(mc.models), dtype=float)

    x0 = mc.parameter_initial_guess_to_parameter_vector()
    x0_2d = np.reshape(x0, (1, -1))

    # Create clean signal
    stick_sig = C1Stick()(scheme, lambda_par=1.7e-9,
                           mu=[np.pi / 2, np.pi / 2])
    ball_sig = G1Ball()(scheme, lambda_iso=3e-9)
    data_clean = 0.7 * stick_sig + 0.3 * ball_sig
    S0 = np.mean(data_clean[scheme.b0_mask])
    data_norm = data_clean / S0

    jax_opt = CsdOsqpOptimizer(scheme, mc, x0_2d)

    # Add Rician bias: data_rician = sqrt(data^2 + eta^2)
    eta = 0.05
    data_rician = np.sqrt(data_norm ** 2 + eta ** 2)

    # Fit with eta correction should be closer to clean fit
    result_clean = jax_opt.fit_batch(data_norm[None], x0_2d)
    result_rician_no_eta = jax_opt.fit_batch(data_rician[None], x0_2d)
    result_rician_with_eta = jax_opt.fit_batch(data_rician[None], x0_2d, eta=eta)

    # The eta-corrected fit on Rician data should be closer to clean-data fit
    # than the uncorrected fit. Compare only non-NaN elements (NaN = fixed params).
    mask = ~np.isnan(result_clean[0])
    diff_corrected = np.abs(
        result_rician_with_eta[0, mask] - result_clean[0, mask]).sum()
    diff_uncorrected = np.abs(
        result_rician_no_eta[0, mask] - result_clean[0, mask]).sum()
    assert diff_corrected < diff_uncorrected, (
        "eta-corrected fit should be closer to clean fit than uncorrected. "
        "diff_corrected={}, diff_uncorrected={}".format(
            diff_corrected, diff_uncorrected))


def test_eta_via_fit_interface():
    """End-to-end: fit() with solver='csd_jax' and eta parameter."""
    ball = G1Ball(lambda_iso=3e-9)
    data_iso = 10.0 * ball(scheme)
    S0_iso, iso_model = estimate_TR1_isotropic_tissue_response_model(
        scheme, np.atleast_2d(data_iso))

    from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
    zeppelin = G2Zeppelin(
        lambda_par=2.2e-9, lambda_perp=1e-9, mu=[np.pi / 2, np.pi / 2])
    data_aniso = 1.0 * zeppelin(scheme)
    S0_aniso, aniso_model = estimate_TR2_anisotropic_tissue_response_model(
        scheme, np.atleast_2d(data_aniso))

    mtcsd = MultiCompartmentSphericalHarmonicsModel(
        models=[iso_model, aniso_model],
        S0_tissue_responses=[S0_iso, S0_aniso])

    data_to_fit = 0.3 * data_iso + 0.7 * data_aniso

    # Should not raise when eta is passed
    csd_fit = mtcsd.fit(scheme, data_to_fit, solver='csd_jax', eta=0.01)

    # Volume fractions should still be reasonable
    vf0 = float(csd_fit.fitted_parameters['partial_volume_0'].flat[0])
    vf1 = float(csd_fit.fitted_parameters['partial_volume_1'].flat[0])
    assert 0.1 < vf0 < 0.5, "VF0 out of range: {}".format(vf0)
    assert 0.5 < vf1 < 0.9, "VF1 out of range: {}".format(vf1)
