"""Tests for MultiCompartmentModel per-compartment S0 scaling.

Phase 2 of the S0/PD gap fill: verify that S0_tissue_responses are applied
correctly in MCM.__call__ and that fit() recovers correct volume fractions
when compartments have different intrinsic b0 amplitudes.
"""
import numpy as np
import pytest

from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme

scheme = wu_minn_hcp_acquisition_scheme()


def test_s0_responses_applied_in_call():
    """S0_tissue_responses should scale each compartment's signal in __call__."""
    ball = G1Ball()
    stick = C1Stick()
    # S0_tissue_responses: ball has 3x the b0 intensity of stick
    mcm = MultiCompartmentModel(
        [stick, ball], S0_tissue_responses=[1.0, 3.0])

    params = {
        'C1Stick_1_lambda_par': 1.7e-9,
        'C1Stick_1_mu': [np.pi / 2, np.pi / 2],
        'G1Ball_1_lambda_iso': 3.0e-9,
        'partial_volume_0': 0.5,
        'partial_volume_1': 0.5,
    }

    signal_with_s0 = mcm(scheme, **params)

    # Build reference: manually apply rho_i = S0_i / max(S0)
    mcm_no_s0 = MultiCompartmentModel([C1Stick(), G1Ball()])
    stick_signal = C1Stick()(scheme, lambda_par=1.7e-9,
                              mu=[np.pi / 2, np.pi / 2])
    ball_signal = G1Ball()(scheme, lambda_iso=3.0e-9)
    rho_stick = 1.0 / 3.0
    rho_ball = 3.0 / 3.0
    expected = 0.5 * rho_stick * stick_signal + 0.5 * rho_ball * ball_signal

    np.testing.assert_allclose(signal_with_s0, expected, rtol=1e-10)


def test_s0_responses_none_is_identity():
    """Without S0_tissue_responses, S0_responses should be all ones."""
    ball = G1Ball()
    stick = C1Stick()
    mcm_no_s0 = MultiCompartmentModel([stick, ball])
    mcm_ones = MultiCompartmentModel(
        [C1Stick(), G1Ball()], S0_tissue_responses=[1.0, 1.0])

    params = {
        'C1Stick_1_lambda_par': 1.7e-9,
        'C1Stick_1_mu': [np.pi / 2, np.pi / 2],
        'G1Ball_1_lambda_iso': 3.0e-9,
        'partial_volume_0': 0.6,
        'partial_volume_1': 0.4,
    }

    sig_no_s0 = mcm_no_s0(scheme, **params)
    sig_ones = mcm_ones(scheme, **params)
    np.testing.assert_allclose(sig_no_s0, sig_ones, rtol=1e-12)


def test_s0_responses_fit_recovers_vf():
    """fit() should recover correct volume fractions when compartments have
    different intrinsic S0 values."""
    S0_stick = 1.0
    S0_ball = 5.0
    true_vf_stick = 0.7
    true_vf_ball = 0.3

    # Generate signal with known per-compartment S0
    stick_signal = C1Stick()(scheme, lambda_par=1.7e-9,
                              mu=[np.pi / 2, np.pi / 2])
    ball_signal = G1Ball()(scheme, lambda_iso=3.0e-9)
    # Raw (unnormalised) data: each compartment contributes S0_i * vf_i * E_i
    data = (true_vf_stick * S0_stick * stick_signal +
            true_vf_ball * S0_ball * ball_signal)

    # Fit WITH S0_tissue_responses -- should recover true VFs
    mcm = MultiCompartmentModel(
        [C1Stick(), G1Ball()], S0_tissue_responses=[S0_stick, S0_ball])
    mcm.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
    mcm.set_fixed_parameter('C1Stick_1_mu', [np.pi / 2, np.pi / 2])
    mcm.set_fixed_parameter('G1Ball_1_lambda_iso', 3.0e-9)
    fit_result = mcm.fit(scheme, data, solver='brute2fine', Ns=5)

    fitted_vf_stick = float(np.asarray(fit_result.fitted_parameters['partial_volume_0']).reshape(-1)[0])
    fitted_vf_ball = float(np.asarray(fit_result.fitted_parameters['partial_volume_1']).reshape(-1)[0])

    np.testing.assert_allclose(fitted_vf_stick, true_vf_stick, atol=0.05)
    np.testing.assert_allclose(fitted_vf_ball, true_vf_ball, atol=0.05)


def test_max_s0_response_normalization():
    """Verify that __init__ computes S0_responses and max_S0_response correctly."""
    mcm = MultiCompartmentModel(
        [C1Stick(), G1Ball()], S0_tissue_responses=[2.0, 8.0])
    np.testing.assert_allclose(mcm.max_S0_response, 8.0)
    np.testing.assert_allclose(mcm.S0_responses, [0.25, 1.0])
