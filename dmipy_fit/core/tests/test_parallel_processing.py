"""
Tests for use_parallel_processing=True with the stdlib
concurrent.futures.ProcessPoolExecutor backend.

These tests use a minimal 2-voxel synthetic dataset so they run quickly
and do not require JAX or any optional dependency.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from dmipy_fit.data import saved_acquisition_schemes
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.modeling_framework import MultiCompartmentModel


@pytest.fixture
def ball_model_and_data():
    """Return a trivial single-Ball model and 2-voxel synthetic data."""
    scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    ball = G1Ball()
    mc_model = MultiCompartmentModel([ball])

    # Ground-truth: two voxels with different isotropic diffusivities
    gt_lambda = np.array([1.0e-9, 2.0e-9])  # m^2/s
    data = np.array([
        mc_model(scheme, G1Ball_1_lambda_iso=gt_lambda[0]),
        mc_model(scheme, G1Ball_1_lambda_iso=gt_lambda[1]),
    ])
    return mc_model, scheme, data, gt_lambda


def test_parallel_processing_matches_serial(ball_model_and_data):
    """Parallel and serial fitting must return identical results on 2 voxels."""
    mc_model, scheme, data, _ = ball_model_and_data

    # Serial fit (baseline)
    fit_serial = mc_model.fit(
        scheme, data,
        solver='brute2fine',
        use_parallel_processing=False,
    )

    # Parallel fit (ProcessPoolExecutor with 2 workers)
    fit_parallel = mc_model.fit(
        scheme, data,
        solver='brute2fine',
        use_parallel_processing=True,
        number_of_processors=2,
    )

    serial_params = fit_serial.fitted_parameters['G1Ball_1_lambda_iso']
    parallel_params = fit_parallel.fitted_parameters['G1Ball_1_lambda_iso']

    assert_allclose(
        parallel_params, serial_params,
        rtol=1e-5,
        err_msg="Parallel and serial brute2fine results differ."
    )
