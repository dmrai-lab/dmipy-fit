import pytest
import numpy as np
from dmipy_fit.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.tissue_response_models import (
    estimate_TR1_isotropic_tissue_response_model,
    estimate_TR2_anisotropic_tissue_response_model)
from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme,)

scheme = wu_minn_hcp_acquisition_scheme()


def _has_jax():
    try:
        import jax
        import jaxopt
        return True
    except ImportError:
        return False


def _make_tissue_data(S0_iso=10., S0_aniso=1.):
    """Create synthetic tissue data and models for CSD S0 tests."""
    ball = G1Ball(lambda_iso=3e-9)
    data_iso = S0_iso * ball(scheme)
    S0_iso_est, iso_model = estimate_TR1_isotropic_tissue_response_model(
        scheme, np.atleast_2d(data_iso))

    zeppelin = G2Zeppelin(
        lambda_par=2.2e-9, lambda_perp=1e-9, mu=[np.pi / 2, np.pi / 2])
    data_aniso = S0_aniso * zeppelin(scheme)
    S0_aniso_est, aniso_model = estimate_TR2_anisotropic_tissue_response_model(
        scheme, np.atleast_2d(data_aniso))

    data_to_fit = 0.3 * data_iso + 0.7 * data_aniso
    return S0_iso_est, S0_aniso_est, iso_model, aniso_model, data_to_fit


def test_fit_S0_response(S0_iso=10., S0_aniso=1.):
    """Original test: default CSD solver with and without S0 responses."""
    S0_iso_est, S0_aniso_est, iso_model, aniso_model, data_to_fit = (
        _make_tissue_data(S0_iso, S0_aniso))

    mccsd = MultiCompartmentSphericalHarmonicsModel(
        models=[iso_model, aniso_model])
    mtcsd = MultiCompartmentSphericalHarmonicsModel(
        models=[iso_model, aniso_model],
        S0_tissue_responses=[S0_iso_est, S0_aniso_est])

    csd_fit_no_S0 = mccsd.fit(scheme, data_to_fit)
    csd_fit_S0 = mtcsd.fit(scheme, data_to_fit)
    np.testing.assert_almost_equal(
        0.3, csd_fit_S0.fitted_parameters['partial_volume_0'], 1)
    np.testing.assert_almost_equal(
        0.7, csd_fit_S0.fitted_parameters['partial_volume_1'], 1)

    # test iso volume fraction overestimated without S0 response
    np.testing.assert_(
        csd_fit_no_S0.fitted_parameters['partial_volume_0'] > 0.3)


def test_fit_S0_response_csd_plus():
    """CSD-Plus solver: S0 tissue responses give correct volume fractions."""
    S0_iso_est, S0_aniso_est, iso_model, aniso_model, data_to_fit = (
        _make_tissue_data(10., 1.))

    mtcsd = MultiCompartmentSphericalHarmonicsModel(
        models=[iso_model, aniso_model],
        S0_tissue_responses=[S0_iso_est, S0_aniso_est])

    csd_fit = mtcsd.fit(scheme, data_to_fit, solver='csd_plus')
    np.testing.assert_almost_equal(
        0.3, csd_fit.fitted_parameters['partial_volume_0'], 1)
    np.testing.assert_almost_equal(
        0.7, csd_fit.fitted_parameters['partial_volume_1'], 1)


@pytest.mark.skipif(not _has_jax(), reason="JAX/jaxopt not installed")
def test_fit_S0_response_csd_jax():
    """CSD-JAX (OSQP) solver: S0 tissue responses give correct VFs."""
    S0_iso_est, S0_aniso_est, iso_model, aniso_model, data_to_fit = (
        _make_tissue_data(10., 1.))

    mtcsd = MultiCompartmentSphericalHarmonicsModel(
        models=[iso_model, aniso_model],
        S0_tissue_responses=[S0_iso_est, S0_aniso_est])

    csd_fit = mtcsd.fit(scheme, data_to_fit, solver='csd_jax')
    np.testing.assert_almost_equal(
        0.3, csd_fit.fitted_parameters['partial_volume_0'], 1)
    np.testing.assert_almost_equal(
        0.7, csd_fit.fitted_parameters['partial_volume_1'], 1)
