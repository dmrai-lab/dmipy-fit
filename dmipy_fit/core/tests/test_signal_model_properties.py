"""Tests for dmipy/core/signal_model_properties.py mixins.

Uses G2Zeppelin (anisotropic) and G1Ball (isotropic) as concrete
representatives — they are the simplest models that exercise each mixin path.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme

scheme = wu_minn_hcp_acquisition_scheme()
N_shells = len(scheme.unique_dwi_indices)  # non-b0 shells


class TestAnisotropicMixin:
    """Tests for AnisotropicSignalModelProperties via G2Zeppelin."""

    def setup_method(self):
        self.model = G2Zeppelin()
        self.params = dict(
            mu=np.array([0.0, 0.0]),
            lambda_par=1.7e-9,
            lambda_perp=0.5e-9,
        )

    def test_rotational_harmonics_output_shape(self):
        rh = self.model.rotational_harmonics_representation(scheme, **self.params)
        # shape: (N_shells, N_rh_coef) where N_rh_coef = max_sh_order//2 + 1
        assert rh.ndim == 2
        assert rh.shape[0] == N_shells

    def test_rotational_harmonics_zeroth_coef_in_range(self):
        # Zeroth RH coefficient encodes the spherical mean — must be in [0, 1]
        rh = self.model.rotational_harmonics_representation(scheme, **self.params)
        rh0 = rh[:, 0]
        assert np.all(rh0 >= 0), "Zeroth RH coef must be non-negative"
        assert np.all(rh0 <= 2 * np.sqrt(np.pi)), (
            "Zeroth RH coef × 2√π must be ≤ 1 (signal attenuation)")

    def test_spherical_mean_output_shape(self):
        E_mean = self.model.spherical_mean(scheme, **self.params)
        assert E_mean.shape == (len(scheme.shell_bvalues),)

    def test_spherical_mean_b0_is_one(self):
        E_mean = self.model.spherical_mean(scheme, **self.params)
        b0_val = E_mean[~scheme.unique_dwi_indices.astype(bool)[0:1]][0] \
            if False else E_mean[0]
        # b0 shell should equal 1.0 (no diffusion weighting)
        # shell_bvalues[0] == 0
        b0_indices = np.where(scheme.shell_bvalues == 0)[0]
        if len(b0_indices):
            npt.assert_allclose(E_mean[b0_indices], 1.0, atol=1e-10)

    def test_spherical_mean_decreases_with_bvalue(self):
        # Signal must be monotonically non-increasing with b-value for fixed params
        E_mean = self.model.spherical_mean(scheme, **self.params)
        bvals = scheme.shell_bvalues
        sort_idx = np.argsort(bvals)
        assert np.all(np.diff(E_mean[sort_idx]) <= 1e-10), (
            "Spherical mean should be non-increasing with b-value")

    def test_spherical_mean_in_unit_interval(self):
        E_mean = self.model.spherical_mean(scheme, **self.params)
        assert np.all(E_mean >= 0) and np.all(E_mean <= 1.0 + 1e-10)

    def test_convolution_kernel_matrix_shape(self):
        lmax = 8
        Ncoef = (lmax + 2) * (lmax + 1) // 2
        A = self.model.convolution_kernel_matrix(scheme, lmax, **self.params)
        assert A.shape == (scheme.number_of_measurements, Ncoef)

    def test_isotropic_params_give_orientation_independent_rh(self):
        # When lambda_par == lambda_perp the model is isotropic,
        # so all higher-order RH coefficients should be ~0
        params_iso = dict(mu=np.array([0., 0.]),
                          lambda_par=1.0e-9, lambda_perp=1.0e-9)
        rh = self.model.rotational_harmonics_representation(scheme, **params_iso)
        npt.assert_allclose(rh[:, 1:], 0.0, atol=1e-6)


class TestIsotropicMixin:
    """Tests for IsotropicSignalModelProperties via G1Ball."""

    def setup_method(self):
        self.model = G1Ball()
        self.params = dict(lambda_iso=2.0e-9)

    def test_rotational_harmonics_output_shape(self):
        rh = self.model.rotational_harmonics_representation(scheme, **self.params)
        # Isotropic: only one RH coefficient per shell
        assert rh.ndim == 2
        assert rh.shape[0] == N_shells
        assert rh.shape[1] == 1

    def test_spherical_mean_equals_signal_directly(self):
        # Isotropic: spherical mean equals the model evaluated at any direction
        E_mean = self.model.spherical_mean(scheme, **self.params)
        # Pick a single direction and compute signal for non-b0 shells only
        from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
        bvals = scheme.shell_bvalues[scheme.unique_dwi_indices]
        dirs = np.tile([0., 0., 1.], (N_shells, 1))
        single_dir_scheme = acquisition_scheme_from_bvalues(bvals, dirs)
        E_direct = self.model(single_dir_scheme, **self.params)
        npt.assert_allclose(E_mean[scheme.unique_dwi_indices],
                            E_direct, rtol=1e-5)

    def test_spherical_mean_in_unit_interval(self):
        E_mean = self.model.spherical_mean(scheme, **self.params)
        assert np.all(E_mean >= 0) and np.all(E_mean <= 1.0 + 1e-10)

    def test_convolution_kernel_matrix_shape(self):
        lmax = 6
        Ncoef = (lmax + 2) * (lmax + 1) // 2
        A = self.model.convolution_kernel_matrix(scheme, lmax, **self.params)
        assert A.shape == (scheme.number_of_measurements, Ncoef)

    def test_convolution_kernel_only_uses_zeroth_sh(self):
        # Isotropic model: only the L=0 column of the kernel matrix is nonzero
        lmax = 4
        A = self.model.convolution_kernel_matrix(scheme, lmax, **self.params)
        npt.assert_allclose(A[:, 1:], 0.0, atol=1e-10)
