"""Phase 4 tests: vmap-vectorized multi-voxel fitting."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')

from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme


@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


class TestVmapFitBall:
    GT_LAMBDAS = [1.0e-9, 1.5e-9, 2.0e-9, 1.7e-9, 0.8e-9]

    def _make_data(self, scheme):
        ball = G1Ball()
        mc = MultiCompartmentModel(models=[ball])
        E_all = np.stack([
            mc.simulate_signal(
                scheme,
                mc.parameters_to_parameter_vector(G1Ball_1_lambda_iso=lam))
            for lam in self.GT_LAMBDAS
        ])  # (N_voxels, N_meas)
        return mc, E_all

    def test_jax_fits_multiple_voxels(self, scheme):
        mc, E_all = self._make_data(scheme)
        result = mc.fit(scheme, E_all, solver='jax')
        fitted = result.fitted_parameters['G1Ball_1_lambda_iso']
        assert fitted.shape[0] == len(self.GT_LAMBDAS)
        for i, gt in enumerate(self.GT_LAMBDAS):
            assert_allclose(float(fitted[i]), gt, rtol=0.05)

    def test_jax_matches_brute2fine_multivoxel(self, scheme):
        mc, E_all = self._make_data(scheme)
        result_jax = mc.fit(scheme, E_all, solver='jax')
        result_b2f = mc.fit(scheme, E_all, solver='brute2fine',
                            use_parallel_processing=False)
        fitted_jax = result_jax.fitted_parameters['G1Ball_1_lambda_iso']
        fitted_b2f = result_b2f.fitted_parameters['G1Ball_1_lambda_iso']
        assert_allclose(fitted_jax, fitted_b2f, rtol=0.05)

    def test_batch_size_gives_same_result(self, scheme):
        mc, E_all = self._make_data(scheme)
        result_full = mc.fit(scheme, E_all, solver='jax', batch_size=None)
        result_batch = mc.fit(scheme, E_all, solver='jax', batch_size=2)
        fitted_full = result_full.fitted_parameters['G1Ball_1_lambda_iso']
        fitted_batch = result_batch.fitted_parameters['G1Ball_1_lambda_iso']
        assert_allclose(fitted_full, fitted_batch, rtol=1e-6)


class TestVmapFitBallStick:
    N_VOXELS = 3
    GT = {
        'G1Ball_1_lambda_iso': 1.7e-9,
        'C1Stick_1_mu': np.array([np.pi / 4, np.pi / 3]),
        'C1Stick_1_lambda_par': 1.7e-9,
        'partial_volume_0': 0.3,
        'partial_volume_1': 0.7,
    }

    def _make_data(self, scheme):
        ball = G1Ball()
        stick = C1Stick()
        mc = MultiCompartmentModel(models=[ball, stick])
        gt_params = mc.parameters_to_parameter_vector(**self.GT)
        E = mc.simulate_signal(scheme, gt_params)
        E_all = np.tile(E, (self.N_VOXELS, 1))
        return mc, E_all

    def test_volume_fractions_sum_to_one(self, scheme):
        mc, E_all = self._make_data(scheme)
        result = mc.fit(scheme, E_all, solver='jax')
        vf0 = result.fitted_parameters['partial_volume_0']
        vf1 = result.fitted_parameters['partial_volume_1']
        assert_allclose(vf0 + vf1, np.ones(self.N_VOXELS), atol=1e-5)

    def test_fitted_shape_correct(self, scheme):
        mc, E_all = self._make_data(scheme)
        result = mc.fit(scheme, E_all, solver='jax')
        assert result.fitted_parameters['G1Ball_1_lambda_iso'].shape == (
            self.N_VOXELS,)
