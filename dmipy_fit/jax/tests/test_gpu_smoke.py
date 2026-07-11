"""GPU smoke test: verify end-to-end fitting runs on all available JAX devices.

On CPU-only machines this test is skipped with a clear message.
On a CUDA machine (``pip install "jax[cuda12]"``), it detects the GPU,
runs a multi-voxel fit, checks numerical correctness, and reports throughput.

Usage
-----
    # CPU-only — skips with informational message
    pytest dmipy/jax/tests/test_gpu_smoke.py -v

    # GPU machine (after: pip install "jax[cuda12]")
    JAX_PLATFORM_NAME=cuda pytest dmipy/jax/tests/test_gpu_smoke.py -v -s
"""
import time
import pytest
import numpy as np
from numpy.testing import assert_allclose

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_devices():
    """Return list of GPU devices (empty on CPU-only builds)."""
    try:
        return jax.devices('gpu')
    except RuntimeError:
        return []


def _device_label():
    backend = jax.default_backend()
    devs = jax.devices()
    return "{} ({})".format(backend.upper(), devs[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def scheme():
    return wu_minn_hcp_acquisition_scheme()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestGpuSmoke:
    """End-to-end fitting on whatever device JAX is running on.

    On a GPU machine these run on-device. On CPU they run on CPU and the
    ``test_requires_gpu`` test is xfailed.
    """

    N_VOXELS = 20
    GT_LAMBDA = 1.7e-9

    def _make_data(self, scheme):
        ball = G1Ball()
        mc = MultiCompartmentModel(models=[ball])
        rng = np.random.default_rng(42)
        lambdas = self.GT_LAMBDA + rng.normal(0, 0.05e-9, self.N_VOXELS)
        E_all = np.stack([
            mc.simulate_signal(
                scheme,
                mc.parameters_to_parameter_vector(G1Ball_1_lambda_iso=lam))
            for lam in lambdas
        ])
        return mc, E_all, lambdas

    def test_device_info(self, scheme):
        """Print device information — always passes."""
        print("\nJAX device: {}".format(_device_label()))
        print("All devices: {}".format(jax.devices()))
        gpus = _gpu_devices()
        if gpus:
            print("GPU devices: {}".format(gpus))
        else:
            print("No GPU devices detected — running on CPU")

    @pytest.mark.xfail(
        not _gpu_devices(),
        reason="No GPU device detected (install jax[cuda12] and rerun)",
        strict=False,
    )
    def test_requires_gpu(self):
        """Soft-fails on CPU-only builds with an informative message."""
        gpus = _gpu_devices()
        assert gpus, (
            "No GPU device found. Install CUDA JAX:\n"
            "  pip install 'jax[cuda12]'\n"
            "and rerun with: JAX_PLATFORM_NAME=cuda pytest ..."
        )

    def test_ball_multivoxel_correctness(self, scheme):
        """G1Ball multi-voxel fit produces correct results on current device."""
        mc, E_all, gt_lambdas = self._make_data(scheme)
        result = mc.fit(scheme, E_all, solver='jax')
        fitted = result.fitted_parameters['G1Ball_1_lambda_iso']
        assert fitted.shape == (self.N_VOXELS,)
        assert_allclose(fitted, gt_lambdas, rtol=0.05,
                        err_msg="Fitted lambdas deviate >5% from ground truth")

    def test_ball_stick_multivoxel(self, scheme):
        """Ball+Stick multi-voxel fit on current device."""
        ball = G1Ball()
        stick = C1Stick()
        mc = MultiCompartmentModel(models=[ball, stick])
        gt = {
            'G1Ball_1_lambda_iso': 1.7e-9,
            'C1Stick_1_mu': np.array([np.pi / 4, np.pi / 3]),
            'C1Stick_1_lambda_par': 1.7e-9,
            'partial_volume_0': 0.3,
            'partial_volume_1': 0.7,
        }
        gt_params = mc.parameters_to_parameter_vector(**gt)
        E = mc.simulate_signal(scheme, gt_params)
        E_all = np.tile(E, (5, 1))
        result = mc.fit(scheme, E_all, solver='jax')
        vf0 = result.fitted_parameters['partial_volume_0']
        vf1 = result.fitted_parameters['partial_volume_1']
        assert_allclose(vf0 + vf1, np.ones(5), atol=1e-5)

    def test_throughput_report(self, scheme, capsys):
        """Report voxels/second — informational, always passes."""
        mc, E_all, _ = self._make_data(scheme)

        # Warm-up compile
        mc.fit(scheme, E_all[:2], solver='jax')

        # Timed run
        t0 = time.perf_counter()
        mc.fit(scheme, E_all, solver='jax')
        elapsed = time.perf_counter() - t0

        vps = self.N_VOXELS / elapsed
        with capsys.disabled():
            print("\n[{}] {} voxels in {:.2f}s = {:.1f} voxels/s".format(
                _device_label(), self.N_VOXELS, elapsed, vps))

    def test_batch_size_consistency(self, scheme):
        """batch_size=5 gives same result as batch_size=None on current device."""
        mc, E_all, _ = self._make_data(scheme)
        r_full = mc.fit(scheme, E_all, solver='jax', batch_size=None)
        r_batched = mc.fit(scheme, E_all, solver='jax', batch_size=5)
        assert_allclose(
            r_full.fitted_parameters['G1Ball_1_lambda_iso'],
            r_batched.fitted_parameters['G1Ball_1_lambda_iso'],
            rtol=1e-5,
        )
