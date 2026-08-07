"""JAX/GPU forward for the Monte-Carlo replay engine (Tier 2): parity with the NumPy engine (diffusion
and the exact surface path) and the vmap-batched multi-scheme call. Skipped if JAX is unavailable."""
import os
import numpy as np
import numpy.testing as npt
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dmipy_sim")

from dmipy_sim.canonical import build_canonical_pack   # noqa: E402
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme   # noqa: E402
from dmipy_fit.core.constants import CONSTANTS   # noqa: E402
from dmipy_fit.data import mc_replay   # noqa: E402
from dmipy_fit.signal_models._replay_fit import (   # noqa: E402
    compile_scheme, replay_complex, replay_complex_jax, replay_batch_jax)

D0 = 2.0e-9
GAMMA = CONSTANTS["water_gyromagnetic_ratio"]
# complex64 vs float64 numpy: ~1e-5; x64 (fit backend default): ~1e-7. Robust bound, << MC floor.
_ATOL = 5e-5


@pytest.fixture(scope="module")
def pack():
    return build_canonical_pack("sphere", 6e-6, D0, n_t=150, n_walkers=1500, seed=7,
                                K=48, blt_temporal_K=32, surface_relaxivity=True,
                                require_gpu=False, verbose=False)


def _WC(pack):
    b = [0.0, 1e9, 2e9, 4e9]
    gd = np.tile([1., 0, 0], (len(b), 1))
    sch = AcquisitionScheme.from_pgse(np.array(b), gd, 10e-3, 30e-3)
    G = mc_replay.resample_waveform_to_grid(sch._G, float(sch._dt), pack.n_t, pack.dt)
    C, w, K, blt = mc_replay._pack_arrays(pack)
    return compile_scheme(G, pack.dt, K, GAMMA), C, w, blt, pack.n_t


def test_jax_matches_numpy_diffusion(pack):
    W, C, w, blt, n_t = _WC(pack)
    np_E = np.abs(replay_complex(C, w, W))
    jx_E = np.abs(np.asarray(replay_complex_jax(C, w, W)))
    npt.assert_allclose(jx_E, np_E, atol=_ATOL)


def test_jax_matches_numpy_surface(pack):
    W, C, w, blt, n_t = _WC(pack)
    rod = 2e-5 / D0
    np_E = np.abs(replay_complex(C, w, W, blt_dct=blt, rho_over_D=rod, n_t=n_t))
    jx_E = np.abs(np.asarray(replay_complex_jax(C, w, W, blt_dct=blt, rho_over_D=rod, n_t=n_t)))
    assert jx_E[0] < 1.0                                   # surface decays b0
    npt.assert_allclose(jx_E, np_E, atol=_ATOL)


def test_batched_vmap_matches_looped(pack):
    "replay_batch_jax over a batch of compiled schemes == looping the single-scheme forward."
    W, C, w, blt, n_t = _WC(pack)
    W_batch = np.stack([W, 0.5 * W, 2.0 * W])              # 3 distinct 'schemes'
    batch = np.asarray(replay_batch_jax(C, w, W_batch))
    looped = np.stack([np.abs(np.asarray(replay_complex_jax(C, w, Wk))) for Wk in W_batch])
    npt.assert_allclose(batch, looped, atol=_ATOL)
    assert batch.shape == (3, W.shape[1])
