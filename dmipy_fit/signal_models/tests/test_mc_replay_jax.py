"""JAX/GPU forward for the Monte-Carlo replay engine (Tier 2): parity with the NumPy engine (diffusion
and the exact surface path) and the vmap-batched multi-scheme call. Skipped if JAX is unavailable."""
import os
import numpy as np
import numpy.testing as npt
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from dmipy_sim.replay import replay_coefficients, replay_signal_jax, replay_batch_jax   # noqa: E402
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme   # noqa: E402
from dmipy_fit.data.mc_replay import PackCoefficients             # noqa: E402
from ._replay_packs import build_public_pack                      # noqa: E402

D0 = 2.0e-9
# complex64 vs float64 numpy: ~1e-5; x64 (fit backend default): ~1e-7. Robust bound, << MC floor.
_ATOL = 5e-5


@pytest.fixture(scope="module")
def pack():
    return build_public_pack("sphere", 6e-6, D0, n_walkers=1500, seed=7, K=48, blt_temporal_K=32)


def _WC(pack):
    b = [0.0, 1e9, 2e9, 4e9]
    gd = np.tile([1., 0, 0], (len(b), 1))
    sch = AcquisitionScheme.from_pgse(np.array(b), gd, 10e-3, 30e-3)
    pk = PackCoefficients(pack)
    return pk.compile(sch._G, float(sch._dt)), pk, sch


def test_jax_matches_numpy_diffusion(pack):
    W, pk, _ = _WC(pack)
    np_E = replay_coefficients(pk.C, pk.w, W)
    jx_E = np.abs(np.asarray(replay_signal_jax(pk.C, pk.w, W)))
    npt.assert_allclose(jx_E, np_E, atol=_ATOL)


def test_jax_matches_numpy_surface(pack):
    W, pk, sch = _WC(pack)
    slw = pk.surface_logweight(2e-5 / D0, np.ones(sch._G.shape[1]), float(sch._dt))
    np_E = replay_coefficients(pk.C, pk.w, W, surface_logw=slw)
    jx_E = np.abs(np.asarray(replay_signal_jax(pk.C, pk.w, W, surface_logw=slw)))
    assert jx_E[0] < 1.0                                   # surface decays b0
    npt.assert_allclose(jx_E, np_E, atol=_ATOL)


def test_batched_vmap_matches_looped(pack):
    "replay_batch_jax over a batch of compiled schemes == looping the single-scheme forward."
    W, pk, _ = _WC(pack)
    W_batch = np.stack([W, 0.5 * W, 2.0 * W])              # 3 distinct 'schemes'
    batch = np.asarray(replay_batch_jax(pk.C, pk.w, W_batch))
    looped = np.stack([np.abs(np.asarray(replay_signal_jax(pk.C, pk.w, Wk))) for Wk in W_batch])
    npt.assert_allclose(batch, looped, atol=_ATOL)
    assert batch.shape == (3, W.shape[1])
