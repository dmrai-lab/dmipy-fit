"""Monte-Carlo replay compartments C6/S6 and the compiled-scheme engine.

Builds tiny CPU replay packs (a 2-diameter family per shape) as a fixture, then checks:
  * the compiled-scheme engine (`_replay_fit`) reproduces the reference `pack.replay` exactly;
  * C6/S6 run end-to-end, obey the physical range + restriction monotonicity;
  * the exact surface-relaxivity replay knob attenuates the signal and matches the engine;
  * a diameter fit recovers the truth.
Skipped if dmipy_sim (the simulator that builds packs) is unavailable.
"""
import os
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("dmipy_sim")
from dmipy_sim.canonical import build_canonical_pack   # noqa: E402
from dmipy_sim import bank                              # noqa: E402

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme   # noqa: E402
from dmipy_fit.core.constants import CONSTANTS                    # noqa: E402
from dmipy_fit.signal_models._replay_fit import compile_scheme, replay_complex   # noqa: E402
from dmipy_fit.signal_models import cylinder_models, sphere_models   # noqa: E402
from dmipy_fit.core.modeling_framework import MultiCompartmentModel  # noqa: E402
from dmipy_fit.data import mc_replay                              # noqa: E402

D0 = 2.0e-9
GAMMA = CONSTANTS["water_gyromagnetic_ratio"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A tiny 2-diameter replay family per shape, written to a dataset dir (CPU, ~seconds)."""
    root = tmp_path_factory.mktemp("mc_replay_ds")
    for shape in ("cylinder", "sphere"):
        sub = root / "canonical" / f"D0-{D0*1e9:.2f}e-9" / shape
        sub.mkdir(parents=True)
        for d_um in (6.0, 8.0):
            pk = build_canonical_pack(shape, d_um * 1e-6, D0, n_t=150, n_walkers=1500, seed=7,
                                      K=48, blt_temporal_K=32, surface_relaxivity=True,
                                      require_gpu=False, verbose=False)
            bank.write_rpk(str(sub / f"d{d_um:05.2f}um.rpk"), dict(pk.arrays), pk.meta)
    mc_replay._FAMILY_CACHE.clear()
    return str(root)


def _scheme(n_t, dt):
    """PGSE scheme (b0 + several b/direction) on the pack save grid."""
    delta, Delta = 10e-3, 30e-3
    bu = (GAMMA * delta) ** 2 * (Delta - delta / 3)
    b = [0.0, 1e9, 2e9, 4e9]
    dirs = [[1, 0, 0], [0, 0, 1], [1, 1, 0]]
    bvals, gd = [], []
    for bb in b:
        for d in dirs:
            bvals.append(bb); gd.append(np.array(d, float) / np.linalg.norm(d))
    return AcquisitionScheme.from_pgse(np.array(bvals), np.array(gd), delta, Delta)


def test_engine_matches_pack_replay(dataset):
    "The compiled-scheme matmul reproduces the engine pack.replay to ~1e-6 (exact by Parseval)."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    pk = fam.packs[0]
    scheme = _scheme(pk.n_t, pk.dt)
    G = mc_replay.resample_waveform_to_grid(scheme._G, float(scheme._dt), pk.n_t, pk.dt)
    ref = np.abs(np.asarray(pk.replay(G, relaxation=False, complex_signal=True)))
    C, w, K, _ = mc_replay._pack_arrays(pk)
    W = compile_scheme(G, pk.dt, K, GAMMA)
    got = np.abs(replay_complex(C, w, W))
    npt.assert_allclose(got, ref, atol=2e-6)


def test_c6_s6_physical_and_monotonic(dataset):
    "C6/S6 run end-to-end: b0=1, signal in (0,1], smaller pore -> higher (more restricted) signal."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    scheme = _scheme(fam.n_t, fam.dt)
    for model, kw in ((sphere_models.S6MonteCarloReplaySphere(dataset_dir=dataset), {}),
                      (cylinder_models.C6MonteCarloReplayCylinder(dataset_dir=dataset),
                       {"mu": [np.pi / 2, 0.0]})):
        E_small = model(scheme, diameter=6e-6, **kw)
        E_large = model(scheme, diameter=8e-6, **kw)
        npt.assert_allclose(E_small[0], 1.0, atol=1e-6)          # b0
        assert np.all((E_small > 0) & (E_small <= 1.0 + 1e-9))
        # smaller pore restricts more -> higher signal. Check where the tiny-fixture MC floor resolves it
        # (b <= 2000 s/mm^2); at the highest b the sub-1500-walker signal is at the noise floor.
        mid = (scheme.bvalues > 0) & (scheme.bvalues <= 2e9)
        assert np.all(E_small[mid] >= E_large[mid] - 3e-3)


def test_surface_relaxivity_uses_replay_knob(dataset):
    "Supplying surface_relaxivity activates the exact boundary-local-time replay (attenuates the signal),"
    " not an analytic S/V tag-on; matches the engine's rho path."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    pk = fam.packs[0]
    scheme = _scheme(fam.n_t, fam.dt)
    s6 = sphere_models.S6MonteCarloReplaySphere(dataset_dir=dataset)
    E0 = s6(scheme, diameter=6e-6)
    Er = s6(scheme, diameter=6e-6, surface_relaxivity=2e-5)      # rho = 20 um/s
    assert np.all(Er <= E0 + 1e-9) and Er[0] < 1.0               # surface relaxation lowers signal (incl b0)
    # engine cross-check on the same pack (exact rho path)
    G = mc_replay.resample_waveform_to_grid(scheme._G, float(scheme._dt), pk.n_t, pk.dt)
    ref = np.abs(np.asarray(pk.replay(G, relaxation=False, rho=2e-5, complex_signal=True)))
    npt.assert_allclose(Er, ref, atol=5e-6)


@pytest.fixture(scope="module")
def dense_sphere_ds(tmp_path_factory):
    "A denser sphere family (5 diameters) so a diameter fit is not degenerate on a 2-point grid."
    root = tmp_path_factory.mktemp("mc_replay_dense")
    sub = root / "canonical" / f"D0-{D0*1e9:.2f}e-9" / "sphere"
    sub.mkdir(parents=True)
    for d_um in (5.0, 6.0, 7.0, 8.0, 9.0):
        pk = build_canonical_pack("sphere", d_um * 1e-6, D0, n_t=150, n_walkers=2000, seed=7,
                                  K=48, blt_temporal_K=32, surface_relaxivity=True,
                                  require_gpu=False, verbose=False)
        bank.write_rpk(str(sub / f"d{d_um:05.2f}um.rpk"), dict(pk.arrays), pk.meta)
    mc_replay._FAMILY_CACHE.clear()
    return str(root)


def test_sphere_diameter_fit_recovers(dense_sphere_ds):
    "A diameter fit on synthetic S6 data recovers the truth (coarse; tiny-pack noise floor)."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dense_sphere_ds)
    scheme = _scheme(fam.n_t, fam.dt)
    model = MultiCompartmentModel([sphere_models.S6MonteCarloReplaySphere(dataset_dir=dense_sphere_ds)])
    truth = model.parameters_to_parameter_vector(S6MonteCarloReplaySphere_1_diameter=7e-6)
    E = model.simulate_signal(scheme, truth)
    fit = model.fit(scheme, E)
    d = float(np.asarray(fit.fitted_parameters["S6MonteCarloReplaySphere_1_diameter"]).reshape(-1)[0])
    npt.assert_allclose(d, 7e-6, atol=1.0e-6)
