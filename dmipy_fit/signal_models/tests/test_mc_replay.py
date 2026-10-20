"""Monte-Carlo replay compartments C6/S6 and the compiled-scheme engine.

Builds tiny CPU replay packs (a 2-diameter family per shape) as a fixture, then checks:
  * the family's compiled contraction reproduces the reference `pack.replay` exactly;
  * a pack written on a save grid is refused by name through the family loader;
  * C6/S6 run end-to-end, obey the physical range + restriction monotonicity;
  * the exact surface-relaxivity replay knob attenuates the signal and matches the engine;
  * a diameter fit recovers the truth.
"""
import copy
import numpy as np
from dmipy_sim.spec import Tissue
import numpy.testing as npt
import pytest

from dmipy_sim.replay import write_rpk
from ._replay_packs import build_public_pack
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models import cylinder_models, sphere_models
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data import mc_replay

D0 = 2.0e-9


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A tiny 2-diameter replay family per shape, written to a dataset dir (CPU, ~seconds)."""
    root = tmp_path_factory.mktemp("mc_replay_ds")
    for shape in ("cylinder", "sphere"):
        sub = root / "canonical" / f"D0-{D0*1e9:.2f}e-9" / shape
        sub.mkdir(parents=True)
        for d_um in (6.0, 8.0):
            pk = build_public_pack(shape, d_um * 1e-6, D0, n_walkers=1500, seed=7, K=48, blt_temporal_K=32)
            write_rpk(str(sub / f"d{d_um:05.2f}um.rpk"), dict(pk.arrays), pk.meta)
    mc_replay._FAMILY_CACHE.clear()
    return str(root)


def _scheme():
    """PGSE scheme (b0 + several b/direction) on its own grid."""
    delta, Delta = 10e-3, 30e-3
    b = [0.0, 1e9, 2e9, 4e9]
    dirs = [[1, 0, 0], [0, 0, 1], [1, 1, 0]]
    bvals, gd = [], []
    for bb in b:
        for d in dirs:
            bvals.append(bb); gd.append(np.array(d, float) / np.linalg.norm(d))
    return AcquisitionScheme.from_pgse(np.array(bvals), np.array(gd), delta, Delta)


def test_engine_matches_pack_replay(dataset):
    "The family's compiled contraction reproduces the engine pack.replay exactly (the same projection, the same kernel)."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    pk = fam.packs[0]
    scheme = _scheme()
    ref = np.abs(np.asarray(pk.replay(scheme.sequence, tissue=None, complex_signal=True)))
    coeffs = fam._pk[0]
    assert coeffs.K == pk.K and coeffs.C.shape == (pk.n_walkers, pk.K + 2, 3)
    got = np.abs(coeffs.signal(coeffs.compile(scheme._G, float(scheme._dt))))
    npt.assert_allclose(got, ref, atol=2e-6)


def test_a_save_grid_pack_is_refused_by_name(dataset, tmp_path):
    "A pack whose header carries a save grid never reaches the contraction: the loader refuses it by name."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    pk = fam.packs[0]
    root = tmp_path / "old"
    sub = root / "canonical" / f"D0-{D0*1e9:.2f}e-9" / "sphere"
    sub.mkdir(parents=True)
    meta = copy.deepcopy(pk.meta)
    meta["walk_params"]["n_t"] = pk.n_t
    write_rpk(str(sub / "d06.00um.rpk"), dict(pk.arrays), meta)
    with pytest.raises(ValueError, match="walk_params.n_t"):
        mc_replay.load_replay_family("sphere", D0, dataset_dir=str(root))
    with pytest.raises(ValueError, match="n_t"):
        mc_replay.PackGrid.of_row({"K": 48, "T_max": 0.06, "block": {"h": 1e-4}, "n_t": 601}, "row")


def test_a_windowed_pack_is_contracted_window_by_window():
    "A pack stored in two storage windows replays an acquisition reaching the second as the engine does: the sum of"
    " each window's contraction (RPK.md 4.3), positions and contact alike."
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    walk = d.simulate_trajectories(800, D0, d.Sphere(radius=3e-6), 60e-3, seed=11, K=24, window_T=30e-3,
                                   require_gpu=False)
    pk = build_replay_pack(walk, id="test/sphere/windowed", license="CC0", citation="test", K=24, blt_temporal_K=16,
                           window_T=30e-3)
    assert pk.n_windows == 2
    scheme = _scheme()                                             # Delta + delta = 40 ms: into window 1
    coeffs = mc_replay.PackCoefficients(pk)
    assert coeffs.C.shape[1] == 2 * (pk.K + 2) and coeffs.windows_reached(scheme._G.shape[1], float(scheme._dt)) == 2
    W = coeffs.compile(scheme._G, float(scheme._dt))
    assert W.shape[0] == 2 * (pk.K + 2) * 3
    ref = np.asarray(pk.replay(scheme.sequence, tissue=None, complex_signal=True))
    npt.assert_allclose(coeffs.signal(W), ref, atol=2e-6)
    gate, dt = np.ones(scheme._G.shape[1]), float(scheme._dt)
    ref_rho = np.asarray(pk.replay(scheme.sequence, tissue=Tissue(rho2=2e-5), complex_signal=True))
    npt.assert_allclose(coeffs.signal(W, rho_over_D=2e-5 / D0, gate=gate, dt_gate=dt), ref_rho, atol=5e-6)
    short = AcquisitionScheme.from_pgse(np.array([0.0, 2e9]), np.array([[1., 0, 0]] * 2), 5e-3, 15e-3)   # 20 ms: window 0
    W0 = coeffs.compile(short._G, float(short._dt))
    assert W0.shape[0] == (pk.K + 2) * 3
    npt.assert_allclose(coeffs.signal(W0), pk.replay(short.sequence, tissue=None, complex_signal=True), atol=2e-6)


def test_c6_s6_physical_and_monotonic(dataset):
    "C6/S6 run end-to-end: b0=1, signal in (0,1], smaller pore -> higher (more restricted) signal."
    scheme = _scheme()
    for model, kw, axis in ((sphere_models.S6MonteCarloReplaySphere(dataset_dir=dataset), {}, None),
                            (cylinder_models.C6MonteCarloReplayCylinder(dataset_dir=dataset),
                             {"mu": [0.0, 0.0]}, np.array([0.0, 0.0, 1.0]))):
        E_small = model(scheme, diameter=6e-6, **kw)
        E_large = model(scheme, diameter=8e-6, **kw)
        npt.assert_allclose(E_small[0], 1.0, atol=1e-6)          # b0
        assert np.all((E_small > 0) & (E_small <= 1.0 + 1e-9))
        # smaller pore restricts more -> higher signal, across the pore: along a cylinder's axis the two are the
        # same free walk. Checked where the tiny-fixture MC floor (~1/sqrt(1500)) resolves it (b <= 2000 s/mm^2).
        mid = (scheme.bvalues > 0) & (scheme.bvalues <= 2e9)
        if axis is not None:
            mid &= np.abs(scheme.gradient_directions @ axis) < 0.9
        assert np.all(E_small[mid] >= E_large[mid] - 1e-2)


def test_surface_relaxivity_uses_replay_knob(dataset):
    "Supplying surface_relaxivity activates the exact boundary-local-time replay (attenuates the signal),"
    " not an analytic S/V tag-on; matches the engine's rho path."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    pk = fam.packs[0]
    scheme = _scheme()
    s6 = sphere_models.S6MonteCarloReplaySphere(dataset_dir=dataset)
    E0 = s6(scheme, diameter=6e-6)
    Er = s6(scheme, diameter=6e-6, surface_relaxivity=2e-5)      # rho = 20 um/s
    assert np.all(Er <= E0 + 1e-9) and Er[0] < 1.0               # surface relaxation lowers signal (incl b0)
    # engine cross-check on the same pack (exact rho path)
    ref = np.abs(np.asarray(pk.replay(scheme.sequence, tissue=Tissue(rho2=2e-5), complex_signal=True)))
    npt.assert_allclose(Er, ref, atol=5e-6)                       # the same exact kernel, the same gate to the echo


@pytest.fixture(scope="module")
def dense_sphere_ds(tmp_path_factory):
    "A denser sphere family (5 diameters) so a diameter fit is not degenerate on a 2-point grid."
    root = tmp_path_factory.mktemp("mc_replay_dense")
    sub = root / "canonical" / f"D0-{D0*1e9:.2f}e-9" / "sphere"
    sub.mkdir(parents=True)
    for d_um in (5.0, 6.0, 7.0, 8.0, 9.0):
        pk = build_public_pack("sphere", d_um * 1e-6, D0, n_walkers=2000, seed=7, K=48, blt_temporal_K=32)
        write_rpk(str(sub / f"d{d_um:05.2f}um.rpk"), dict(pk.arrays), pk.meta)
    mc_replay._FAMILY_CACHE.clear()
    return str(root)


def test_sphere_diameter_fit_recovers(dense_sphere_ds):
    "A diameter fit on synthetic S6 data recovers the truth (coarse; tiny-pack noise floor)."
    scheme = _scheme()
    model = MultiCompartmentModel([sphere_models.S6MonteCarloReplaySphere(dataset_dir=dense_sphere_ds)])
    truth = model.parameters_to_parameter_vector(S6MonteCarloReplaySphere_1_diameter=7e-6)
    E = model.simulate_signal(scheme, truth)
    fit = model.fit(scheme, E)
    d = float(np.asarray(fit.fitted_parameters["S6MonteCarloReplaySphere_1_diameter"]).reshape(-1)[0])
    npt.assert_allclose(d, 7e-6, atol=1.0e-6)
