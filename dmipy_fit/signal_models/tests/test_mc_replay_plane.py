"""P6MonteCarloReplayPlane -- the 1-D slab replay compartment. Builds a tiny CPU plane-pack family and
checks engine parity, physical range + restriction monotonicity, the exact surface path, and a
thickness fit."""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.spec import Tissue
from dmipy_sim.replay import write_rpk
from ._replay_packs import build_public_pack
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.data import mc_replay
from dmipy_fit.signal_models.plane_models import P6MonteCarloReplayPlane
from dmipy_fit.core.modeling_framework import MultiCompartmentModel

D0 = 2.0e-9
MU_X = [np.pi / 2, 0.0]     # plane normal along x (== the Box1D restricted axis)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("plane_ds")
    sub = root / "canonical" / f"D0-{D0*1e9:.2f}e-9" / "plane"
    sub.mkdir(parents=True)
    for d_um in (5.0, 7.0, 9.0):
        pk = build_public_pack("plane", d_um * 1e-6, D0, n_walkers=2000, seed=7, K=48, blt_temporal_K=32)
        write_rpk(str(sub / f"d{d_um:05.2f}um.rpk"), dict(pk.arrays), pk.meta)
    mc_replay._FAMILY_CACHE.clear()
    return str(root)


def _scheme(direction):
    b = np.array([0.0, 1e9, 2e9, 4e9])
    gd = np.tile(direction, (len(b), 1))
    return AcquisitionScheme.from_pgse(b, gd, 10e-3, 30e-3)


def test_p6_matches_engine_replay(dataset):
    "P6 with normal along x + x-gradients reproduces the engine pack.replay at a grid diameter."
    fam = mc_replay.load_replay_family("plane", D0, dataset_dir=dataset)
    j = int(np.argmin(np.abs(fam.diameters - 7e-6)))            # the 7 um grid pack
    pk = fam.packs[j]
    scheme = _scheme([1., 0, 0])
    ref = np.abs(np.asarray(pk.replay(scheme.sequence, tissue=None, complex_signal=True)))
    got = P6MonteCarloReplayPlane(dataset_dir=dataset)(scheme, diameter=7e-6, mu=MU_X)
    npt.assert_allclose(got, ref, atol=2e-6)                   # the rotation is the identity here


def test_p6_physical_and_monotonic(dataset):
    "b0=1, signal in (0,1], thinner slab -> more restricted -> higher signal (gradient along normal)."
    scheme = _scheme([1., 0, 0])
    p6 = P6MonteCarloReplayPlane(dataset_dir=dataset)
    E_thin = p6(scheme, diameter=5e-6, mu=MU_X)
    E_thick = p6(scheme, diameter=9e-6, mu=MU_X)
    npt.assert_allclose(E_thin[0], 1.0, atol=1e-6)
    assert np.all((E_thin > 0) & (E_thin <= 1.0 + 1e-9))
    assert E_thin[-1] >= E_thick[-1] - 1e-6


def test_p6_free_in_plane(dataset):
    "A gradient IN the plane (perpendicular to the normal) sees free diffusion, not restriction."
    fam = mc_replay.load_replay_family("plane", D0, dataset_dir=dataset)
    n = fam.packs[0].n_walkers
    p6 = P6MonteCarloReplayPlane(dataset_dir=dataset)
    E = p6(_scheme([0., 0, 1.]), diameter=5e-6, mu=MU_X)       # gradient along z, normal along x
    E_across = p6(_scheme([1., 0, 0]), diameter=5e-6, mu=MU_X)
    b = _scheme([0., 0, 1.]).bvalues
    free = np.exp(-b * D0)
    # a mean of n unit phasors has standard error sqrt(1 / 2n) at most; the in-plane signal is the free one to 3 of
    # those (a 20000-walker slab agrees with exp(-b D) to 1e-4), and nothing like the restricted one across the slab
    npt.assert_allclose(E, free, atol=3 * np.sqrt(0.5 / n))
    assert np.all(E[b > 0] < 0.5 * E_across[b > 0])


def test_p6_surface_uses_replay_knob(dataset):
    fam = mc_replay.load_replay_family("plane", D0, dataset_dir=dataset)
    pk = fam.packs[int(np.argmin(np.abs(fam.diameters - 7e-6)))]
    scheme = _scheme([1., 0, 0])
    p6 = P6MonteCarloReplayPlane(dataset_dir=dataset)
    E0 = p6(scheme, diameter=7e-6, mu=MU_X)
    Er = p6(scheme, diameter=7e-6, mu=MU_X, surface_relaxivity=2e-5)
    assert np.all(Er <= E0 + 1e-9) and Er[0] < 1.0
    ref = np.abs(np.asarray(pk.replay(scheme.sequence, tissue=Tissue(rho2=2e-5), complex_signal=True)))
    npt.assert_allclose(Er, ref, atol=5e-6)


def test_p6_thickness_fit(dataset):
    scheme = _scheme([1., 0, 0])
    model = MultiCompartmentModel([P6MonteCarloReplayPlane(dataset_dir=dataset, mu=MU_X)])
    # fix the normal (fitting orientation off 4 x-gradients is degenerate); fit thickness only
    model.set_fixed_parameter("P6MonteCarloReplayPlane_1_mu", MU_X)
    truth = model.parameters_to_parameter_vector(P6MonteCarloReplayPlane_1_diameter=7e-6)
    E = model.simulate_signal(scheme, truth)
    fit = model.fit(scheme, E)
    d = float(np.asarray(fit.fitted_parameters["P6MonteCarloReplayPlane_1_diameter"]).reshape(-1)[0])
    npt.assert_allclose(d, 7e-6, atol=1.2e-6)
