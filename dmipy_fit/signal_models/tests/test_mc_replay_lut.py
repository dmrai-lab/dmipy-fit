"""Tier-3 signal-LUT fit form: the KB-scale kernel reproduces the full replay for a PGSE family, is far
smaller than the packs, and is fast to evaluate."""
import os
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.replay import write_rpk   # noqa: E402
from ._replay_packs import build_public_pack   # noqa: E402
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme   # noqa: E402
from dmipy_fit.data import mc_replay                    # noqa: E402
from dmipy_fit.data.mc_replay_lut import build_pgse_kernel   # noqa: E402
from dmipy_fit.signal_models import cylinder_models, sphere_models   # noqa: E402

D0 = 2.0e-9
DELTA, DELTAB = 10e-3, 30e-3


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("lut_ds")
    for shape in ("cylinder", "sphere"):
        sub = root / "canonical" / f"D0-{D0*1e9:.2f}e-9" / shape
        sub.mkdir(parents=True)
        for d_um in (5.0, 7.0, 9.0):
            pk = build_public_pack(shape, d_um * 1e-6, D0, n_walkers=4000, seed=7, K=48, blt_temporal_K=32)
            write_rpk(str(sub / f"d{d_um:05.2f}um.rpk"), dict(pk.arrays), pk.meta)
    mc_replay._FAMILY_CACHE.clear()
    return str(root)


def _pgse_scheme():
    b = [0.0, 1e9, 2e9, 3e9]
    dirs = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
    bv, gd = [], []
    for bb in b:
        for d in dirs:
            bv.append(bb); gd.append(np.array(d, float) / np.linalg.norm(d))
    return AcquisitionScheme.from_pgse(np.array(bv), np.array(gd), DELTA, DELTAB)


def _floor(E, n):
    """Three standard errors of a mean of ``n`` unit phasors whose mean is ``E``: what two directions of one finite
    walk differ by at most. The kernel is built from one direction per angle and the full replay reads the walk in
    every direction, so their difference carries this besides the kernel's interpolation error."""
    return 3 * np.sqrt((1 - np.asarray(E) ** 2) / (2 * n))


def test_sphere_lut_matches_full_replay(dataset):
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    n = fam.packs[0].n_walkers
    ker = build_pgse_kernel(fam, DELTA, DELTAB, b_grid=np.linspace(0, 3e9, 25))
    scheme = _pgse_scheme()
    s6 = sphere_models.S6MonteCarloReplaySphere(dataset_dir=dataset)
    for d in (5.5e-6, 7e-6, 8.5e-6):
        full = s6(scheme, diameter=d)
        lut = ker.signal(scheme.bvalues, scheme.gradient_directions, d)
        assert np.all(np.abs(lut - full) <= 5e-3 + _floor(full, n))      # interpolation-grade vs the exact replay
    # leanness: the kernel is orders of magnitude smaller than one pack
    one_pack = os.path.getsize(sorted(__import__("glob").glob(
        os.path.join(dataset, "canonical", "*", "sphere", "*.rpk")))[0])
    assert ker.nbytes < one_pack / 50


def test_cylinder_lut_matches_full_replay(dataset):
    fam = mc_replay.load_replay_family("cylinder", D0, dataset_dir=dataset)
    ker = build_pgse_kernel(fam, DELTA, DELTAB, b_grid=np.linspace(0, 3e9, 25),
                            cos_grid=np.linspace(0, 1, 19))
    n = fam.packs[0].n_walkers
    scheme = _pgse_scheme()
    c6 = cylinder_models.C6MonteCarloReplayCylinder(dataset_dir=dataset)
    mu = [np.pi / 2, 0.0]
    for d in (5.5e-6, 7e-6, 8.5e-6):
        full = c6(scheme, diameter=d, mu=mu)
        lut = ker.signal(scheme.bvalues, scheme.gradient_directions, d, mu=mu)
        # interpolation-grade; the extra error over the sphere is the angular (cos) interpolation, tightened by a
        # finer cos_grid (halves at 2x density). Along the axis the kernel's closed form meets the full replay's
        # finite-walk axial noise, which is the floor's.
        assert np.all(np.abs(lut - full) <= 1.5e-2 + _floor(full, n))


def test_surface_axis_in_lut(dataset):
    "A rho axis in the kernel reproduces the surface-attenuated signal (b0 decays). Exact on the rho grid;"
    " off-grid is coarser (rho enters exp(rho/D * s) nonlinearly -> linear interp is interpolation-grade)."
    fam = mc_replay.load_replay_family("sphere", D0, dataset_dir=dataset)
    rho_on = 1e-5
    ker = build_pgse_kernel(fam, DELTA, DELTAB, b_grid=np.linspace(0, 3e9, 25),
                            rho_grid=(0.0, 1e-5, 2e-5, 3e-5))
    scheme = _pgse_scheme()
    s6 = sphere_models.S6MonteCarloReplaySphere(dataset_dir=dataset)
    # on-grid rho: matches the exact replay to the pure b/diameter interpolation grade
    full = s6(scheme, diameter=7e-6, surface_relaxivity=rho_on)
    lut = ker.signal(scheme.bvalues, scheme.gradient_directions, 7e-6, rho=rho_on)
    assert lut[0] < 1.0                                    # surface decays b0
    n = fam.packs[0].n_walkers
    assert np.all(np.abs(lut - full) <= 5e-3 + _floor(full, n))
    # off-grid rho stays in a sane band (documented interpolation-grade, ~1e-2 at this rho density)
    lut_off = ker.signal(scheme.bvalues, scheme.gradient_directions, 7e-6, rho=1.5e-5)
    full_off = s6(scheme, diameter=7e-6, surface_relaxivity=1.5e-5)
    assert np.all(np.abs(lut_off - full_off) <= 3e-2 + _floor(full_off, n))
