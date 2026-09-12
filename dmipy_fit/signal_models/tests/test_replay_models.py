"""``MonteCarloReplay``: one pack as a compartment, posed by ``mu``, equal to the pack's own replay."""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.replay_models import MonteCarloReplay
from ._replay_packs import build_public_pack

D0 = 2.0e-9


@pytest.fixture(scope="module")
def pack():
    return build_public_pack("cylinder", 6e-6, D0, n_t=150, n_walkers=1200, seed=3, K=48, blt_temporal_K=32)


def _scheme():
    b = [0.0, 1e9, 2e9]
    dirs = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0]], float); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvals = np.repeat(b, len(dirs)); gd = np.tile(dirs, (len(b), 1))
    return AcquisitionScheme.from_pgse(bvals, gd, 10e-3, 30e-3)


def test_the_model_is_the_packs_replay_at_its_pose(pack):
    sch = _scheme()
    m = MonteCarloReplay(pack)
    npt.assert_allclose(m(sch, mu=[0.0, 0.0]), pack.replay(sch.sequence, tissue=False), atol=2e-6)          # axis along z
    npt.assert_allclose(m(sch, mu=[np.pi / 2, 0.0]),
                        pack.replay(sch.sequence, tissue=False, orientation=(1.0, 0.0, 0.0)), atol=2e-6)   # along x
    assert not np.allclose(m(sch, mu=[0.0, 0.0]), m(sch, mu=[np.pi / 2, 0.0]))                            # a pose matters
    npt.assert_allclose(m(sch, mu=[0.0, 0.0], surface_relaxivity=2e-5),
                        pack.replay(sch.sequence, tissue=False, rho=2e-5), atol=5e-6)                      # the C2 knob
    assert m(sch, mu=[0.0, 0.0], surface_relaxivity=2e-5)[0] < 1.0
    with pytest.raises(ValueError, match="mu"):
        m(sch)


def test_a_declared_frame_is_honoured(pack):
    """The same walk stored with its axis along x and its frame declared: posed by ``mu`` it is the pack's own
    replay at that axis (exactly, the same rotation), and the z-stored pack's to the walk's Monte-Carlo floor --
    ``mu`` names an axis and leaves the roll about it to the frame's gauge, which a finite walk is not blind to."""
    import copy
    from dmipy_sim.replay import ReplayPack, so3
    from dmipy_sim.replay.compression import pack_position_arrays, read_position_coeffs
    Q = so3.rotation_of((1.0, 0.0, 0.0))
    C = read_position_coeffs(pack.arrays, dtype=np.float64) @ Q.T
    arrays = dict(pack.arrays); arrays.update(pack_position_arrays(C, np.float32))
    meta = copy.deepcopy(pack.meta); meta.setdefault("walk_params", {})["substrate_frame"] = Q.tolist()
    x = ReplayPack(arrays, meta)
    sch = _scheme()
    for mu, axis in (([0.0, 0.0], (0.0, 0.0, 1.0)), ([np.pi / 2, np.pi / 2], (0.0, 1.0, 0.0))):
        npt.assert_allclose(MonteCarloReplay(x)(sch, mu=mu), x.replay(sch.sequence, tissue=False, orientation=axis), atol=2e-6)
        npt.assert_allclose(MonteCarloReplay(x)(sch, mu=mu), MonteCarloReplay(pack)(sch, mu=mu), atol=3 / np.sqrt(pack.n_walkers))
    # the pose as a rotation is exact: stored -> lab is R F^T, so the x-stored pack at R = I is the z-stored one
    npt.assert_allclose(x.replay(sch.sequence, tissue=False, orientation=np.eye(3)), pack.replay(sch.sequence, tissue=False), rtol=1e-5)
