"""``ModelSubstrate``: a compartment model as an analytic substrate of a replay phantom (dmipy-fit#28)."""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_fit.phantom import ModelSubstrate, analytic_substrate
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G1Ball

sim_phantom = pytest.importorskip("dmipy_sim.phantom")
from dmipy_sim import sequences                                     # noqa: E402
from dmipy_sim.phantom import Grid, Peaks, Phantom, Watson          # noqa: E402
from dmipy_sim.replay import so3                                    # noqa: E402


def _seq():
    dirs = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return sequences.pgse(dirs, 5e-3, 15e-3, bvalues=[1e9, 1e9, 2e9, 2e9, 3e9], TE=25e-3)


def _grid():
    return Grid(shape=(1, 1, 1), voxel_size_m=(2e-3, 2e-3, 2e-3))


def test_the_adapter_validates_and_refuses_dispersed_models():
    from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
    with pytest.raises(ValueError, match="disperse it twice"):
        ModelSubstrate(SD1WatsonDistributed([C1Stick()]), m0=1.0)
    with pytest.raises(ValueError, match="missing"):
        ModelSubstrate(C1Stick(), m0=1.0)
    with pytest.raises(ValueError, match="outside"):
        ModelSubstrate(C1Stick(), m0=1.0, lambda_par=9e-9)
    s = ModelSubstrate(C1Stick(), m0=0.7, lambda_par=1.7e-9)
    assert s.oriented and s.to_meta()["model"] == "dmipy_fit:C1Stick" and s.to_meta()["oriented"]
    b = ModelSubstrate(G1Ball(), m0=1.0, lambda_iso=3e-9)
    assert not b.oriented and "oriented" not in b.to_meta()


def test_a_peak_evaluates_the_model_at_that_pose():
    seq = _seq(); stick = ModelSubstrate(C1Stick(), m0=1.0, lambda_par=1.7e-9)
    axis = np.array([1.0, 2.0, 0.5]); axis /= np.linalg.norm(axis)
    ph = Phantom.compose(_grid(), fractions={stick: np.ones((1, 1, 1))},
                         orientation=Peaks(np.broadcast_to(axis, (1, 1, 1, 1, 3)).copy()))
    S = ph.replay(seq)[0, 0, 0]
    b = np.asarray(seq.encoding.bvalues); g = np.asarray(seq.encoding.gradient_directions)
    npt.assert_allclose(S, np.exp(-b * 1.7e-9 * (g @ axis) ** 2), atol=5e-4)


def test_a_watson_field_on_a_stick_is_fits_own_watson_stick():
    """One Watson, stated by the phantom, equals fit's SD1Watson-distributed stick at the same concentration:
    the two spellings of one integral agree, which is why only one is offered."""
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
    from dmipy_fit.utils.utils import cart2sphere
    from dmipy_fit.distributions.distributions import kappa2odi
    seq = _seq(); stick = ModelSubstrate(C1Stick(), m0=1.0, lambda_par=1.7e-9)
    mu = np.array([0.3, 0.2, 1.0]); mu /= np.linalg.norm(mu); kappa = 4.0
    ph = Phantom.compose(_grid(), fractions={stick: np.ones((1, 1, 1))},
                         orientation=Watson(mu=np.broadcast_to(mu, (1, 1, 1, 3)).copy(), kappa=np.full((1, 1, 1), kappa), lmax=12))
    S = ph.replay(seq)[0, 0, 0]
    wd = SD1WatsonDistributed([C1Stick()])
    E = wd(AcquisitionScheme(seq, require_b0=False), SD1Watson_1_mu=cart2sphere(mu)[1:], SD1Watson_1_odi=kappa2odi(kappa), C1Stick_1_lambda_par=1.7e-9)
    npt.assert_allclose(S, np.asarray(E).reshape(-1), atol=5e-3)


def test_round_trip_and_refusal_without_dmipy_fit(tmp_path):
    seq = _seq(); stick = ModelSubstrate(C1Stick(), m0=0.8, lambda_par=1.7e-9, T2_s=0.08)
    ball = ModelSubstrate(G1Ball(), m0=1.0, lambda_iso=3e-9)
    axis = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
    ph = Phantom.compose(_grid(), fractions={stick: np.full((1, 1, 1), 0.6)}, remainder=ball,
                         orientation=Peaks(np.broadcast_to(axis, (1, 1, 1, 1, 3)).copy()))
    ph.write(tmp_path / "models.rph", id="test/models", license="CC0", citation="test")
    back = Phantom.read(tmp_path / "models.rph")
    npt.assert_allclose(back.replay(seq), ph.replay(seq), rtol=1e-6)
    assert back.substrates[0].to_meta() == stick.to_meta()
    m = analytic_substrate(stick.to_meta()); assert m.params == stick.params and m.T2_s == 0.08
    import subprocess, sys, os
    code = ("import sys; sys.modules['dmipy_fit'] = None\n"
            "from dmipy_sim.phantom import Phantom\n"
            "try:\n    Phantom.read(sys.argv[1]); print('READ')\nexcept ValueError as e:\n    print('REFUSED', str(e))\n")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "models.rph")], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert "REFUSED" in out.stdout and "'dmipy_fit'" in out.stdout, out.stdout + out.stderr


def test_physics_the_model_lacks_is_stated_once():
    import warnings
    seq = _seq(); stick = ModelSubstrate(C1Stick(), m0=1.0, lambda_par=1.7e-9)
    ph = Phantom.compose(_grid(), fractions={stick: np.ones((1, 1, 1))},
                         orientation=Peaks(np.zeros((1, 1, 1, 1, 3)) + [0, 0, 1.0]))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        S1 = ph.replay(seq, B0_T=3.0, chi_iso=1e-7); ph.replay(seq, B0_T=3.0, chi_iso=1e-7)
    assert len([w for w in rec if "closed form" in str(w.message)]) == 1
    npt.assert_allclose(S1, ph.replay(seq))
