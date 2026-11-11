"""Every acquisition constructor reports the b of its own waveform.

A three-node Gauss rule integrates the quadratic moment within each held
interval independently of the production polynomial implementation.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from dmipy_analytic.constants import GAMMA
from dmipy_fit.core.acquisition_scheme import (
    AcquisitionScheme, acquisition_scheme_from_bvalues,
    acquisition_scheme_from_qvalues, acquisition_scheme_from_gradient_strengths,
    acquisition_scheme_from_schemefile, gtab_dipy2dmipy,
)
from dmipy_fit.core.waveform import waveform_bvalues

B = np.array([0., 1.1e9, 2.3e9])
DIRS = np.array([[1., 0, 0], [0, .6, .8], [0, 0, 1.]])


def integrated_tensor(g, dt):
    h = np.broadcast_to(dt, (len(g)-1,))
    nodes, weights = np.polynomial.legendre.leggauss(3)
    total = np.sum(g[:-1] * h[:, None], axis=0)      # net moment, removed (readout anchor)
    moment, tensor = -total, np.zeros((3, 3))
    for gradient, duration in zip(g[:-1], h):
        q = GAMMA * (moment + (nodes[:, None]+1)*duration/2 * gradient)
        tensor += np.einsum('k,ki,kj->ij', weights*duration/2, q, q)
        moment += gradient * duration
    return tensor


def build(name, tmp_path):
    from dmipy_replay.sequences import pgse
    legacy = acquisition_scheme_from_bvalues(B, DIRS, .007, .037)
    if name == 'bvalues':
        return legacy
    if name == 'qvalues':
        return acquisition_scheme_from_qvalues(legacy.qvalues, DIRS, .007, .037)
    if name == 'gradient_strengths':
        return acquisition_scheme_from_gradient_strengths(legacy.gradient_strengths, DIRS, .007, .037)
    if name == 'schemefile':
        path = tmp_path / 'protocol.scheme'
        legacy.to_schemefile(str(path))
        return acquisition_scheme_from_schemefile(str(path))
    if name == 'dipy':
        gradient_table = pytest.importorskip('dipy.core.gradients').gradient_table
        return gtab_dipy2dmipy(gradient_table(bvals=B/1e6, bvecs=DIRS, small_delta=.007, big_delta=.037))
    if name == 'pgse':
        return AcquisitionScheme.from_pgse(B, DIRS, .007, .037, n_t=200)
    if name == 'pgste':
        return AcquisitionScheme.from_pgste(B, DIRS, .007, .037, n_t=200)
    if name == 'fexi':
        return AcquisitionScheme.from_fexi(DIRS, .004, .01, [.02, .07], b_f=1.2e9, bvalues=B, dt=.0002)
    if name == 'cpmg':
        return AcquisitionScheme.from_cpmg(2, .03, bvalues=B, gradient_directions=DIRS)
    if name == 'ogse':
        return AcquisitionScheme.from_ogse(B, DIRS, 50., .04, n_t=200)
    if name == 'ste':
        return AcquisitionScheme.from_btensor_ste(B, .04, n_t=200)
    if name == 'pte':
        return AcquisitionScheme.from_btensor_pte(B, [0, 0, 1], .04, n_t=200)
    seq = pgse(DIRS, .007, .037, bvalues=B, n_t=200)
    if name == 'sequence':
        return AcquisitionScheme.from_sequence(seq)
    if name == 'init':
        return AcquisitionScheme(seq)
    if name == 'waveform':
        return AcquisitionScheme.from_waveform(seq.G_eff, seq.dt, DIRS)
    if name == 'btensor_waveform':
        return AcquisitionScheme.from_btensor_waveform(seq.G, seq.dt)
    if name == 'concatenate':
        return AcquisitionScheme.concatenate([AcquisitionScheme(seq), AcquisitionScheme(seq)])
    raise AssertionError(name)


@pytest.mark.parametrize('name', [
    'bvalues', 'qvalues', 'gradient_strengths', 'schemefile', 'dipy',
    'pgse', 'pgste', 'fexi', 'cpmg', 'ogse', 'ste', 'pte',
    'sequence', 'init', 'waveform', 'btensor_waveform', 'concatenate',
])
def test_constructor_b_equals_own_waveform_integral(name, tmp_path):
    scheme = build(name, tmp_path)
    tensors = np.array([integrated_tensor(*scheme.waveform_of(m))
                        for m in range(scheme.number_of_measurements)])
    assert_allclose(scheme.bvalues, np.trace(tensors, axis1=-2, axis2=-1), rtol=3e-13, atol=0)
    assert_allclose(scheme.btensor(), tensors, rtol=3e-13, atol=1e-6)


def test_sampled_pgse_export_keeps_its_reported_b():
    scheme = acquisition_scheme_from_bvalues(B, DIRS, .007, .037)
    gradient, dt = scheme.to_gradient_array(n_t=137)
    assert_allclose(waveform_bvalues(gradient, dt), scheme.bvalues, rtol=3e-13, atol=0)


def test_unrefocused_waveform_is_anchored_at_the_readout():
    gradient = np.zeros((2, 11, 3))
    gradient[1, :5, 0] = .02
    dt = .001
    scheme = AcquisitionScheme.from_waveform(gradient, dt, [[1, 0, 0]]*2, allow_unrefocused=True)
    expected = GAMMA**2 * .02**2 * (5*dt)**3 / 3          # integral of (q(t) - q(T))^2
    assert_allclose(scheme.bvalues, [0, expected], rtol=6e-8, atol=0)  # gradient storage is float32


def test_waveform_free_pgse_grid_preserves_timing_strength_and_b():
    from dmipy_fit.core.waveform import pgse_waveform_blocks
    scheme = acquisition_scheme_from_bvalues(B, DIRS, .007, .037)
    for gradient, dt, stored, rows in pgse_waveform_blocks(scheme):
        assert_allclose((gradient.shape[1]-1)*dt, .044, rtol=4e-16)
        assert_allclose(gradient[:, 0], scheme.gradient_strengths[rows, None]*DIRS[rows], rtol=0, atol=0)
        assert_allclose(waveform_bvalues(gradient, dt), scheme.bvalues[rows], rtol=3e-13, atol=0)
        assert not stored.any()


def test_unrepresentable_pgse_timing_raises_without_rounding():
    from dmipy_fit.core.waveform import PGSETimingDomainError, pgse_waveform_blocks
    scheme = acquisition_scheme_from_bvalues(B, DIRS, .007, .007*np.sqrt(2))
    with pytest.raises(PGSETimingDomainError, match='uniform grid'):
        pgse_waveform_blocks(scheme)


@pytest.mark.parametrize('name', ['pgse', 'pgste', 'ogse', 'ste', 'pte'])
def test_builders_keep_the_requested_b(name, tmp_path):
    # the sequence stores its gradient in float32, so the requested b is met to 1e-7
    assert_allclose(build(name, tmp_path).bvalues, B, rtol=2e-7, atol=0)
