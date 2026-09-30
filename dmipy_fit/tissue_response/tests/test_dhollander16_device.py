"""The three-tissue responses of Dhollander 2016 with the brain mask given: the mask is required by name; on a
synthetic three-tissue volume the selected voxels are the right tissue and the S0 responses the tissues' S0; the
returned response models are the numpy/dipy estimators (the oracle, dipy's WLS tensor) on the selected voxels to
1e-9 (measured 4e-16); the torch and JAX backends agree; the optimal threshold is the best split; the white-matter
algorithms take the candidate voxels as a 2-D array; a scheme without timing works."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from dmipy_fit.tissue_response.tests import three_tissue_phantom as ph
from dmipy_fit.tissue_response.three_tissue_response import (
    optimal_threshold, three_tissue_response_dhollander16)
from dmipy_fit.tissue_response.white_matter_response import (
    white_matter_response_tournier07, white_matter_response_tournier13)


@pytest.fixture(scope="module")
def phantom():
    sch = ph.multishell_scheme()
    data, mask, fr = ph.volume(sch)
    return sch, data, mask, fr


@pytest.fixture(scope="module")
def torch_responses(phantom):
    sch, data, mask, _ = phantom
    return three_tissue_response_dhollander16(sch, data, mask=mask, backend='torch')


def test_the_mask_is_required_by_name(phantom):
    sch, data, mask, _ = phantom
    with pytest.raises(TypeError, match='mask'):
        three_tissue_response_dhollander16(sch, data)
    with pytest.raises(ValueError, match='mask='):
        three_tissue_response_dhollander16(sch, data, mask=None)
    with pytest.raises(ValueError, match='shape'):
        three_tissue_response_dhollander16(sch, data, mask=mask[1:])
    with pytest.raises(ValueError, match='backend'):
        three_tissue_response_dhollander16(sch, data, mask=mask, backend='numpy')


def test_the_selection_is_the_right_tissue(phantom, torch_responses):
    _, _, mask, fr = phantom
    S0s, models, sel = torch_responses
    assert sel.shape == mask.shape + (3,)
    tissue = np.argmax(fr, -1)
    for k in range(3):
        picked = sel[..., k] > 0
        assert picked.sum() > 0 and not (picked & ~mask).any()
        assert (tissue[picked] == k).all()                       # measured: every selected voxel
    assert (sel.reshape(-1, 3)[:, 0] > 0).sum() == 300
    np.testing.assert_allclose(S0s, [ph.S0['wm'], ph.S0['gm'], ph.S0['csf']], atol=0.02)   # measured 0.006


def _dipy_TR2_rh(scheme, data):
    """The rotational harmonics of the dipy-based estimator (dipy's WLS tensor, per-voxel pseudo-inverse)."""
    from dipy.reconst import dti
    from dipy.reconst.shm import real_sh_descoteaux_from_index
    from dmipy_fit.core.acquisition_scheme import dti_gradient_table
    gtab, pgse = dti_gradient_table(scheme)
    evecs = dti.TensorModel(gtab).fit(data[:, pgse]).evecs
    max_order = max(scheme.shell_sh_orders.values())
    rh = np.zeros((len(data), scheme.N_dwi_shells, max_order // 2 + 1))
    for i in range(len(data)):
        cos = scheme.gradient_directions @ evecs[i][:, 0]
        for j, index in enumerate(scheme.unique_dwi_indices):
            order = scheme.shell_sh_orders[index]
            m = scheme.shell_indices == index
            l = np.arange(0, order + 1, 2)
            theta = np.arccos(np.clip(cos[m], -1, 1))
            B = real_sh_descoteaux_from_index(np.zeros(len(l)), l, theta[:, None], np.zeros((m.sum(), 1)),
                                              legacy=False)
            rh[i, j, :len(l)] = np.linalg.pinv(B) @ data[i][m]
    return rh.mean(0) / np.mean(data[:, scheme.b0_mask])


def test_the_responses_are_the_dipy_estimators_on_the_selected_voxels(phantom, torch_responses):
    pytest.importorskip('dipy.reconst.dti')
    sch, data, _, _ = phantom
    S0s, (wm, gm, csf), sel = torch_responses
    wm_data = data[sel[..., 0] > 0]
    np.testing.assert_allclose(wm._rotational_harmonics, _dipy_TR2_rh(sch, wm_data), rtol=1e-9, atol=1e-12)
    for k, model in ((1, gm), (2, csf)):
        vox = data[sel[..., k] > 0]
        s0 = np.mean(vox[:, sch.b0_mask])
        ref = [np.mean([np.linalg.pinv(sch.shell_sh_matrices[i])[0] @ v[sch.shell_indices == i] for v in vox])
               for i in sch.unique_dwi_indices]
        np.testing.assert_allclose(model._rotational_harmonics[:, 0], np.array(ref) / s0, rtol=1e-12)
        assert S0s[k] == pytest.approx(s0, rel=1e-12)


def test_the_jax_backend_agrees(phantom, torch_responses):
    pytest.importorskip('jax')
    sch, data, mask, _ = phantom
    S0j, models_j, sel_j = three_tissue_response_dhollander16(sch, data, mask=mask, backend='jax')
    S0t, models_t, sel_t = torch_responses
    np.testing.assert_allclose(S0j, S0t, rtol=2e-3)                      # measured 1.9e-4 (the CSF selection)
    overlap = (sel_j[..., 0] * sel_t[..., 0]).sum() / 300
    assert overlap > 0.9, overlap                                        # measured 0.997
    np.testing.assert_allclose(models_j[0]._rotational_harmonics, models_t[0]._rotational_harmonics,
                               rtol=0, atol=5e-3)                        # measured 9e-4


def test_the_optimal_threshold_is_the_best_split():
    from scipy.stats import pearsonr
    rng = np.random.default_rng(0)
    for x in (np.linspace(0, 8, 101), rng.normal(size=300), np.r_[rng.normal(0, 1, 200), rng.normal(5, 1, 50)],
              np.round(rng.uniform(0, 3, 200), 1)):
        t = optimal_threshold(x)
        xs = np.unique(x)
        cuts = 0.5 * (xs[1:] + xs[:-1])
        best = max(pearsonr(x, x > c)[0] for c in cuts)
        assert pearsonr(x, x > t)[0] == pytest.approx(best, rel=1e-12)
    with pytest.raises(ValueError, match='distinct'):
        optimal_threshold(np.ones(5))


@pytest.mark.parametrize('algorithm', [white_matter_response_tournier07, white_matter_response_tournier13])
def test_the_wm_algorithms_take_candidate_voxels(phantom, algorithm):
    sch, data, mask, _ = phantom
    with pytest.raises(ValueError, match=r'data\[mask\]'):
        algorithm(sch, data, N_candidate_voxels=10)
    with pytest.raises(ValueError, match='backend'):
        algorithm(sch, data[mask], N_candidate_voxels=10, backend='numpy')


def test_a_scheme_of_b_values_and_directions_alone(phantom):
    """A scheme with no timing (built from bvals + bvecs, as a BIDS sidecar without timing gives) estimates the
    three responses (tournier13 fits the FODs on it) and fits MSMT-CSD; the scheme check compares the shell fields
    both schemes have and names the ones that differ."""
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
    sch, data, mask, _ = phantom
    bare = acquisition_scheme_from_bvalues(sch.bvalues, sch.gradient_directions)
    assert bare.shell_delta is None and bare.shell_Delta is None
    S0s, models, sel = three_tissue_response_dhollander16(bare, data, mask=mask)
    model = MultiCompartmentSphericalHarmonicsModel(models, S0_tissue_responses=S0s)
    fit = model.fit(bare, data, mask=mask, solver='csd_msmt_torch', verbose=False)
    assert np.isfinite(fit.fitted_parameters['partial_volume_0'][mask]).all()
    with pytest.raises(ValueError, match='shell_delta, shell_Delta, shell_gradient_strengths differ'):
        model.fit(sch, data, mask=mask, solver='csd_msmt_torch', verbose=False)
    shifted = acquisition_scheme_from_bvalues(sch.bvalues * 1.01, sch.gradient_directions)
    with pytest.raises(ValueError, match='shell_bvalues differ'):
        model.fit(shifted, data, mask=mask, solver='csd_msmt_torch', verbose=False)
