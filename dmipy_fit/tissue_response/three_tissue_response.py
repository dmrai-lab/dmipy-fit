import warnings

import numpy as np

from . import white_matter_response
from ..signal_models.tissue_response_models import (
    estimate_TR1_isotropic_tissue_response_model)
from ..utils.tensor_backend import check_backend, tensor_fa_and_direction

_white_matter_response_algorithms = {
    'tournier07': white_matter_response.white_matter_response_tournier07,
    'tournier13': white_matter_response.white_matter_response_tournier13
}


def three_tissue_response_dhollander16(
        acquisition_scheme, data, *, mask, wm_algorithm='tournier13',
        wm_N_candidate_voxels=300, gm_perc=0.02, csf_perc=0.1, backend='torch',
        **kwargs):
    """
    Heuristic approach to estimating the white matter, grey matter and CSF
    tissue response kernels [1]_, to be used in e.g. Multi-Tissue CSD [2]_. The
    method makes used of so-called 'optimal' thresholds between grey-scale
    images and segmentations [3]_, with iteratively refined binary thresholds
    based on an ad-hoc 'signal decay metric', to finally find candidate voxels
    from which to estimate the three tissue response kernels.

    Parameters
    ----------
    acquisition_scheme : PGSEAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : array of size (..., N_DWIs),
        Measured diffusion signal array (numpy, or a torch tensor with
        ``backend='torch'``).
    mask : boolean array of size data.shape[:-1], required keyword,
        The brain mask: the voxels the three tissues are selected from.
    wm_algorithm : string,
        selection of white matter response estimation algorithm:
        - 'tournier07': classic FA-based estimation,
        - 'tournier13': iterative peak-ratio based estimation.
    wm_N_candidate_voxels : positive integer,
        number of voxels to be included in the white matter response function.
        Default: 300 as done in [4]_.
    gm_perc : positive float between [0, 1],
        fraction of candidate voxels to use in grey matter response function.
        Default: 0.02 as done in [1]_.
    csf_perc : positive float between [0, 1],
        fraction of candidate voxels to use in CSF response function.
        Default: 0.1 as done in [1]_.
    backend : 'torch' or 'jax',
        where the signal decay metric, the diffusion tensors and the white
        matter FODs are computed (the names of the batched CSD solvers,
        ``solver='csd_tournier07_torch'`` / ``'csd_tournier07_jax'``); the
        thresholds and the selections are array operations on the voxels'
        metrics.
    kwargs : optional keyword arguments for WM algorithm,
        see white matter algorithms themselves for possible arguments.

    Returns
    -------
    [S0_wm, S0_gm, S0_csf] : list of floats,
        white matter, grey matter and csf responses.
    [TR2_wm_model, TR1_gm_model, TR1_csf_model]: list of
            TR2AnisotropicTissueResponseModel and
            2 TR1IsotropicTissueResponseModels,
        Modelfree signal representations of white/grey matter and csf.
    three_tissue_selection: array of size data.shape[:-1] + (3,),
        RGB mask of selected voxels used for white/grey matter and csf.

    References
    ----------
    .. [1] Dhollander, T.; Raffelt, D. & Connelly, A. Unsupervised 3-tissue
        response function estimation from single-shell or multi-shell diffusion
        MR data without a co-registered T1 image. ISMRM Workshop on Breaking
        the Barriers of Diffusion MRI, 2016, 5
    .. [2] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b value and number of gradient
        directions for high-angular-resolution diffusion-weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    .. [3] Ridgway, Gerard R., et al. "Issues with threshold masking in
        voxel-based morphometry of atrophied brains." Neuroimage 44.1 (2009):
        99-111.
    .. [4] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b value and number of gradient
        directions for high-angular-resolution diffusion-weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """
    check_backend(backend)
    if mask is None:
        raise ValueError(
            "three_tissue_response_dhollander16 needs mask=, the brain mask (a boolean array of the data's "
            "spatial shape): the three tissues are selected inside it")
    mask = np.asarray(mask, bool)
    if mask.shape != tuple(data.shape[:-1]):
        raise ValueError("mask has shape {}, the data's spatial shape is {}".format(
            mask.shape, tuple(data.shape[:-1])))
    if wm_algorithm not in _white_matter_response_algorithms:
        raise ValueError("wm_algorithm must be one of {}, got {!r}".format(
            sorted(_white_matter_response_algorithms), wm_algorithm))
    brain = np.flatnonzero(mask)
    brain_data = data[_index(mask, data)]                       # (N, N_meas) on the data's device, C order
    if backend == 'torch' and not type(brain_data).__module__.startswith('torch'):
        import torch                                            # one upload serves the SDM, the tensors and the rows
        brain_data = torch.as_tensor(brain_data, device='cuda' if torch.cuda.is_available() else 'cpu')

    # the signal decay metric (SDM), the mean b0 and the FA of every brain voxel, on the backend
    mean_b0, SDM = _mean_b0_and_sdm(acquisition_scheme, brain_data)
    fa, _ = tensor_fa_and_direction(acquisition_scheme, brain_data, backend=backend)
    has_b0 = mean_b0 > 0

    mask_WM = fa > 0.2

    # Separate grey and CSF based on optimal threshold of the FA < 0.2 voxels
    opt = optimal_threshold(SDM[(fa < 0.2) & has_b0])
    mask_CSF = has_b0 & (fa < 0.2) & (SDM > opt)
    mask_GM = has_b0 & (fa < 0.2) & (SDM < opt)

    # Refine Mask, high WM SDM outliers above Q 3 +(Q 3 -Q 1 ) are removed.
    SDM_WM = SDM[mask_WM]
    median_WM = np.median(SDM_WM)
    Q1 = (SDM_WM.min() + median_WM) / 2.0
    Q3 = (SDM_WM.max() + median_WM) / 2.0
    SDM_upper_threshold = Q3 + (Q3 - Q1)
    mask_WM_refine = mask_WM & (SDM < SDM_upper_threshold)
    WM_outlier = mask_WM & (SDM > SDM_upper_threshold)

    # For both the voxels below and above the GM SDM median, optimal thresholds
    # [3] are computed and both parts closer to the initial GM median are
    # retained.
    SDM_GM = SDM[mask_GM]
    median_GM = np.median(SDM_GM)
    optimal_threshold_upper = optimal_threshold(SDM_GM[SDM_GM > median_GM])
    optimal_threshold_lower = optimal_threshold(SDM_GM[SDM_GM < median_GM])
    mask_GM_refine = (
        mask_GM & (SDM > optimal_threshold_lower) & (SDM < optimal_threshold_upper))

    # The high SDM outliers that were removed from the WM are reconsidered for
    # the CSF if they have higher SDM than the current minimal CSF SDM.
    SDM_CSF_min = SDM[mask_CSF].min()
    mask_CSF_updated = mask_CSF | (WM_outlier & (SDM > SDM_CSF_min))

    # An optimal threshold [3] is computed for the resulting CSF and only the
    # higher SDM valued voxels are retained.
    optimal_threshold_CSF = optimal_threshold(SDM[mask_CSF_updated])
    mask_CSF_refine = mask_CSF_updated & (SDM > optimal_threshold_CSF)

    # for WM we use WM response selection algorithm
    wm_voxels = np.flatnonzero(mask_WM_refine)
    response_wm_algorithm = _white_matter_response_algorithms[wm_algorithm]
    S0_wm, TR2_wm_model, indices_wm_selected = response_wm_algorithm(
        acquisition_scheme, _host(brain_data, wm_voxels),
        N_candidate_voxels=wm_N_candidate_voxels, backend=backend, **kwargs)

    # for GM, the voxels closest gm_perc to GM SDM median are selected.
    gm_voxels = np.flatnonzero(mask_GM_refine)
    median_GM = np.median(SDM[gm_voxels])
    N_threshold = int(len(gm_voxels) * gm_perc)
    indices_gm_selected = np.argsort(
        np.abs(SDM[gm_voxels] - median_GM))[:N_threshold]
    S0_gm, TR1_gm_model = estimate_TR1_isotropic_tissue_response_model(
        acquisition_scheme, _host(brain_data, gm_voxels[indices_gm_selected]))

    # for CSF, the csf_perc highest SDM valued voxels are selected.
    csf_voxels = np.flatnonzero(mask_CSF_refine)
    N_threshold = int(len(csf_voxels) * csf_perc)
    indices_csf_selected = np.argsort(SDM[csf_voxels])[::-1][:N_threshold]
    S0_csf, TR1_csf_model = estimate_TR1_isotropic_tissue_response_model(
        acquisition_scheme, _host(brain_data, csf_voxels[indices_csf_selected]))

    # the selected WM/GM/CSF voxels as an RGB volume of the data's spatial shape.
    selection = np.zeros((mask.size, 3))
    selection[brain[wm_voxels[indices_wm_selected]], 0] = 1
    selection[brain[gm_voxels[indices_gm_selected]], 1] = 1
    selection[brain[csf_voxels[indices_csf_selected]], 2] = 1
    three_tissue_selection = selection.reshape(mask.shape + (3,))

    return ([S0_wm, S0_gm, S0_csf],
            [TR2_wm_model, TR1_gm_model, TR1_csf_model],
            three_tissue_selection)


def _index(indices, like):
    """``indices`` in the form that indexes ``like`` (a torch tensor takes them on its device)."""
    if type(like).__module__.startswith('torch'):
        import torch
        return torch.as_tensor(indices, device=like.device)
    return indices


def _host(data, rows):
    """``data[rows]`` as a float64 numpy array, the rows gathered on the data's device first."""
    sub = data[_index(rows, data)]
    if type(sub).__module__.startswith('torch'):
        sub = sub.cpu().numpy()
    return np.asarray(sub, float)


def _mean_b0_and_sdm(acquisition_scheme, data):
    """``(mean_b0 (N,), SDM (N,))`` as float64 numpy arrays for ``data (N, N_meas)``: the per-shell means are one
    float64 product with the averaging matrix of the scheme, on the data's device."""
    W = _shell_averaging_matrix(acquisition_scheme)                    # (N_meas, 1 + N_dwi_shells)
    if type(data).__module__.startswith('torch'):
        import torch
        from ..torch.csd_tournier_torch import full_precision
        with full_precision():
            means = data.double() @ torch.as_tensor(W, dtype=torch.float64, device=data.device)
        means = means.cpu().numpy()
    else:
        means = np.asarray(data, float) @ W
    return means[:, 0], _sdm_from_means(means)


def _shell_averaging_matrix(acquisition_scheme):
    """``(N_meas, 1 + N_dwi_shells)``: column 0 averages the b0 measurements, column j the j-th DWI shell."""
    cols = [np.asarray(acquisition_scheme.b0_mask, float)]
    for index in acquisition_scheme.unique_dwi_indices:
        cols.append(np.asarray(acquisition_scheme.shell_indices == index, float))
    W = np.stack(cols, axis=1)
    return W / W.sum(0)


def _sdm_from_means(means):
    """The SDM from ``means (N, 1 + N_dwi_shells)`` (the mean b0 first): the mean over the shells of
    ``log(mean_b0 / mean_shell)`` where every mean is positive, else 0; clipped to [0, 10] with a warning when
    it leaves that range."""
    ok = means.min(-1) > 0
    SDM = np.zeros(means.shape[0])
    SDM[ok] = np.mean(np.log(means[ok, :1] / means[ok, 1:]), axis=-1)
    if SDM.size and (SDM.max() > 10 or SDM.min() < 0):
        warnings.warn(("The signal decay metric reached unrealistically " +
                      "high or negative values and was clipped to [0, 10]"),
                      RuntimeWarning)
        SDM = np.clip(SDM, 0, 10)
    return SDM


def signal_decay_metric(acquisition_scheme, data):
    """
    Estimation of the Signal Decay Metric (SDM) for the three-tissue tissue
    response kernel estimation [1]_. The metric is a simple division of the S0
    signal intensity by the b>0 shell's signal intensity - of the mean of their
    intensities if there are multiple shells.

    Parameters
    ----------
    acquisition_scheme : PGSEAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : NDarray,
        Measured diffusion signal array.

    Returns
    -------
    SDM : array of size data,
        Estimated Signal Decay Metric (SDK)

    References
    ----------
    .. [1] Dhollander, T.; Raffelt, D. & Connelly, A. Unsupervised 3-tissue
        response function estimation from single-shell or multi-shell diffusion
        MR data without a co-registered T1 image. ISMRM Workshop on Breaking
        the Barriers of Diffusion MRI, 2016, 5
    """
    data = np.asarray(data, float)
    means = data.reshape((-1, data.shape[-1])) @ _shell_averaging_matrix(acquisition_scheme)
    return _sdm_from_means(means).reshape(data.shape[:-1])


def optimal_threshold(data):
    """Optimal image threshold based on pearson correlation [1]_: the
    threshold T* whose mask ``data > T*`` correlates best with the data,

    T* = argmax_T (\rho(data, data>T)).

    The correlation of the data with the mask of its k largest values is
    ``(mean of those k - mean) sqrt(p / (1 - p)) / std`` with ``p = k / n``;
    it is evaluated for every k at once from the sorted data, and T* is the
    midpoint between the two values the best k separates (a k that would
    split equal values is not a threshold).

    Parameters
    ----------
    data: 1D array,
        scalar array to estimate an 'optimal' threshold on.

    Returns
    -------
    optimal_threshold: float,
        optimal threshold value that maximizes correlation between the original
        and masked data.

    References
    ----------
    .. [1] Ridgway, Gerard R., et al. "Issues with threshold masking in
        voxel-based morphometry of atrophied brains." Neuroimage 44.1 (2009):
        99-111.
    """
    x = np.sort(np.asarray(data, float).ravel())
    n = x.size
    if n < 2 or x[0] == x[-1]:
        raise ValueError("an optimal threshold needs at least two distinct values, got {}".format(n))
    k = np.arange(1, n)                                          # the mask holds the k largest values
    top_mean = np.cumsum(x[::-1])[:-1] / k
    p = k / n
    rho = (top_mean - x.mean()) * np.sqrt(p / (1 - p))
    lower, upper = x[n - k - 1], x[n - k]                        # the values below / above the cut
    rho[lower == upper] = -np.inf
    best = int(np.argmax(rho))
    return 0.5 * (lower[best] + upper[best])
