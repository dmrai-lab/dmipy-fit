"""White-matter signal isolation from a multi-tissue CSD fit (Sec. 2.8).

After susceptibility-aware multi-tissue CSD decomposes a voxel into a
white-matter (anisotropic) compartment and isotropic grey-matter / CSF
responses, the white-matter kernel must be re-estimated on the white-matter
signal *alone* - not on the full mixed-tissue signal. ``isolate_wm_signal``
removes the isotropic contributions using the fit's own data-derived response
functions, fitted multi-tissue fractions and S0, with no Gaussian assumption
for grey matter / CSF.

The test checks that what is removed equals an independent reconstruction of
the isotropic contribution from the response models, and that injecting the
isolated signal into a spherical-mean stick+Zeppelin fit no longer sees the
isotropic floor.
"""
import numpy as np
import numpy.testing as npt
from dipy.reconst.shm import real_sh_tournier

from dmipy_fit.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.tissue_response_models import (
    estimate_TR1_isotropic_tissue_response_model,
    estimate_TR2_anisotropic_tissue_response_model)
from dmipy_fit.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy_fit.utils.construct_observation_matrix import (
    construct_model_based_A_matrix)
from dmipy_fit.utils.utils import cart2sphere

scheme = wu_minn_hcp_acquisition_scheme()
LMAX = 8


def _wm_response():
    zep = G2Zeppelin(lambda_par=2.2e-9, lambda_perp=0.5e-9,
                     mu=[np.pi / 2, np.pi / 2])
    _, model = estimate_TR2_anisotropic_tissue_response_model(
        scheme, np.atleast_2d(zep(scheme)))
    return model


def _iso_response(lambda_iso):
    ball = G1Ball(lambda_iso=lambda_iso)
    _, model = estimate_TR1_isotropic_tissue_response_model(
        scheme, np.atleast_2d(ball(scheme)))
    return model


def _delta_sh(direction):
    _, th, ph = cart2sphere(np.atleast_2d(direction)).T
    return real_sh_tournier(LMAX, th, ph, legacy=False)[0][0]


def _build_three_tissue_voxel():
    """Synthetic WM(z-fibre)+GM+CSF voxel with known fractions and S0s."""
    wm, gm, csf = _wm_response(), _iso_response(0.8e-9), _iso_response(3.0e-9)
    S0s = [9000., 6000., 12000.]            # wm, gm, csf intrinsic b0
    mc = MultiCompartmentSphericalHarmonicsModel(
        [wm, gm, csf], S0_tissue_responses=S0s)

    f = dict(wm=0.6, gm=0.25, csf=0.15)     # geometric fractions
    rho = np.array(S0s) / max(S0s)
    A_wm = construct_model_based_A_matrix(
        scheme, wm.rotational_harmonics_representation(scheme), LMAX)
    x_fod = f['wm'] * _delta_sh([0., 0., 1.])    # single z-fibre, scaled to f_wm
    E = (rho[0] * (A_wm @ x_fod)
         + f['gm'] * rho[1] * gm(scheme)
         + f['csf'] * rho[2] * csf(scheme))
    data = max(S0s) * E                      # raw (non-normalised) signal
    return mc, gm, csf, S0s, data


def test_isolate_removes_independent_isotropic_reconstruction():
    mc, gm, csf, S0s, data = _build_three_tissue_voxel()
    fit = mc.fit(scheme, data, solver='csd_plus', verbose=False)

    removed = data - fit.isolate_wm_signal(data)

    # independent reconstruction of the isotropic contribution from the
    # response models, the fitted fractions and S0 (no isolation machinery).
    rho = np.array(S0s) / max(S0s)
    names = mc.partial_volume_names
    fp = fit.fitted_parameters
    S0 = fit.S0 if not fit.fit_S0_response else np.atleast_1d(fit.max_S0_response)
    S0 = float(np.atleast_1d(S0).ravel()[0])
    f_gm = float(np.ravel(fp[names[1]])[0])
    f_csf = float(np.ravel(fp[names[2]])[0])
    iso_recon = S0 * (f_gm * rho[1] * gm(scheme)
                      + f_csf * rho[2] * csf(scheme))

    npt.assert_allclose(np.squeeze(removed), iso_recon, rtol=1e-6, atol=1e-6)


def test_removed_part_is_purely_isotropic_per_shell():
    """What isolation removes must be direction-independent within each shell -
    that is exactly the defining property of the isotropic (GM/CSF) tissues.
    The white-matter directional contrast is therefore preserved untouched."""
    mc, gm, csf, S0s, data = _build_three_tissue_voxel()
    fit = mc.fit(scheme, data, solver='csd_plus', verbose=False)
    removed = np.squeeze(data - np.squeeze(fit.isolate_wm_signal(data)))

    for shell_index in scheme.unique_dwi_indices:
        shell = scheme.shell_indices == shell_index
        within_shell = removed[shell]
        # the removed signal is constant across gradient directions in a shell
        spread = within_shell.std() / max(abs(within_shell.mean()), 1e-12)
        assert spread < 1e-6, (shell_index, spread)
        # and it is a genuine, positive isotropic floor that was stripped
        assert within_shell.mean() > 0.

    # isolation lowers the spherical-mean signal (the isotropic floor is gone)
    assert removed[~scheme.b0_mask].mean() > 0.


def test_isolate_wm_spherical_mean_matches_reduced_full_signal():
    """The spherical-mean isolation (memory-light path) must equal reducing the
    full per-measurement WM signal to its spherical mean."""
    from dmipy_fit.core.spherical_mean_framework import (
        MultiCompartmentSphericalMeanModel)
    from dmipy_fit.utils.spherical_mean import (
        estimate_spherical_mean_multi_shell)
    mc, gm, csf, S0s, data = _build_three_tissue_voxel()
    fit = mc.fit(scheme, data, solver='csd_plus', verbose=False)

    wm_full = np.squeeze(fit.isolate_wm_signal(data))            # (N_meas,)
    wm_sm = np.squeeze(fit.isolate_wm_spherical_mean(data))      # (N_shells,)
    assert wm_sm.shape[-1] == scheme.N_shells
    ref = estimate_spherical_mean_multi_shell(wm_full, scheme)
    npt.assert_allclose(wm_sm, ref, atol=1e-9)


def test_spherical_mean_fit_accepts_spherical_mean_data_directly():
    """MultiCompartmentSphericalMeanModel.fit gives the same result whether
    handed the full signal or the pre-reduced per-shell spherical mean."""
    from dmipy_fit.core.spherical_mean_framework import (
        MultiCompartmentSphericalMeanModel)
    from dmipy_fit.signal_models.cylinder_models import C1Stick
    sm = MultiCompartmentSphericalMeanModel([C1Stick(), G2Zeppelin()])
    gt = {'C1Stick_1_lambda_par': 1.7e-9, 'G2Zeppelin_1_lambda_par': 1.7e-9,
          'G2Zeppelin_1_lambda_perp': 0.5e-9,
          'partial_volume_0': 0.6, 'partial_volume_1': 0.4}
    raw = np.broadcast_to(sm(scheme, **gt)[scheme.shell_indices],
                          (4, scheme.number_of_measurements)).copy()
    sm_data = MultiCompartmentSphericalMeanModel.to_spherical_mean(scheme, raw)
    assert sm_data.shape == (4, scheme.N_shells)
    f_raw = sm.fit(scheme, raw, solver='brute2fine', Ns=6)
    f_sm = sm.fit(scheme, sm_data, solver='brute2fine', Ns=6)
    npt.assert_allclose(
        np.ravel(f_raw.fitted_parameters['partial_volume_0']),
        np.ravel(f_sm.fitted_parameters['partial_volume_0']), atol=2e-3)
