"""Spherical-mean <-> full-model parity for occupancy-gated relaxation factors.

Regression tests for the fix that lets compartment-differential T2 (e.g. surface
relaxivity) flow through the spherical-mean path with the same physics as the full
signal:

  * the reduced SphericalMeanAcquisitionScheme carries per-shell TE,
  * direction-INDEPENDENT factors (T2 / surface relaxivity) factor straight through
    the angular mean and apply on every shell (b0 included) at the shell TE,
  * the spherical-mean framework drops b0-normalisation when a T2 is optimised
    (raw signal vs raw model), matching MultiCompartmentModel.

A single, shared T2 is (correctly) absorbed by per-TE b0 normalisation; the point
of these tests is compartment-*differential* T2, which must survive.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, TransverseRelaxation, AttenuationFactor)
from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalMeanModel


def _multi_te_scheme():
    rng = np.random.default_rng(0)

    def dirs(n):
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    bvals, bvecs, TE = [], [], []
    for te in (0.04, 0.07, 0.10):
        bvals += [0.] * 8 + [1e9] * 40 + [2e9] * 40
        bvecs += [dirs(8), dirs(40), dirs(40)]
        TE += [te] * 88
    return acquisition_scheme_from_bvalues(
        np.array(bvals), np.vstack(bvecs), delta=0.012, Delta=0.040,
        TE=np.array(TE))


def test_spherical_mean_scheme_carries_TE():
    """The reduced (per-shell) scheme must expose the same TEs as the full scheme."""
    sch = _multi_te_scheme()
    assert sch.spherical_mean_scheme.TE is not None
    npt.assert_array_equal(np.unique(sch.spherical_mean_scheme.TE),
                           np.unique(sch.TE))


def test_relaxation_factors_through_spherical_mean():
    """A direction-independent T2 factor multiplies the diffusion spherical mean on
    every shell -- b0 included -- by exp(-TE/T2)."""
    sch = _multi_te_scheme()
    T2, D = 0.09, 1.7e-9
    base = C1Stick()
    # Compare OG-with-T2 against OG-WITHOUT the factor (same rotational-harmonics
    # spherical-mean path) times exp(-TE/T2): the relaxation is exactly the
    # direction-independent scalar the factor should contribute.
    og = OccupancyGatedModel(base, factors=[TransverseRelaxation()])
    og_diff = OccupancyGatedModel(base, factors=[])
    diff_sm = np.asarray(og_diff.spherical_mean(
        sch, mu=np.array([0., 0.]), lambda_par=D)).ravel()
    og_sm = np.asarray(og.spherical_mean(
        sch, mu=np.array([0., 0.]), lambda_par=D, T2=T2)).ravel()
    expected = diff_sm * np.exp(-sch.spherical_mean_scheme.TE / T2)
    npt.assert_allclose(og_sm, expected, atol=1e-8)
    # b0 shells specifically carry exp(-TE/T2), NOT forced to 1
    b0 = sch.shell_b0_mask
    npt.assert_allclose(
        og_sm[b0], np.exp(-sch.spherical_mean_scheme.TE[b0] / T2), atol=1e-8)


def test_no_relaxation_factor_leaves_b0_unity():
    """Without a relaxation factor the spherical mean is b0-normalised (b0 = 1)."""
    sch = _multi_te_scheme()
    og = OccupancyGatedModel(C1Stick(), factors=[])
    sm = np.asarray(og.spherical_mean(
        sch, mu=np.array([0., 0.]), lambda_par=1.7e-9)).ravel()
    npt.assert_allclose(sm[sch.shell_b0_mask], 1.0, atol=1e-8)


def test_attenuation_factors_default_separable():
    """Relaxation factors are direction-independent (separable through the mean)."""
    assert AttenuationFactor.spherical_mean_separable is True
    assert TransverseRelaxation().spherical_mean_separable is True


@pytest.mark.slow
def test_smt_recovers_differential_compartment_T2():
    """The headline parity: with compartment T2 optimised, the spherical-mean fit
    recovers the (differential) T2s and the spin-population fraction -- as the full
    MultiCompartmentModel does. Data is the model's own spherical-mean forward, so
    recovery is exact up to the optimiser grid."""
    sch = _multi_te_scheme()
    T2i, T2e, D, PV = 0.090, 0.050, 1.7e-9, 0.5
    intra = OccupancyGatedModel(C1Stick(), factors=[TransverseRelaxation()])
    extra = OccupancyGatedModel(G1Ball(), factors=[TransverseRelaxation()])
    sm = MultiCompartmentSphericalMeanModel([intra, extra])

    def T2n(i):
        return [n for n in sm.parameter_names
                if n.endswith('_T2') and '_%d_' % i in n][0]

    truth = {T2n(1): T2i, T2n(2): T2e,
             **{n: D for n in sm.parameter_names if 'lambda' in n},
             'partial_volume_0': PV, 'partial_volume_1': 1 - PV}
    data = np.asarray(sm(sch, **truth))[None]

    fit = MultiCompartmentSphericalMeanModel([intra, extra])
    for n in fit.parameter_names:
        if 'lambda' in n:
            fit.set_fixed_parameter(n, D)
    fit.set_initial_guess_parameter(T2n(1), 0.07)
    fit.set_initial_guess_parameter(T2n(2), 0.07)
    res = fit.fit(sch, data, solver='brute2fine')
    f = float(np.ravel(res.fitted_parameters['partial_volume_0'])[0])
    npt.assert_allclose(f, PV, atol=0.02)
    npt.assert_allclose(float(np.ravel(res.fitted_parameters[T2n(1)])[0]),
                        T2i, atol=0.01)
    npt.assert_allclose(float(np.ravel(res.fitted_parameters[T2n(2)])[0]),
                        T2e, atol=0.01)
