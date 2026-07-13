"""Longitudinal (T1) occupancy-gated relaxation + PGSTE support.

The longitudinal sibling of the transverse-relaxation factor. During a
stimulated-echo mixing time TM the magnetisation is stored along the field, so the
transverse factors (T2 / surface relaxivity) see only the encoding lobes while
LongitudinalRelaxation applies exp(-TM/T1) over the storage window. A plain spin
echo has no mixing time (TM unset), so the factor is the identity.

These tests lock:
  * the factor is identity when TM is None and exp(-TM/T1) when TM is set,
  * from_pgste carries TM (Delta = delta + TM, transverse TE = 2*delta),
  * spherical-mean <-> full-model parity for a gated compartment including T1.
"""
import numpy as np
import numpy.testing as npt

from dmipy_fit.core.acquisition_scheme import (
    AcquisitionScheme, acquisition_scheme_from_bvalues)
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, TransverseRelaxation, LongitudinalRelaxation,
    _tau_par)


def _dirs(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _pgste_scheme(delta=6e-3, TM=40e-3):
    bvals = np.concatenate([[0.] * 6, [1e9] * 30, [2e9] * 30])
    bvecs = np.vstack([_dirs(6), _dirs(30, 1), _dirs(30, 2)])
    return AcquisitionScheme.from_pgste(bvals, bvecs, delta=delta, TM=TM)


def test_factor_identity_without_TM():
    """A spin-echo scheme (no TM) -> the longitudinal factor is exactly 1.0."""
    scheme = acquisition_scheme_from_bvalues(
        np.array([0., 1e9, 2e9]), _dirs(3), delta=6e-3, Delta=40e-3,
        TE=np.array([0.06, 0.06, 0.06]))
    assert _tau_par(scheme) is None
    f = LongitudinalRelaxation()
    assert f.factor(scheme, None, {}, T1=1.0) == 1.0
    # T1 unset is also identity, even if a TM were present
    pgste = _pgste_scheme()
    assert f.factor(pgste, None, {}, T1=None) == 1.0


def test_factor_is_exp_minus_TM_over_T1():
    """With TM set, the factor is exp(-TM/T1) per measurement."""
    pgste = _pgste_scheme(delta=6e-3, TM=40e-3)
    T1 = 1.1
    f = LongitudinalRelaxation()
    got = f.factor(pgste, None, {}, T1=T1)
    npt.assert_allclose(got, np.exp(-pgste.TM / T1), atol=1e-12)
    npt.assert_allclose(np.unique(got), np.exp(-0.040 / T1), atol=1e-12)


def test_from_pgste_timing():
    """PGSTE carries TM, Delta = delta + TM, and the transverse echo time = 2*delta."""
    delta, TM = 6e-3, 40e-3
    pgste = _pgste_scheme(delta=delta, TM=TM)
    npt.assert_allclose(pgste.TM, TM)
    npt.assert_allclose(pgste.Delta, delta + TM)
    npt.assert_allclose(pgste.TE, 2.0 * delta)
    # explicit TE overrides the 2*delta default
    bvals = np.array([0., 1e9]); bvecs = _dirs(2)
    ov = AcquisitionScheme.from_pgste(bvals, bvecs, delta=delta, TM=TM, TE=0.05)
    npt.assert_allclose(ov.TE, 0.05)


def test_spherical_mean_matches_full_model_with_T1():
    """The direction-independent T1 factor multiplies the diffusion spherical mean on
    every shell by exp(-TM/T1), exactly as the full signal does (separability)."""
    scheme = _pgste_scheme()
    D, T2, T1 = 1.7e-9, 0.06, 1.1
    og = OccupancyGatedModel(
        C1Stick(), [TransverseRelaxation(), LongitudinalRelaxation()])
    og_diff = OccupancyGatedModel(C1Stick(), [])
    diff_sm = np.asarray(og_diff.spherical_mean(
        scheme, mu=np.array([0., 0.]), lambda_par=D)).ravel()
    og_sm = np.asarray(og.spherical_mean(
        scheme, mu=np.array([0., 0.]), lambda_par=D, T2=T2, T1=T1)).ravel()
    sms = scheme.spherical_mean_scheme
    expected = diff_sm * np.exp(-sms.TE / T2) * np.exp(-sms.TM / T1)
    npt.assert_allclose(og_sm, expected, atol=1e-10)


def test_full_model_applies_T1_per_measurement():
    """The full-signal T1 factor is exp(-TM/T1) times the T2/diffusion signal."""
    scheme = _pgste_scheme()
    D, T2, T1 = 1.7e-9, 0.06, 1.1
    og = OccupancyGatedModel(
        C1Stick(), [TransverseRelaxation(), LongitudinalRelaxation()])
    og_diff = OccupancyGatedModel(C1Stick(), [TransverseRelaxation()])
    E = np.asarray(og(scheme, mu=[0., 0.], lambda_par=D, T2=T2, T1=T1))
    E_noT1 = np.asarray(og_diff(scheme, mu=[0., 0.], lambda_par=D, T2=T2))
    npt.assert_allclose(E, E_noT1 * np.exp(-scheme.TM / T1), atol=1e-12)


def test_pgste_gates_surface_relaxivity_vs_pgse():
    """Physics: on PGSTE the storage window carries only T1, so T2/surface relaxivity
    accrue over 2*delta (the encoding lobes) rather than the whole diffusion time as
    in a matched PGSE. The b0 attenuation therefore differs in the gated direction."""
    delta, TM = 6e-3, 40e-3
    bvals = np.array([0., 1e9]); bvecs = _dirs(2)
    pgste = AcquisitionScheme.from_pgste(bvals, bvecs, delta=delta, TM=TM)
    # matched-diffusion-time PGSE: same Delta, transverse for the full echo time
    pgse = AcquisitionScheme.from_pgse(
        bvals, bvecs, delta=delta, Delta=delta + TM, TE=2 * delta + TM)
    og = OccupancyGatedModel(
        C1Stick(), [TransverseRelaxation(), LongitudinalRelaxation()])
    p = dict(mu=[0., 0.], lambda_par=1.7e-9, T2=0.06, T1=1.1)
    E_pgste = np.asarray(og(pgste, **p))
    E_pgse = np.asarray(og(pgse, **p))
    # PGSE relaxes transversely over the whole (2*delta + TM) echo; PGSTE only over
    # 2*delta transverse + TM longitudinal (T1 >> T2), so PGSTE retains more signal.
    assert E_pgste[0] > E_pgse[0]
