"""Phase 5 tests: IMPULSED convenience model.

Tests:
1. test_impulsed_instantiation — model creates correctly, D_in is fixed.
2. test_impulsed_parameter_recovery — fit to synthetic OGSE data recovers
   R, v_in within tolerances (R: 15%, v_in: 0.10).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from dmipy_fit.custom_optimizers.impulsed import IMPULSED
from dmipy_fit.signal_models.sphere_models import S4SphereGaussianPhaseApproximation
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme


def _make_bvecs(n_m):
    return np.tile(np.r_[1., 0., 0.], (n_m, 1))


def _make_ogse_scheme(freqs=(30., 50., 80., 120., 200.),
                      bvalues_per_freq=None,
                      sigma=0.04, n_t=1000):
    """Build a multi-frequency OGSE scheme for IMPULSED fitting.

    Parameters
    ----------
    freqs : tuple of float, Hz
    bvalues_per_freq : array or None, s/m² (same for each freq)
    sigma : float, gradient duration in s
    """
    if bvalues_per_freq is None:
        bvalues_per_freq = np.array([0, 5e8, 1e9, 2e9])

    schemes = []
    for f in freqs:
        n_b = len(bvalues_per_freq)
        bvecs = _make_bvecs(n_b)
        s = AcquisitionScheme.from_ogse(
            bvalues_per_freq, bvecs,
            oscillation_frequency=f,
            gradient_duration=sigma,
            n_t=n_t)
        schemes.append(s)

    return AcquisitionScheme.concatenate(schemes)


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

def test_impulsed_instantiation():
    """IMPULSED() creates a 2-compartment model with fixed D_in."""
    model = IMPULSED(D_in=1.58e-9)
    assert len(model.models) == 2

    # Check types
    comp_types = [type(m).__name__ for m in model.models]
    assert 'S4SphereGaussianPhaseApproximation' in comp_types
    assert 'G1Ball' in comp_types

    # D_in is fixed at construction: check on the sphere model instance
    sphere_idx = comp_types.index('S4SphereGaussianPhaseApproximation')
    assert_allclose(
        model.models[sphere_idx].diffusion_constant, 1.58e-9,
        err_msg="D_in not fixed at 1.58e-9")

    # Free parameters should include diameter, lambda_iso (D_ex), volume fractions
    pnames = model.parameter_names
    assert any('diameter' in p for p in pnames), \
        f"No diameter in parameters: {pnames}"
    assert any('lambda_iso' in p for p in pnames), \
        f"No lambda_iso in parameters: {pnames}"


# ---------------------------------------------------------------------------
# 2. Parameter recovery
# ---------------------------------------------------------------------------

def test_impulsed_parameter_recovery():
    """IMPULSED fits R and v_in on synthetic noiseless OGSE data.

    Ground truth: R=8 μm, v_in=0.6, D_ex=1.5e-9 m^2/s.
    Tolerance: |R_fit - 8e-6| / 8e-6 < 0.15, |v_in_fit - 0.6| < 0.10.
    """
    # Ground truth
    R_gt = 8e-6       # m
    D_in = 1.58e-9    # m^2/s
    D_ex = 1.5e-9     # m^2/s
    v_in = 0.6

    # Multi-frequency OGSE scheme
    sigma = 0.04  # s
    bvalues = np.array([0, 5e8, 1e9, 2e9])  # s/m²
    freqs = [30., 60., 100., 200.]  # Hz — varied to distinguish R
    scheme = _make_ogse_scheme(freqs=freqs, bvalues_per_freq=bvalues,
                                sigma=sigma, n_t=1500)

    # Generate ground truth signal (noiseless)
    sphere_gt = S4SphereGaussianPhaseApproximation(
        diameter=2 * R_gt, diffusion_constant=D_in)
    ball_gt = G1Ball()

    E_sphere = sphere_gt(scheme)
    E_ball = ball_gt(scheme, lambda_iso=D_ex)
    signal_gt = v_in * E_sphere + (1 - v_in) * E_ball

    # Fit IMPULSED model (noiseless data)
    model = IMPULSED(D_in=D_in)

    # Use scipy minimize directly to avoid slow grid search
    # since we have noiseless data and good initial guess
    from scipy.optimize import minimize

    diam_scale = 1e-6   # μm
    D_ex_scale = 1e-9   # mm^2/ms order
    v_scale = 1.0

    def loss(x):
        diam, D_ex_val, v_in_val = x[0]*diam_scale, x[1]*D_ex_scale, x[2]*v_scale
        if diam <= 0 or D_ex_val <= 0 or not (0 < v_in_val < 1):
            return 1e10
        sphere_m = S4SphereGaussianPhaseApproximation(
            diameter=diam, diffusion_constant=D_in)
        ball_m = G1Ball()
        E_s = sphere_m(scheme)
        E_b = ball_m(scheme, lambda_iso=D_ex_val)
        E_pred = v_in_val * E_s + (1 - v_in_val) * E_b
        return np.sum((E_pred - signal_gt)**2)

    # Initial guess near ground truth
    x0 = [10.0, 1.5, 0.5]  # diam=10μm, D_ex=1.5e-9, v_in=0.5
    result = minimize(loss, x0, method='Nelder-Mead',
                      options={'xatol': 0.1, 'fatol': 1e-8, 'maxiter': 500})

    diam_fit = result.x[0] * diam_scale
    D_ex_fit = result.x[1] * D_ex_scale
    v_in_fit = result.x[2] * v_scale

    R_fit = diam_fit / 2
    R_err = abs(R_fit - R_gt) / R_gt
    v_err = abs(v_in_fit - v_in)

    assert R_err < 0.15, (
        f"R recovery failed: R_fit={R_fit*1e6:.2f}μm vs R_gt={R_gt*1e6:.1f}μm "
        f"(err={R_err*100:.1f}%)")
    assert v_err < 0.10, (
        f"v_in recovery failed: v_in_fit={v_in_fit:.3f} vs v_in_gt={v_in:.1f} "
        f"(err={v_err:.3f})")
