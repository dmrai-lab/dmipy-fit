"""Noise-free recovery for the spherical-mean factory models (sandi, mcsmt).

Each model's own forward is simulated at a mid-range point on a rich
multi-shell scheme and fitted with every solver the model supports.

Prediction: the data are reproduced to the optimizer tolerance and every
parameter the scheme identifies is recovered. SANDI's neurite diffusivity and
neurite fraction are the exception: along (D_in, f_in) the signal changes by
< 5e-4 over a 20 % move (the stick's b^-1/2 tail), so only the data fit, the
soma diameter and the soma fraction are asserted tightly.
"""
import numpy as np
import pytest

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.custom_optimizers import reference_models as rm


def _scheme(b, per_shell, delta, Delta):
    rng = np.random.default_rng(0)
    d = rng.normal(size=(per_shell * len(b), 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    bv = np.r_[0.0, np.repeat(np.asarray(b, dtype=float), per_shell)]
    d = np.vstack([[[1.0, 0.0, 0.0]], d])
    return acquisition_scheme_from_bvalues(bv, d, delta=delta, Delta=Delta)


SANDI_SCHEME = dict(b=np.array([0.3, 0.6, 1, 1.5, 2, 3, 4, 5, 6, 8]) * 1e9,
                    per_shell=6, delta=0.006, Delta=0.02)
SANDI_TRUTH = {
    "S4SphereGaussianPhaseApproximation_1_diameter": 12e-6,
    "C1Stick_1_lambda_par": 1.2e-9, "G1Ball_1_lambda_iso": 1.1e-9,
    "partial_volume_0": 0.33, "partial_volume_1": 0.32,
    "partial_volume_2": 0.35}
MCSMT_SCHEME = dict(b=(1e9, 2e9, 3e9, 4e9), per_shell=12, delta=0.0106,
                    Delta=0.0431)
MCSMT_TRUTH = {"C1Stick_1_lambda_par": 1.7e-9, "partial_volume_0": 0.4,
               "partial_volume_1": 0.6}


def _fit(model, scheme, truth, solver):
    data = np.asarray(model.simulate_signal(scheme, truth))
    fit = model.fit(scheme, data, solver=solver)
    got = {k: float(np.squeeze(v)) for k, v in fit.fitted_parameters.items()}
    return data, got, np.asarray(model.simulate_signal(scheme, got))


def _check_sandi(solver):
    model, scheme = rm.sandi(), _scheme(**SANDI_SCHEME)
    data, got, refit = _fit(model, scheme, SANDI_TRUTH, solver)
    assert np.abs(refit - data).max() < 1e-4
    diam = "S4SphereGaussianPhaseApproximation_1_diameter"
    assert got[diam] == pytest.approx(SANDI_TRUTH[diam], rel=0.02)
    assert got["partial_volume_0"] == pytest.approx(0.33, abs=0.02)
    assert got["G1Ball_1_lambda_iso"] == pytest.approx(1.1e-9, rel=0.05)
    assert got["C1Stick_1_lambda_par"] == pytest.approx(1.2e-9, rel=0.3)


def test_sandi_recovery_brute2fine():
    _check_sandi("brute2fine")


def test_sandi_recovery_mix():
    _check_sandi("mix")


@pytest.mark.slow
def test_sandi_recovery_jax_default():
    """The default JAX precision recovers the soma diameter: the model declares float64."""
    jax = pytest.importorskip("jax")
    assert not jax.config.jax_enable_x64
    _check_sandi("jax")
    assert not jax.config.jax_enable_x64       # the ambient setting is restored


def test_float64_models_are_declared():
    pytest.importorskip("jax")
    from dmipy_fit.jax.precision import float64_compartments
    assert float64_compartments(rm.sandi()) == [
        "S4SphereGaussianPhaseApproximation"]
    assert float64_compartments(rm.mcsmt()) == []


def test_jax_refuses_by_name_when_float64_is_unavailable(monkeypatch):
    import contextlib
    jax = pytest.importorskip("jax")
    monkeypatch.setattr(jax, "enable_x64", lambda flag: contextlib.nullcontext())
    model, scheme = rm.sandi(), _scheme(**SANDI_SCHEME)
    data = np.asarray(model.simulate_signal(scheme, SANDI_TRUTH))
    with pytest.raises(ValueError, match="S4SphereGaussianPhaseApproximation"):
        model.fit(scheme, data, solver="jax")


def test_mcsmt_recovery_brute2fine():
    model, scheme = rm.mcsmt(), _scheme(**MCSMT_SCHEME)
    data, got, refit = _fit(model, scheme, MCSMT_TRUTH, "brute2fine")
    assert np.abs(refit - data).max() < 1e-5
    for k, v in MCSMT_TRUTH.items():
        assert got[k] == pytest.approx(v, rel=1e-3)


@pytest.mark.slow
def test_mcsmt_recovery_jax():
    """A float32 model is fitted in float32: the ambient setting is untouched."""
    jax = pytest.importorskip("jax")
    assert not jax.config.jax_enable_x64
    model, scheme = rm.mcsmt(), _scheme(**MCSMT_SCHEME)
    _, got, _ = _fit(model, scheme, MCSMT_TRUTH, "jax")
    for k, v in MCSMT_TRUTH.items():
        assert got[k] == pytest.approx(v, rel=1e-3)


def test_mcsmt_mix_is_refused_with_the_tortuosity_constraint():
    model, scheme = rm.mcsmt(), _scheme(**MCSMT_SCHEME)
    data = np.asarray(model.simulate_signal(scheme, MCSMT_TRUTH))
    with pytest.raises(ValueError, match="Tortuosity"):
        model.fit(scheme, data, solver="mix")


@pytest.mark.slow
def test_full_model_with_sphere_is_fitted_in_float64():
    """The full-model JAX path carries the required precision through the brute grid and the fit."""
    jax = pytest.importorskip("jax")
    from dmipy_fit.core.modeling_framework import MultiCompartmentModel
    from dmipy_fit.signal_models.gaussian_models import G1Ball
    from dmipy_fit.signal_models.sphere_models import (
        S4SphereGaussianPhaseApproximation)
    model = MultiCompartmentModel(
        [S4SphereGaussianPhaseApproximation(diffusion_constant=3e-9), G1Ball()])
    scheme = _scheme(**SANDI_SCHEME)
    truth = {"S4SphereGaussianPhaseApproximation_1_diameter": 12e-6,
             "G1Ball_1_lambda_iso": 1.1e-9,
             "partial_volume_0": 0.4, "partial_volume_1": 0.6}
    _, got, _ = _fit(model, scheme, truth, "jax")
    assert not jax.config.jax_enable_x64
    for k, v in truth.items():
        assert got[k] == pytest.approx(v, rel=0.02)
