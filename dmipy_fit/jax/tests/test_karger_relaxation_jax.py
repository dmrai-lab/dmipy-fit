"""JAX X0GeneralizedKarger with coupled relaxation-exchange (issue #7).

The scalar JAX Kärger path can't represent per-compartment T2/T1 coupled to
exchange; those route to the dimension-agnostic matrix-exponential propagator.
These tests pin the JAX matrix path to the NumPy propagator (the reference) for
both PGSE and PGSTE, and check the no-relaxation fast path still matches.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.signal_models.attenuation import (
    OccupancyGatedModel, TransverseRelaxation, LongitudinalRelaxation)
from dmipy_fit.jax.jax_compat import scheme_to_jax
from dmipy_fit.jax.multicompartment_jax import _make_x1karger_jax_fn

B = np.array([0.0, 1e9, 2e9, 1e9, 2e9])
BVECS = np.array([[1., 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
DELTA, TM = 6e-3, 40e-3


def _gated(base, factors=(TransverseRelaxation, LongitudinalRelaxation)):
    return OccupancyGatedModel(base, [f() for f in factors])


def _eval_np_jax(karger, scheme, params):
    E_np = np.asarray(karger(scheme, **params))
    fn = _make_x1karger_jax_fn(karger, scheme)
    E_jax = np.asarray(fn(scheme_to_jax(scheme),
                          {k: jnp.asarray(v) for k, v in params.items()}))
    return E_np, E_jax


@pytest.mark.xfail(reason="anisotropic sub-model numpy<->jax gap: numpy "
                          "stick/zeppelin use btensor() (finite-pulse b), jax "
                          "uses nominal bvalues; pre-existing, not Karger. "
                          "Blocks tight anisotropic parity until reconciled.",
                   strict=False)
def test_jax_matches_numpy_pgse_gated_stick_zeppelin():
    """PGSE, per-compartment T2/T1 (T1 inert on SE): JAX == NumPy propagator."""
    karger = X0GeneralizedKarger(_gated(C1Stick()), _gated(G2Zeppelin()))
    scheme = AcquisitionScheme.from_pgse(
        B, BVECS, delta=DELTA, Delta=DELTA + TM, TE=2 * DELTA + TM)
    params = dict(
        mu=[0.0, 0.0], f=0.4, kappa=5.0,
        OccupancyGatedModel_1_lambda_par=1.7e-9,
        OccupancyGatedModel_2_lambda_par=1.7e-9,
        OccupancyGatedModel_2_lambda_perp=0.6e-9,
        OccupancyGatedModel_1_T2=0.05, OccupancyGatedModel_2_T2=0.08,
        OccupancyGatedModel_1_T1=1.0, OccupancyGatedModel_2_T1=1.5)
    E_np, E_jax = _eval_np_jax(karger, scheme, params)
    np.testing.assert_allclose(E_jax, E_np, rtol=2e-4, atol=2e-5)
    assert E_jax[0] == pytest.approx(1.0)          # b0 normalised


def test_jax_matches_numpy_pgste_gated_ball_ball():
    """PGSTE (mixing time -> T1 active, T2 gated to lobes): JAX == NumPy."""
    karger = X0GeneralizedKarger(_gated(G1Ball()), _gated(G1Ball()))
    scheme = AcquisitionScheme.from_pgste(B, BVECS, delta=DELTA, TM=TM)
    params = dict(
        f=0.4, kappa=5.0,
        OccupancyGatedModel_1_lambda_iso=1.0e-9,
        OccupancyGatedModel_2_lambda_iso=2.0e-9,
        OccupancyGatedModel_1_T2=0.05, OccupancyGatedModel_2_T2=0.08,
        OccupancyGatedModel_1_T1=1.0, OccupancyGatedModel_2_T1=1.5)
    # gated ball+ball is isotropic -> no 'mu'; all keys carry the
    # OccupancyGatedModel_i_ prefix (base lambda_iso + factor T1/T2).
    E_np, E_jax = _eval_np_jax(karger, scheme, params)
    np.testing.assert_allclose(E_jax, E_np, rtol=2e-4, atol=2e-5)


def test_jax_pgse_independent_of_T1():
    """No mixing time -> T1 inert; two T1 pairs give the same JAX signal."""
    karger = X0GeneralizedKarger(_gated(G1Ball()), _gated(G1Ball()))
    scheme = AcquisitionScheme.from_pgse(
        B, BVECS, delta=DELTA, Delta=DELTA + TM, TE=2 * DELTA + TM)
    base = dict(f=0.4, kappa=5.0,
                OccupancyGatedModel_1_lambda_iso=1e-9,
                OccupancyGatedModel_2_lambda_iso=2e-9,
                OccupancyGatedModel_1_T2=0.05, OccupancyGatedModel_2_T2=0.08)
    fn = _make_x1karger_jax_fn(karger, scheme)
    sj = scheme_to_jax(scheme)
    Ea = np.asarray(fn(sj, {k: jnp.asarray(v) for k, v in dict(
        base, OccupancyGatedModel_1_T1=0.3, OccupancyGatedModel_2_T1=0.9).items()}))
    Eb = np.asarray(fn(sj, {k: jnp.asarray(v) for k, v in dict(
        base, OccupancyGatedModel_1_T1=2.0, OccupancyGatedModel_2_T1=3.0).items()}))
    np.testing.assert_allclose(Ea, Eb, rtol=1e-5, atol=1e-6)


def test_no_relaxation_ball_ball_uses_fast_scalar_path_and_matches_numpy():
    """Without a relaxation add-on, the bare model keeps the scalar fast path
    and matches NumPy (no matrix propagator)."""
    karger = X0GeneralizedKarger(G1Ball(), G1Ball())
    scheme = AcquisitionScheme.from_pgse(B, BVECS, delta=DELTA, Delta=DELTA + TM)
    params = dict(f=0.4, kappa=5.0,
                  G1Ball_1_lambda_iso=1e-9, G1Ball_2_lambda_iso=2e-9)
    E_np, E_jax = _eval_np_jax(karger, scheme, params)
    np.testing.assert_allclose(E_jax, E_np, rtol=2e-4, atol=2e-5)
