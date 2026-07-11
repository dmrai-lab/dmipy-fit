"""Tests for OGSE AcquisitionScheme additions (Phase 1).

Covers:
- from_ogse() constructor
- oscillation_frequency / gradient_rise_time / n_oscillation_cycles fields
- is_ogse property
- from_pgse() leaves OGSE fields as None
- concatenate() for mixed PGSE + OGSE schemes
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.core.constants import CONSTANTS


GAMMA = CONSTANTS['water_gyromagnetic_ratio']


def _make_bvecs(n_m):
    """n_m identical unit vectors along x."""
    return np.tile(np.r_[1., 0., 0.], (n_m, 1))


def _make_pgse_scheme(n_m=4):
    bvalues = np.linspace(0, 2e9, n_m)
    bvecs = _make_bvecs(n_m)
    return AcquisitionScheme.from_pgse(
        bvalues, bvecs, delta=0.02, Delta=0.04)


def _make_ogse_scheme(n_m=3, f=50.0, sigma=0.04, b_max=2e9):
    bvalues = np.linspace(0, b_max, n_m)
    bvecs = _make_bvecs(n_m)
    return AcquisitionScheme.from_ogse(
        bvalues, bvecs, oscillation_frequency=f,
        gradient_duration=sigma)


# ---------------------------------------------------------------------------
# test_from_ogse_bvalues_roundtrip
# ---------------------------------------------------------------------------

def test_from_ogse_bvalues_roundtrip():
    """b-values computed numerically from stored G(t) match input within 1%."""
    n_m = 5
    f = 50.0        # Hz
    sigma = 0.04    # s
    b_vals = np.array([0, 5e8, 1e9, 1.5e9, 2e9])
    bvecs = _make_bvecs(n_m)
    scheme = AcquisitionScheme.from_ogse(
        b_vals, bvecs, oscillation_frequency=f,
        gradient_duration=sigma, n_t=2000)

    # Recompute b from stored G(t) via γ² ∫|q(t)|² dt
    G = scheme._G.astype(np.float64)      # (n_m, n_t, 3)
    dt = scheme._dt
    q = np.cumsum(G * dt, axis=1) * GAMMA  # (n_m, n_t, 3)
    q_sq = np.sum(q ** 2, axis=2)          # (n_m, n_t)
    b_num = np.trapezoid(q_sq, dx=dt, axis=1)  # (n_m,)

    # b=0 rows: both should be ~0
    for m in range(n_m):
        if b_vals[m] <= 1e6:
            assert b_num[m] < 1e6, f"b=0 row {m} has non-zero numerical b={b_num[m]}"
        else:
            assert_allclose(
                b_num[m], b_vals[m], rtol=0.01,
                err_msg=f"b-value roundtrip fails at measurement {m}: "
                        f"expected {b_vals[m]:.3e}, got {b_num[m]:.3e}")


# ---------------------------------------------------------------------------
# test_from_ogse_stores_fields
# ---------------------------------------------------------------------------

def test_from_ogse_stores_fields():
    """from_ogse stores oscillation_frequency, gradient_rise_time, n_oscillation_cycles."""
    f = 75.0
    sigma = 0.030
    n_cyc = 2
    t_r = 1e-3
    n_m = 4
    bvalues = np.linspace(0, 2e9, n_m)
    bvecs = _make_bvecs(n_m)

    scheme = AcquisitionScheme.from_ogse(
        bvalues, bvecs,
        oscillation_frequency=f,
        gradient_duration=sigma,
        n_cycles=n_cyc,
        gradient_rise_time=t_r)

    assert scheme.oscillation_frequency is not None
    assert_allclose(scheme.oscillation_frequency, np.full(n_m, f))
    assert_allclose(scheme.gradient_rise_time, np.full(n_m, t_r))
    assert_allclose(scheme.n_oscillation_cycles, np.full(n_m, float(n_cyc)))
    assert scheme.number_of_measurements == n_m


# ---------------------------------------------------------------------------
# test_from_pgse_has_no_ogse_fields
# ---------------------------------------------------------------------------

def test_from_pgse_has_no_ogse_fields():
    """AcquisitionScheme.from_pgse leaves OGSE fields as None."""
    scheme = _make_pgse_scheme(n_m=4)
    assert scheme.oscillation_frequency is None, (
        "from_pgse should not set oscillation_frequency")
    assert scheme.gradient_rise_time is None
    assert scheme.n_oscillation_cycles is None


# ---------------------------------------------------------------------------
# test_concatenate_pgse_ogse
# ---------------------------------------------------------------------------

def test_concatenate_pgse_ogse():
    """Concatenating a 4-measurement PGSE + 3-measurement OGSE gives 7 measurements."""
    pgse = _make_pgse_scheme(n_m=4)
    ogse = _make_ogse_scheme(n_m=3)

    mixed = AcquisitionScheme.concatenate([pgse, ogse])

    assert mixed.number_of_measurements == 7
    assert mixed.bvalues.shape == (7,)
    assert mixed.gradient_directions.shape == (7, 3)
    # First 4 are PGSE (osc_freq=0), last 3 are OGSE (osc_freq=50 Hz)
    assert_allclose(mixed.oscillation_frequency[:4], 0.0)
    assert_allclose(mixed.oscillation_frequency[4:], 50.0)


# ---------------------------------------------------------------------------
# test_is_ogse_property
# ---------------------------------------------------------------------------

def test_is_ogse_property():
    """is_ogse returns correct boolean array for mixed scheme."""
    pgse = _make_pgse_scheme(n_m=4)
    ogse = _make_ogse_scheme(n_m=3)
    mixed = AcquisitionScheme.concatenate([pgse, ogse])

    expected = np.array([False, False, False, False, True, True, True])
    # b=0 OGSE measurements have G=0, osc_freq is still set to f
    # (only the b>0 OGSE rows matter for is_ogse — all 3 have freq=50 > 0)
    assert_array_equal(mixed.is_ogse, expected)


def test_is_ogse_pgse_only():
    """Pure PGSE scheme: is_ogse is all False."""
    scheme = _make_pgse_scheme(n_m=6)
    assert not np.any(scheme.is_ogse)


def test_is_ogse_ogse_only():
    """Pure OGSE scheme: is_ogse is all True (freq > 0)."""
    scheme = _make_ogse_scheme(n_m=4)
    # All measurements have the oscillation frequency set (even b=0)
    assert np.all(scheme.oscillation_frequency == 50.0)
    # is_ogse: True wherever osc_freq > 0
    assert np.all(scheme.is_ogse)
