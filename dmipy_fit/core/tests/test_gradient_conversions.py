"""Tests for dmipy/core/gradient_conversions.py

All conversions use SI units:
  b-value  : s/m²  (typical dMRI: ~1e9)
  q-value  : 1/m   (typical: ~1e4–1e5)
  g        : T/m   (typical clinical: ~40–80 mT/m = 0.04–0.08 T/m)
  delta    : s     (gradient pulse duration, typical: ~0.01–0.03)
  Delta    : s     (diffusion time, typical: ~0.02–0.06)
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_fit.core.gradient_conversions import (
    b_from_g, b_from_q,
    g_from_b, g_from_q,
    q_from_b, q_from_g,
)
from dmipy_fit.core.constants import CONSTANTS

GAMMA = CONSTANTS['water_gyromagnetic_ratio']  # 267.513e6 rad/(s·T)

# Typical HCP-style pulse parameters
DELTA = 0.0106   # s
DELTA_BIG = 0.0431  # s


class TestRoundTrips:
    """Each conversion pair must be exact inverses of each other."""

    def test_b_q_roundtrip(self):
        b = 1e9  # s/m²
        q = q_from_b(b, DELTA, DELTA_BIG)
        b_back = b_from_q(q, DELTA, DELTA_BIG)
        npt.assert_allclose(b_back, b, rtol=1e-10)

    def test_q_b_roundtrip(self):
        q = 3e4  # 1/m
        b = b_from_q(q, DELTA, DELTA_BIG)
        q_back = q_from_b(b, DELTA, DELTA_BIG)
        npt.assert_allclose(q_back, q, rtol=1e-10)

    def test_b_g_roundtrip(self):
        b = 1e9
        g = g_from_b(b, DELTA, DELTA_BIG)
        b_back = b_from_g(g, DELTA, DELTA_BIG)
        npt.assert_allclose(b_back, b, rtol=1e-10)

    def test_g_b_roundtrip(self):
        g = 0.06  # T/m  (~60 mT/m)
        b = b_from_g(g, DELTA, DELTA_BIG)
        g_back = g_from_b(b, DELTA, DELTA_BIG)
        npt.assert_allclose(g_back, g, rtol=1e-10)

    def test_q_g_roundtrip(self):
        q = 2e4  # 1/m
        g = g_from_q(q, DELTA)
        q_back = q_from_g(g, DELTA)
        npt.assert_allclose(q_back, q, rtol=1e-10)

    def test_g_q_roundtrip(self):
        g = 0.04  # T/m
        q = q_from_g(g, DELTA)
        g_back = g_from_q(q, DELTA)
        npt.assert_allclose(g_back, g, rtol=1e-10)


class TestZeroBValue:
    """Zero b-value (b0) must produce zero q and zero g."""

    def test_q_from_b_zero(self):
        q = q_from_b(0., DELTA, DELTA_BIG)
        assert q == 0.

    def test_g_from_b_zero(self):
        g = g_from_b(0., DELTA, DELTA_BIG)
        assert g == 0.


class TestPhysicalPlausibility:
    """Sanity-check magnitudes against known HCP acquisition values.

    HCP WU-Minn: b=1000 s/mm² = 1e9 s/m², δ≈10.6 ms, Δ≈43.1 ms,
    G≈26 mT/m.  We allow ±50% to be robust to parameter variation.
    """

    def test_g_from_b_magnitude(self):
        b = 1e9   # s/m²
        g = g_from_b(b, DELTA, DELTA_BIG)
        # Expect roughly 20–60 mT/m for typical PGSE parameters
        assert 0.01 < g < 0.1, f"g={g:.4f} T/m outside plausible range"

    def test_q_from_b_magnitude(self):
        b = 1e9
        q = q_from_b(b, DELTA, DELTA_BIG)
        # q should be in the 1e4–1e5 1/m range
        assert 1e4 < q < 1e5, f"q={q:.1f} 1/m outside plausible range"

    def test_b_from_g_magnitude(self):
        g = 0.04  # T/m  (40 mT/m)
        b = b_from_g(g, DELTA, DELTA_BIG)
        # Should be in the 1e8–5e9 s/m² range for these parameters
        assert 1e8 < b < 5e9, f"b={b:.2e} s/m² outside plausible range"


class TestArrayInputs:
    """Conversions must work element-wise on numpy arrays."""

    def test_array_b_to_q(self):
        b = np.array([0., 1e9, 3e9])
        q = q_from_b(b, DELTA, DELTA_BIG)
        assert q.shape == b.shape
        assert q[0] == 0.
        assert np.all(np.diff(q) > 0)  # monotonically increasing

    def test_array_g_to_b(self):
        g = np.array([0.01, 0.04, 0.08])
        b = b_from_g(g, DELTA, DELTA_BIG)
        assert b.shape == g.shape
        assert np.all(np.diff(b) > 0)


class TestGyromagneticRatio:
    """The gyromagnetic ratio constant must be physically correct."""

    def test_gamma_value(self):
        # Water proton gyromagnetic ratio: 267.513e6 rad/(s·T)
        # (NIST value: 267.52218744e6; we use a rounded version)
        npt.assert_allclose(GAMMA, 267.513e6, rtol=1e-3)
