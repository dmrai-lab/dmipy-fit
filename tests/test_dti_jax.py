"""Tests for dti_jax: JAX DTI fitter accuracy and JaxOptimizer.make_x0.

All tests use synthetic data with known ground-truth tensors so they
don't require on-disk brain data and pass on CPU or GPU.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from dmipy_fit.jax.dti_jax import build_dti_fitter, detect_mu_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheme(bvals_mm2, bvecs):
    """Build a dmipy_fit AcquisitionScheme (bvalues in s/m²)."""
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    bvals_si = np.array(bvals_mm2, dtype=np.float64) * 1e6  # s/mm² → s/m²
    return acquisition_scheme_from_bvalues(bvals_si, np.array(bvecs, dtype=np.float64))


def _pgse_scheme(n_b0=5, b_mm2=1000.0, n_dw=64):
    """Single-shell PGSE scheme with random gradient directions.

    Returns a full dmipy_fit AcquisitionScheme (needed by JaxOptimizer).
    """
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n_dw, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    bvals_si = np.concatenate([np.zeros(n_b0), np.full(n_dw, b_mm2)]) * 1e6  # s/m²
    bvecs = np.concatenate([np.zeros((n_b0, 3)), vecs])
    return acquisition_scheme_from_bvalues(bvals_si, bvecs)


def _simulate_dti_signal(scheme, d_tensor, S0=1.0):
    """Generate noiseless DTI signal for a 3×3 tensor.

    Parameters
    ----------
    d_tensor : (3, 3) symmetric tensor in mm²/s
    """
    bvals = np.array(scheme.bvalues) * 1e-6   # s/m² → s/mm²
    bvecs = np.array(scheme.gradient_directions)
    attenuation = np.exp(-bvals * np.einsum('ni,ij,nj->n', bvecs, d_tensor, bvecs))
    return (S0 * attenuation).astype(np.float32)


def _axisymmetric_tensor(theta, phi, d_par=1.7e-3, d_perp=0.3e-3):
    """Build axisymmetric tensor aligned to (theta, phi) spherical coords."""
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    v = np.array([vx, vy, vz])
    # D = d_par * v⊗v + d_perp * (I - v⊗v)
    return d_par * np.outer(v, v) + d_perp * (np.eye(3) - np.outer(v, v))


# ---------------------------------------------------------------------------
# Tests: build_dti_fitter
# ---------------------------------------------------------------------------

class TestBuildDtiFitter:
    def test_returns_callable(self):
        scheme = _pgse_scheme()
        fn = build_dti_fitter(scheme)
        assert callable(fn)

    def test_lru_cache_same_object(self):
        """Same scheme → same compiled function (no recompile)."""
        scheme = _pgse_scheme()
        fn1 = build_dti_fitter(scheme)
        fn2 = build_dti_fitter(scheme)
        assert fn1 is fn2

    def test_output_shapes(self):
        scheme = _pgse_scheme()
        fn = build_dti_fitter(scheme)
        N = 32
        d = _axisymmetric_tensor(np.pi / 4, np.pi / 6)
        sig = _simulate_dti_signal(scheme, d)
        data = jnp.tile(jnp.array(sig)[None], (N, 1))
        mu, fa = fn(data)
        assert mu.shape == (N, 2)
        assert fa.shape == (N,)

    def test_fa_range(self):
        """FA should always be in [0, 1]."""
        scheme = _pgse_scheme()
        fn = build_dti_fitter(scheme)
        rng = np.random.default_rng(7)
        thetas = rng.uniform(0.1, np.pi - 0.1, 20)
        phis = rng.uniform(-np.pi, np.pi, 20)
        signals = np.array([
            _simulate_dti_signal(scheme, _axisymmetric_tensor(th, ph))
            for th, ph in zip(thetas, phis)
        ])
        mu, fa = fn(jnp.array(signals))
        fa_np = np.array(fa)
        assert fa_np.min() >= 0.0
        assert fa_np.max() <= 1.0

    def test_isotropic_fa_near_zero(self):
        """Isotropic tensor → FA ≈ 0."""
        scheme = _pgse_scheme()
        fn = build_dti_fitter(scheme)
        d_iso = 1.0e-3 * np.eye(3)
        sig = _simulate_dti_signal(scheme, d_iso)
        _, fa = fn(jnp.array(sig[None]))
        assert float(fa[0]) < 0.05

    def test_anisotropic_fa_high(self):
        """High-anisotropy tensor → FA > 0.7."""
        scheme = _pgse_scheme()
        fn = build_dti_fitter(scheme)
        d = _axisymmetric_tensor(np.pi / 3, np.pi / 5, d_par=1.7e-3, d_perp=0.3e-3)
        sig = _simulate_dti_signal(scheme, d)
        _, fa = fn(jnp.array(sig[None]))
        assert float(fa[0]) > 0.7

    def test_orientation_accuracy_degrees(self):
        """Principal eigenvector accurate to < 5° on noiseless data."""
        scheme = _pgse_scheme(n_dw=60, b_mm2=1000.0)
        fn = build_dti_fitter(scheme)
        rng = np.random.default_rng(3)
        N = 50
        thetas_gt = rng.uniform(0.1, np.pi - 0.1, N)
        phis_gt = rng.uniform(-np.pi, np.pi, N)
        signals = np.array([
            _simulate_dti_signal(scheme, _axisymmetric_tensor(th, ph))
            for th, ph in zip(thetas_gt, phis_gt)
        ])
        mu, fa = fn(jnp.array(signals))
        mu_np = np.array(mu)
        fa_np = np.array(fa)

        # Convert spherical → Cartesian
        th_j, ph_j = mu_np[:, 0], mu_np[:, 1]
        mu_xyz = np.stack([
            np.sin(th_j) * np.cos(ph_j),
            np.sin(th_j) * np.sin(ph_j),
            np.cos(th_j),
        ], axis=1)
        gt_xyz = np.stack([
            np.sin(thetas_gt) * np.cos(phis_gt),
            np.sin(thetas_gt) * np.sin(phis_gt),
            np.cos(thetas_gt),
        ], axis=1)
        dot = np.clip(np.abs((mu_xyz * gt_xyz).sum(1)), 0, 1)
        angle_deg = np.degrees(np.arccos(dot))

        # On high-FA noiseless data the error should be tiny
        high_fa = fa_np > 0.5
        if high_fa.sum() > 5:
            assert angle_deg[high_fa].mean() < 5.0, (
                f"Mean orientation error {angle_deg[high_fa].mean():.1f}° > 5°"
            )

    def test_b_max_restricts_measurements(self):
        """b_max=1.5e9 should exclude high-b measurements."""
        # Multi-shell scheme
        rng = np.random.default_rng(11)
        vecs = rng.standard_normal((60, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        bvals = np.concatenate([
            np.zeros(5),
            np.full(30, 1000.0),
            np.full(30, 3000.0),
        ])
        bvecs = np.concatenate([np.zeros((5, 3)), vecs])
        scheme = _make_scheme(bvals, bvecs)

        fn_all = build_dti_fitter(scheme, b_max=None)
        fn_b15 = build_dti_fitter(scheme, b_max=1.5e9)

        # Different b_max → different objects (not same cache entry)
        assert fn_all is not fn_b15

        d = _axisymmetric_tensor(np.pi / 4, np.pi / 3)
        sig = _simulate_dti_signal(scheme, d)
        data = jnp.array(sig[None])
        _, fa_all = fn_all(data)
        _, fa_b15 = fn_b15(data)
        # b_max version uses only b=1000 — FA should be closer to ground truth
        # (>0.7 for d_par=1.7e-3, d_perp=0.3e-3)
        assert float(fa_b15[0]) > 0.7, f"FA with b_max=1500: {fa_b15[0]:.3f}"


# ---------------------------------------------------------------------------
# Tests: detect_mu_indices
# ---------------------------------------------------------------------------

class TestDetectMuIndices:
    def test_empty_for_no_mu(self):
        """Model with no _mu params → empty list."""
        class FakeModel:
            parameter_cardinality = {'f_intra': 1, 'odi': 1}
            parameter_optimization_flags = {}
        assert detect_mu_indices(FakeModel()) == []

    def test_single_mu(self):
        """Single _mu with cardinality 2 → one entry."""
        class FakeModel:
            parameter_cardinality = {'f_intra': 1, 'odi': 1, 'bundle1_mu': 2}
            parameter_optimization_flags = {}
        result = detect_mu_indices(FakeModel())
        assert len(result) == 1
        name, i0, i1 = result[0]
        assert name == 'bundle1_mu'
        assert i1 - i0 == 2
        assert i0 == 2   # after f_intra (1) + odi (1)

    def test_mu_indices_skip_fixed(self):
        """Fixed params (flag=False) don't count toward the index."""
        class FakeModel:
            parameter_cardinality = {'fixed_param': 1, 'odi': 1, 'bundle1_mu': 2}
            parameter_optimization_flags = {'fixed_param': False}
        result = detect_mu_indices(FakeModel())
        assert len(result) == 1
        _, i0, _ = result[0]
        assert i0 == 1   # only odi (1) counted; fixed_param skipped

    def test_two_mu_params(self):
        """Two _mu params → two entries in order."""
        class FakeModel:
            parameter_cardinality = {'b1_mu': 2, 'b2_mu': 2}
            parameter_optimization_flags = {}
        result = detect_mu_indices(FakeModel())
        assert len(result) == 2
        assert result[0][1] == 0 and result[0][2] == 2
        assert result[1][1] == 2 and result[1][2] == 4


# ---------------------------------------------------------------------------
# Tests: JaxOptimizer.make_x0
# ---------------------------------------------------------------------------

class TestMakeX0:
    def _stick_optimizer(self, warm_start_mu=True):
        """Build JaxOptimizer with single-bundle C1Stick + warm_start_mu."""
        from dmipy_fit.signal_models.cylinder_models import C1Stick
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel
        from dmipy_fit.jax.optimizers_jax import JaxOptimizer

        scheme = _pgse_scheme(n_b0=5, b_mm2=1000.0, n_dw=60)
        model = MultiCompartmentModel([C1Stick()])
        return JaxOptimizer(model, scheme, maxiter=5, warm_start_mu=warm_start_mu), scheme

    def test_make_x0_shape(self):
        opt, scheme = self._stick_optimizer()
        d = _axisymmetric_tensor(np.pi / 4, np.pi / 5)
        sig = _simulate_dti_signal(scheme, d)
        data = jnp.array(np.tile(sig, (16, 1)))
        x0 = opt.make_x0(data)
        N_params = len(opt._lower)   # size of nested parameter vector
        assert x0.shape == (16, N_params)

    def test_make_x0_mu_not_midpoint(self):
        """With warm_start_mu=True, orientation indices should NOT all be midpoint."""
        opt, scheme = self._stick_optimizer(warm_start_mu=True)
        rng = np.random.default_rng(99)
        thetas = rng.uniform(0.2, np.pi - 0.2, 20)
        phis = rng.uniform(-np.pi, np.pi, 20)
        signals = np.array([
            _simulate_dti_signal(scheme, _axisymmetric_tensor(th, ph))
            for th, ph in zip(thetas, phis)
        ])
        data = jnp.array(signals)
        x0 = np.array(opt.make_x0(data))

        mu_s = opt._mu_slice
        # Midpoint for theta is π/2 ≈ 1.57, for phi is 0 — check that not all
        # voxels are at midpoint (warm-start has spread the orientations)
        theta_vals = x0[:, mu_s][:, 0]
        assert theta_vals.std() > 0.1, "Orientations stuck at midpoint — warm-start not working"

    def test_make_x0_midpoint_without_warm_start(self):
        """Without warm_start_mu, all x0 rows are identical midpoints."""
        opt, scheme = self._stick_optimizer(warm_start_mu=False)
        d = _axisymmetric_tensor(np.pi / 4, np.pi / 5)
        sig = _simulate_dti_signal(scheme, d)
        data = jnp.array(np.tile(sig, (16, 1)))
        x0 = np.array(opt.make_x0(data))
        # All rows identical (midpoints)
        assert np.allclose(x0, x0[0:1, :]), "Expected all-midpoint x0 without warm_start_mu"

    def test_warm_start_mu_raises_for_multi_bundle(self):
        """warm_start_mu=True on two-orientation model raises ValueError."""
        from dmipy_fit.signal_models.cylinder_models import C1Stick
        from dmipy_fit.signal_models.gaussian_models import G2Zeppelin
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel
        from dmipy_fit.jax.optimizers_jax import JaxOptimizer

        scheme = _pgse_scheme()
        model = MultiCompartmentModel([C1Stick(), G2Zeppelin()])
        with pytest.raises(ValueError, match="exactly one orientation"):
            JaxOptimizer(model, scheme, maxiter=5, warm_start_mu=True)
