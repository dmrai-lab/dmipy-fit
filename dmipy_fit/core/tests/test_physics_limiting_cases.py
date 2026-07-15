"""
Adversarial physics tests — limiting cases, boundary conditions, and known bugs.

SC-023  _S3SphereCallaghanApproximation fixed + MC-validated (DP-006)
SC-025  SD1WatsonDistributed(C3Callaghan) agrees with numerical orientation average
SC-026  T2 is silently ignored when acquisition scheme has TE=None
SC-027  C3CylinderCallaghan(d→0) approaches C1Stick
SC-028  SD1WatsonDistributed signal bounded to [0,1] for all kappa after renormalization fix.
        Root cause: under-sampling of sharp Watson peak on 362-point hemisphere grid
        causes SH L=0 coefficient drift; fix enforces c_0^0 = 1/(2*sqrt(pi)) analytically.

All tests run <30 s on CPU.
"""
import os
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import (
    C1Stick,
    C3CylinderCallaghanApproximation,
)
from dmipy_fit.signal_models.sphere_models import (
    S1Dot,
    S2SphereStejskalTannerApproximation,
)


# ---------------------------------------------------------------------------
# Shared acquisition scheme fixtures
# ---------------------------------------------------------------------------

def _single_b(b, direction=(1., 0., 0.), delta=0.01, Delta=0.03):
    """One-measurement scheme at given b-value and gradient direction."""
    bvals = np.array([float(b)])
    bvecs = np.atleast_2d(direction).astype(float)
    bvecs /= np.linalg.norm(bvecs)
    return acquisition_scheme_from_bvalues(bvals, bvecs, delta, Delta)


def _multi_b(bvalues, direction=(1., 0., 0.), delta=0.01, Delta=0.03):
    bvals = np.asarray(bvalues, dtype=float)
    bvecs = np.tile(direction, (len(bvals), 1)).astype(float)
    return acquisition_scheme_from_bvalues(bvals, bvecs, delta, Delta)


def _six_directions(b, delta=0.01, Delta=0.03):
    """Six gradient directions at the same b-value (±x, ±y, ±z)."""
    dirs = np.array([
        [1., 0., 0.], [-1., 0., 0.],
        [0., 1., 0.], [0., -1., 0.],
        [0., 0., 1.], [0., 0., -1.],
    ])
    bvals = np.full(6, float(b))
    return acquisition_scheme_from_bvalues(bvals, dirs, delta, Delta)


# =========================================================================
# SC-027 — Parameter boundary / limiting cases (analytical, CPU-only)
# =========================================================================

class TestParameterBoundaries:
    """Tests for parameter limits and boundary conditions.

    All tests are pure analytical (no MC simulation) and run in <1 s.
    """

    # ------------------------------------------------------------------
    # b=0 → E=1.0 for all models
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("model,params", [
        (G1Ball(),                     {"lambda_iso": 2e-9}),
        (G2Zeppelin(),                 {"mu": [0., 0.], "lambda_par": 2e-9, "lambda_perp": 1e-9}),
        (C1Stick(),                    {"mu": [0., 0.], "lambda_par": 2e-9}),
        (C3CylinderCallaghanApproximation(),
                                       {"mu": [0., 0.], "lambda_par": 2e-9, "diameter": 4e-6}),
        (S2SphereStejskalTannerApproximation(), {"diameter": 10e-6}),
        (S1Dot(),                      {}),
    ])
    def test_b_zero_returns_one(self, model, params):
        """SC-new: Every model must return E=1.0 exactly at b=0."""
        scheme = _single_b(0.0)
        E = model(scheme, **params)
        npt.assert_allclose(
            E, 1.0, atol=1e-14,
            err_msg=f"{type(model).__name__} returned E={E} at b=0, expected 1.0")

    # ------------------------------------------------------------------
    # C3CylinderCallaghan(d→0) → C1Stick  (SC-027)
    # ------------------------------------------------------------------

    def test_c3_tiny_radius_approaches_stick_parallel(self):
        """SC-027: C3Callaghan(d=1nm) matches C1Stick at parallel gradient.

        When the gradient is parallel to the fiber axis (theta=0), both the
        Stick and the Callaghan cylinder give exp(-b*lambda_par). With a 1 nm
        cylinder radius the perpendicular restriction is negligible.
        """
        D = 2e-9
        bvals = np.array([1e9, 2e9])
        # Fiber along z, gradient along z (parallel)
        mu_z = [0., 0.]
        bvecs = np.tile([0., 0., 1.], (2, 1))
        scheme = acquisition_scheme_from_bvalues(bvals, bvecs, 0.01, 0.03)

        E_stick = C1Stick()(scheme, mu=mu_z, lambda_par=D)
        E_c3 = C3CylinderCallaghanApproximation()(
            scheme, mu=mu_z, lambda_par=D, diameter=1e-9)

        npt.assert_allclose(
            E_c3, E_stick, atol=1e-6,
            err_msg="C3Callaghan(d=1nm) vs C1Stick at parallel gradient")

    def test_c3_tiny_radius_approaches_stick_perpendicular(self):
        """SC-027: C3Callaghan(d=1nm) → E≈1 at perpendicular gradient (as C1Stick).

        C1Stick with gradient perpendicular to fiber gives E=1.0 (no
        attenuation). A 1 nm cylinder should likewise give E≈1.0.
        """
        D = 2e-9
        bvals = np.array([1e9, 2e9])
        # Fiber along z, gradient along x (perpendicular)
        mu_z = [0., 0.]
        bvecs = np.tile([1., 0., 0.], (2, 1))
        scheme = acquisition_scheme_from_bvalues(bvals, bvecs, 0.01, 0.03)

        E_stick = C1Stick()(scheme, mu=mu_z, lambda_par=D)
        E_c3 = C3CylinderCallaghanApproximation()(
            scheme, mu=mu_z, lambda_par=D, diameter=1e-9)

        # Both should be ~1.0 (stick perpendicular is exactly 1)
        npt.assert_allclose(E_stick, 1.0, atol=1e-14)
        npt.assert_allclose(
            E_c3, 1.0, atol=1e-6,
            err_msg="C3Callaghan(d=1nm) vs 1.0 at perpendicular gradient")

    # ------------------------------------------------------------------
    # G2Zeppelin(λ_perp=λ_par) == G1Ball  (no new SC, supplements SC-022)
    # ------------------------------------------------------------------

    def test_zeppelin_isotropic_limit_equals_ball(self):
        """G2Zeppelin(λ_perp=λ_par=D) == G1Ball(λ_iso=D) at all directions.

        In the isotropic limit the Zeppelin formula exp(-b*(λ_par*(g·n)²
        + λ_perp*(1-(g·n)²))) simplifies to exp(-b*D) independent of direction.
        Supplements the MC validation in SC-022.
        """
        D = 1.7e-9
        scheme = _six_directions(1e9)

        E_ball = G1Ball()(scheme, lambda_iso=D)
        E_zep = G2Zeppelin()(
            scheme, mu=[0., 0.], lambda_par=D, lambda_perp=D)

        npt.assert_allclose(
            E_zep, E_ball, atol=1e-10,
            err_msg="G2Zeppelin isotropic limit differs from G1Ball")

    def test_zeppelin_isotropic_direction_independent(self):
        """G2Zeppelin(λ_perp=λ_par) must give identical signal in all 6 directions."""
        D = 2e-9
        scheme = _six_directions(2e9)
        E_zep = G2Zeppelin()(
            scheme, mu=[0., 0.], lambda_par=D, lambda_perp=D)

        npt.assert_allclose(
            E_zep, E_zep[0], atol=1e-10,
            err_msg="G2Zeppelin isotropic: signal varies across gradient directions")


# =========================================================================
# SC-026 — T2 is an occupancy-gated factor, gated by TE
# =========================================================================

class TestT2OccupancyGatedFactor:
    """SC-026 (refactored): T2 is no longer baked into compartments — it is a
    composable :class:`TransverseRelaxation` factor on
    :class:`OccupancyGatedModel`.  The base compartments are pure diffusion.

    The original SC-026 boundary still holds at the factor level: the T2 factor
    applies ``exp(-TE/T2)`` when the scheme carries a non-None TE, and is a
    no-op (returns 1) when TE=None.  Base compartments never apply T2.
    """

    def _scheme_no_te(self):
        bvals = np.array([0., 1e9])
        bvecs = np.tile([1., 0., 0.], (2, 1))
        return acquisition_scheme_from_bvalues(
            bvals, bvecs, 0.01, 0.03, TE=None)

    def _scheme_with_te(self, te_s=50e-3):
        bvals = np.array([0., 1e9])
        bvecs = np.tile([1., 0., 0.], (2, 1))
        TE = np.full(2, te_s)
        return acquisition_scheme_from_bvalues(
            bvals, bvecs, 0.01, 0.03, TE=TE)

    def _gated(self):
        from dmipy_fit.signal_models.attenuation import (
            OccupancyGatedModel, TransverseRelaxation)
        return OccupancyGatedModel(G1Ball(), [TransverseRelaxation()])

    def test_base_compartments_have_no_t2(self):
        """Base G1Ball/C1Stick/G2Zeppelin are pure diffusion — no T2 parameter."""
        for model in (G1Ball(), C1Stick(), G2Zeppelin()):
            assert not any(k == 'T2' or k.endswith('_T2')
                           for k in model.parameter_ranges)

    def test_t2_factor_is_noop_when_te_none(self):
        """The T2 factor returns the bare diffusion signal when TE=None."""
        scheme = self._scheme_no_te()
        gated = self._gated()
        E_with_T2 = gated(scheme, lambda_iso=2e-9, T2=80e-3)
        E_diff = G1Ball()(scheme, lambda_iso=2e-9)
        npt.assert_array_equal(
            E_with_T2, E_diff,
            err_msg="T2 factor should be a no-op when TE=None")

    def test_t2_factor_is_applied_when_te_is_set(self):
        """The T2 factor multiplies exp(-TE/T2) when the scheme carries TE > 0."""
        te, T2 = 50e-3, 80e-3
        scheme = self._scheme_with_te(te)
        gated = self._gated()
        E_with_T2 = gated(scheme, lambda_iso=2e-9, T2=T2)
        E_diff = G1Ball()(scheme, lambda_iso=2e-9)
        npt.assert_allclose(
            E_with_T2 / E_diff, np.exp(-te / T2), atol=1e-10,
            err_msg="T2 relaxation factor incorrect when TE is set")


# =========================================================================
# SC-023 / DP-006 — _S3SphereCallaghanApproximation, fixed and validated
# =========================================================================

class TestS3SphereCallaghan:
    """SC-023, DP-006: the finite-time Callaghan sphere, fixed and validated.

    The model previously had three bugs (spherical_jn called without its order
    argument; cylinder Bessel roots instead of sphere Neumann roots; wrong
    normalization). It is now the correct finite-time SGP series

        E(q, tau) = |F(x)|^2 + 6 sum_{n,k: alpha>0} (2n+1)
                    alpha^2/(alpha^2 - n(n+1)) [x j_n'(x)/(x^2 - alpha^2)]^2
                    exp(-alpha^2 D tau / R^2),

    with x = 2*pi*q*R and F(x) = 3(sin x - x cos x)/x^3 the uniform-sphere
    structure factor. We validate the two analytical limits it must satisfy:
      (1) tau -> infinity  =>  the SGP long-time structure factor S2, exactly;
      (2) tau -> 0         =>  E = 1 (completeness);
    and agreement with the independent S4 Gaussian-Phase approximation in the
    regime where both are valid (small delta, moderate Delta, small radius).

    The finite-time excited-mode decay (not just the two limits) was validated
    against a dmipy-sim reflecting-sphere PGSE Monte-Carlo in the *strongly
    attenuated* regime (E ~ 0.47): as delta -> 0 the Monte-Carlo signal
    converges onto this series to within Monte-Carlo noise (E_MC - E_S3 =
    -0.0008 at delta = 0.1 ms, R = 5 um, q = 6e4, MC noise ~0.0018). The
    strong-attenuation regression values below are pinned to that validated
    curve so the series cannot silently regress.
    """

    D = 1.7e-9

    def _s3(self):
        from dmipy_fit.signal_models.sphere_models import (
            _S3SphereCallaghanApproximation)
        return _S3SphereCallaghanApproximation(diffusion_constant=self.D)

    def test_s3_long_time_limit_is_s2_structure_factor(self):
        """tau -> inf: S3 reduces to the SGP sphere structure factor (S2), exactly."""
        from dmipy_fit.signal_models.sphere_models import (
            S2SphereStejskalTannerApproximation)
        s3 = self._s3()
        s2 = S2SphereStejskalTannerApproximation()
        for diameter in (4e-6, 8e-6, 12e-6):
            q = np.linspace(1e4, 4e5, 12)
            tau = np.full_like(q, 3.0)  # 3 s >> R^2/D: fully restricted
            e3 = s3.sphere_attenuation(q, tau, diameter)
            e2 = s2.sphere_attenuation(q, diameter)
            assert np.max(np.abs(e3 - e2)) < 1e-4

    def test_s3_short_time_limit_is_unity(self):
        """tau -> 0: no attenuation, E -> 1 (eigenmode completeness)."""
        s3 = self._s3()
        q = np.array([5e4, 1e5, 2e5, 3e5])
        tau = np.full_like(q, 1e-8)  # << R^2/D: high eigenmodes undamped
        e3 = s3.sphere_attenuation(q, tau, 8e-6)
        assert np.all(np.abs(e3 - 1.0) < 1e-3)

    def test_s3_matches_s4_gpa_in_overlap_regime(self):
        """S3 (SGP) and S4 (GPA) agree where both approximations are valid."""
        from dmipy_fit.signal_models.sphere_models import (
            S4SphereGaussianPhaseApproximation)
        gamma = 2.6751525e8
        s3 = self._s3()
        for radius, delta, Delta in ((2e-6, 2e-3, 30e-3),
                                     (3e-6, 2e-3, 40e-3)):
            diameter = 2 * radius
            G = np.linspace(0.02, 0.28, 6)
            q = gamma * G * delta / (2 * np.pi)
            tau = np.full_like(q, Delta - delta / 3.0)
            s4 = S4SphereGaussianPhaseApproximation(
                diffusion_constant=self.D, diameter=diameter)
            e4 = np.array([float(np.asarray(
                s4.sphere_attenuation(g, delta, Delta, diameter)).ravel()[0])
                for g in G])
            e3 = s3.sphere_attenuation(q, tau, diameter)
            assert np.max(np.abs(e3 - e4)) < 0.03

    def test_s3_strong_attenuation_regression(self):
        """Pin the MC-validated strong-attenuation curve (E from 0.72 to 0.09)."""
        s3 = self._s3()
        q = np.array([4e4, 6e4, 8e4, 1.0e5])
        tau = np.full_like(q, 40e-3)
        E = s3.sphere_attenuation(q, tau, 1e-5)  # diameter 10 um
        expected = np.array([0.723754, 0.471877, 0.245339, 0.092397])
        assert np.allclose(E, expected, atol=1e-5)

    def test_s3_converges_onto_reflecting_sphere_mc(self):
        """Excited-mode decay matches a reflecting-sphere PGSE MC as delta->0.

        Offline fixture (tools/precompute_s3_sphere_mc.py): seed-averaged dmipy-sim
        reflecting-sphere PGSE signal at R=5um over a narrow-pulse sweep, in the
        strongly-attenuated regime (E ~ 0.72 -> 0.09). The S3 series is SGP
        (delta->0); the MC carries a finite-pulse bias that must vanish linearly
        as delta shrinks and converge monotonically onto the analytic curve.
        """
        from dmipy_fit.signal_models.sphere_models import (
            _S3SphereCallaghanApproximation)
        fx = np.load(os.path.join(os.path.dirname(__file__), "fixtures",
                                  "s3_sphere_callaghan_mc.npz"))
        q, tau, diameter = fx["q"], float(fx["tau"]), 2 * float(fx["radius"])
        deltas, E_mc = fx["deltas"], fx["E_mc"]

        s3 = _S3SphereCallaghanApproximation(diffusion_constant=float(fx["D"]))
        E_s3 = s3.sphere_attenuation(q, np.full_like(q, tau), diameter)

        # (1) the analytic curve the fixture was measured against is reproduced
        assert np.allclose(E_s3, fx["E_s3"], atol=1e-6)

        # (2) monotone delta->0 convergence onto the series (deltas descending)
        resid = np.max(np.abs(E_mc - E_s3), axis=1)
        assert np.all(np.diff(resid) < 0), (
            f"MC residual not shrinking as delta->0: {resid}")

        # (3) at the smallest delta the MC agrees with S3 to <1e-3 (near MC noise)
        assert resid[-1] < 1e-3, (
            f"S3 disagrees with reflecting-sphere MC at delta={deltas[-1]:.2e}s: "
            f"max|E_MC-E_S3|={resid[-1]:.2e}")

    def test_s3_call_returns_finite_attenuation(self):
        """The full __call__ path runs and returns physical (0, 1] attenuation."""
        from dmipy_fit.signal_models.sphere_models import (
            _S3SphereCallaghanApproximation)
        bvals = np.array([0., 1e9, 2e9])
        bvecs = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        scheme = acquisition_scheme_from_bvalues(bvals, bvecs, 0.1e-3, 40e-3)
        s3 = _S3SphereCallaghanApproximation(diameter=10e-6)
        E = s3(scheme, diameter=10e-6)
        assert np.all(np.isfinite(E))
        assert np.all(E > 0) and np.all(E <= 1.0 + 1e-9)
        assert E[0] == pytest.approx(1.0, abs=1e-9)  # b=0 unattenuated


# =========================================================================
# SC-025 — Watson-distributed C3Callaghan vs numerical orientation average
# =========================================================================

class TestWatsonC3OrientationAverage:
    """SC-025: SD1WatsonDistributed(C3Callaghan) agrees with brute-force
    orientation average.

    Physical setup:
    - Watson distribution with moderate alignment: odi=0.15 (kappa≈2)
    - Cylinder diameter = 4 µm, D = 2e-9 m²/s, SGP regime
    - Gradient along x, b = [0, 5e8, 1e9, 2e9] s/m², δ=0.5ms, Δ=40ms

    Procedure: evaluate C3Callaghan at 724 orientations uniformly distributed
    over the sphere, weight by the Watson PDF, sum. Compare against the
    SD1WatsonDistributed SH-convolution result.

    Note on kappa choice: odi=0.15 (kappa≈2) is chosen to test the regime
    where both the SH convolution and numerical orientation average are well-
    conditioned. High-kappa signal bounds are tested separately in
    TestWatsonHighKappaSignalBound (SC-028), including the previously-failing
    odi=0.007 regime after the renormalization fix.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
        from dmipy_fit.distributions import distributions
        from dipy.data import get_sphere

        self.D = 2e-9
        self.diameter = 4e-6
        # odi=0.15 (kappa≈2): safe regime where SH normalization is accurate
        self.odi = 0.15
        self.mu_axis = [0., 0.]  # Watson peak along z

        self.B_VALUES = np.array([0., 5e8, 1e9, 2e9])
        self.BVECS = np.tile([1., 0., 0.], (4, 1))
        self.scheme = acquisition_scheme_from_bvalues(
            self.B_VALUES, self.BVECS, 0.5e-3, 40e-3)

        # Analytical: SD1WatsonDistributed(C3Callaghan)
        cyl = C3CylinderCallaghanApproximation(diffusion_perpendicular=self.D)
        watson_dist = SD1WatsonDistributed([cyl])
        self.E_analytical = watson_dist(
            self.scheme,
            **{
                "C3CylinderCallaghanApproximation_1_lambda_par": self.D,
                "C3CylinderCallaghanApproximation_1_diameter": self.diameter,
                "SD1Watson_1_mu": self.mu_axis,
                "SD1Watson_1_odi": self.odi,
            },
        )

        # Numerical: weighted orientation average over sphere vertices
        sphere = get_sphere(name="symmetric724")
        watson = distributions.SD1Watson()
        w_pdf = watson(sphere.vertices, mu=self.mu_axis, odi=self.odi)
        w_norm = w_pdf / w_pdf.sum()

        cyl2 = C3CylinderCallaghanApproximation(diffusion_perpendicular=self.D)
        E_per_dir = np.zeros((len(sphere.vertices), len(self.B_VALUES)))
        for i, n_vec in enumerate(sphere.vertices):
            theta = np.arccos(np.clip(n_vec[2], -1.0, 1.0))
            phi = np.arctan2(n_vec[1], n_vec[0])
            E_per_dir[i] = cyl2(
                self.scheme,
                mu=[theta, phi],
                lambda_par=self.D,
                diameter=self.diameter,
            )
        self.E_numerical = (w_norm[:, None] * E_per_dir).sum(axis=0)

    def test_analytical_matches_numerical_orientation_average(self):
        """SC-025: SH-convolution signal agrees with orientation average ≤ 0.5%."""
        max_diff = np.abs(self.E_analytical - self.E_numerical).max()
        assert max_diff < 0.005, (
            f"SD1WatsonDistributed(C3Callaghan): max |analytical - numerical| = "
            f"{max_diff:.4f} > 0.005 (0.5%). "
            f"analytical={self.E_analytical}, numerical={self.E_numerical}")

    def test_signal_in_unit_interval(self):
        """SC-025: Watson-C3 signal must be in [0, 1] at every b-value."""
        assert np.all(self.E_analytical >= 0), (
            f"Watson-C3 signal has negative values: {self.E_analytical}")
        assert np.all(self.E_analytical <= 1.0 + 1e-10), (
            f"Watson-C3 signal exceeds 1.0: {self.E_analytical}")


# =========================================================================
# SC-028 — Watson E>1 bug at high kappa (new: documents known failure mode)
# =========================================================================

class TestWatsonHighKappaSignalBound:
    """SC-028: SD1WatsonDistributed signal must be in [0,1] for all kappa.

    Regression test for the normalization bug where high kappa (kappa≥50,
    odi<0.01) caused the 362-point hemisphere grid to under-sample the sharp
    Watson peak, making the SH L=0 coefficient exceed 1/(2*sqrt(pi)) and
    thus producing E > 1, which is physically impossible.

    Fix applied in SD1Watson.spherical_harmonics_representation(): after
    computing watson_sh via pseudoinverse, renormalize by enforcing
    watson_sh[0] == 1/(2*sqrt(pi)) (the exact analytical value for any
    normalized PDF in the Tournier real-SH convention).

    The test covers:
    - Low kappa (odi=0.5, kappa≈0.8)
    - Moderate kappa (odi=0.15, kappa≈2)
    - High kappa (odi=0.05, kappa≈6)
    - Very high kappa (odi=0.007, kappa≈45) — previously failed before fix
    """

    @staticmethod
    def _watson_c3_signal(odi):
        from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
        cyl = C3CylinderCallaghanApproximation(diffusion_perpendicular=2e-9)
        watson = SD1WatsonDistributed([cyl])
        bvals = np.array([0., 1e9])
        bvecs = np.tile([1., 0., 0.], (2, 1))
        scheme = acquisition_scheme_from_bvalues(bvals, bvecs, 0.5e-3, 40e-3)
        return watson(
            scheme,
            **{
                "C3CylinderCallaghanApproximation_1_lambda_par": 2e-9,
                "C3CylinderCallaghanApproximation_1_diameter": 4e-6,
                "SD1Watson_1_mu": [0., 0.],
                "SD1Watson_1_odi": odi,
            },
        )

    @pytest.mark.parametrize("odi", [0.5, 0.15, 0.05, 0.007])
    def test_signal_bounded(self, odi):
        """Watson-C3 signal is in [0, 1] for all odi values including very high kappa.

        odi=0.007 corresponds to kappa≈45, the regime that previously failed
        before the renormalization fix was applied (SC-028).
        """
        E = self._watson_c3_signal(odi)
        assert np.all(E >= 0.0), f"odi={odi}: signal has negative values {E}"
        assert np.all(E <= 1.0 + 1e-10), (
            f"odi={odi}: signal exceeds 1.0: {E}")


# =========================================================================
# H8/H9 — Spherical mean numerical stability at high b
# =========================================================================

class TestSphericalMeanHighBvalue:
    """Spherical mean stays finite and in (0,1] at very high b-values.

    This guards against future numerical regressions (NaN, Inf, or overflow)
    in the erf-based spherical mean formulas.  These tests run analytically
    on CPU in <1 s.
    """

    @pytest.mark.parametrize("b", [5e9, 1e10, 5e10])
    def test_c1stick_spherical_mean_finite(self, b):
        """C1Stick spherical mean stays finite at ultra-high b."""
        bvals = np.array([b])
        bvecs = np.array([[1., 0., 0.]])
        scheme = acquisition_scheme_from_bvalues(bvals, bvecs, 0.01, 0.04)
        E_mean = C1Stick().spherical_mean(scheme, lambda_par=1.7e-9)
        assert not np.any(np.isnan(E_mean)), f"C1Stick SM NaN at b={b}"
        assert not np.any(np.isinf(E_mean)), f"C1Stick SM Inf at b={b}"
        assert np.all(E_mean >= 0.0) and np.all(E_mean <= 1.0 + 1e-10), (
            f"C1Stick SM out of [0,1] at b={b}: {E_mean}")

    @pytest.mark.parametrize("b", [5e9, 1e10, 5e10])
    def test_g2zeppelin_spherical_mean_finite(self, b):
        """G2Zeppelin spherical mean stays finite at ultra-high b."""
        bvals = np.array([b])
        bvecs = np.array([[1., 0., 0.]])
        scheme = acquisition_scheme_from_bvalues(bvals, bvecs, 0.01, 0.04)
        E_mean = G2Zeppelin().spherical_mean(
            scheme, mu=[0., 0.], lambda_par=1.7e-9, lambda_perp=0.5e-9)
        assert not np.any(np.isnan(E_mean)), f"G2Zeppelin SM NaN at b={b}"
        assert not np.any(np.isinf(E_mean)), f"G2Zeppelin SM Inf at b={b}"
        assert np.all(E_mean >= 0.0) and np.all(E_mean <= 1.0 + 1e-10), (
            f"G2Zeppelin SM out of [0,1] at b={b}: {E_mean}")


# =========================================================================
# MC-backed regression test: DD1Gamma vs packed-cylinders benchmark
# =========================================================================

_DD1_REF_PATH = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
    "packed_cylinders_mc_reference.npy",
)
_DD1_META_PATH = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
    "packed_cylinders_mc_metadata.yaml",
)


class TestDD1GammaVsPackedCylindersMC:
    """CPU regression test for DD1GammaDistributed vs packed-cylinders MC.

    NOTE (SC-029): DD1GammaDistributed is currently broken when used via
    MultiCompartmentModel.simulate_signal / fit. The beta parameter is passed
    as a normalized value (e.g. 1.0) but DD1Gamma.__call__ uses it directly
    as a physical scale (metres), so the interpolators that determine the
    integration range are evaluated far outside their calibration range
    [1e-9, 2e-6 m], returning zero radii and a garbage signal.

    The existing test_spherical_mean_models.py tests call DD1GammaDistributed
    directly with physical parameter values and pass; this deeper path breaks.

    This test is marked xfail until the scaling bug is fixed.
    Tracking: SC-029 in scientific ledger.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "SC-029: DD1GammaDistributed beta parameter scaling bug — "
            "beta is passed as a normalized value to DD1Gamma.__call__ "
            "which expects a physical value (metres), causing integration "
            "over zero-radius cylinders and a non-physical signal."
        ),
    )
    def test_dd1gamma_agrees_with_mc_reference(self):
        """DD1GammaDistributed(C3Callaghan) agrees with packed-cylinders MC to ±2%."""
        from dmipy_fit.distributions.distribute_models import DD1GammaDistributed
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel

        E_mc_ref = np.load(_DD1_REF_PATH)

        # Parameters must match the GPU benchmark
        D_SIM = 2e-9
        ALPHA = 4.0
        BETA_R = 0.5e-6
        DELTA_S, DELTA_L = 0.5e-3, 40e-3
        B_VALUES = np.array([0., 5e8, 1e9, 2e9])
        BVECS = np.tile([1., 0., 0.], (4, 1))
        scheme = acquisition_scheme_from_bvalues(B_VALUES, BVECS, DELTA_S, DELTA_L)

        beta_d_norm = 2.0 * BETA_R / 1e-6

        cyl = C3CylinderCallaghanApproximation(diffusion_perpendicular=D_SIM)
        dd1 = DD1GammaDistributed(models=[cyl])
        mc_model = MultiCompartmentModel(models=[dd1])
        params = {
            "DD1GammaDistributed_1_C3CylinderCallaghanApproximation_1_mu":
                np.array([0., 0.]),
            "DD1GammaDistributed_1_C3CylinderCallaghanApproximation_1_lambda_par":
                D_SIM,
            "DD1GammaDistributed_1_DD1Gamma_1_alpha": ALPHA,
            "DD1GammaDistributed_1_DD1Gamma_1_beta": beta_d_norm,
            "partial_volume_0": 1.0,
        }
        E_analytical = mc_model.simulate_signal(
            scheme, mc_model.parameters_to_parameter_vector(**params))

        npt.assert_allclose(
            E_analytical, E_mc_ref, atol=0.02,
            err_msg=(
                "DD1GammaDistributed(C3Callaghan) differs from MC reference "
                "by more than 2% (atol=0.02). "
                "This may indicate a regression in DD1Gamma or C3Callaghan."
            ),
        )
