"""Physics-level integration tests for Rician MLE in the spherical mean framework.

Design rationale
----------------
The Rice(ν_sm, σ_eff) likelihood is built on the exact variance-propagation
result σ_eff = σ · ‖w‖₂, where w is the L=0 row of pinv(Y)/(2√π).  For the
HCP scheme (90 directions per shell), ‖w‖₂ ≈ 1/√90 ≈ 0.105, giving
σ_eff ≈ σ/9.5.  Every SM shell is therefore in the high-SNR (Gaussian) regime
of Rice(ν_sm, σ_eff) — Rice ≈ N(ν_sm, σ_eff²) — so the primary benefit over
plain MSE is per-shell variance weighting (1/σ_eff²), not noise-floor correction.
See docs/noise_rician_sm.md for the full derivation.

The tests below use G1Ball (lambda_iso only) with the HCP scheme (4 SM shells)
at σ=0.08 (per-direction SNR ≈ 1.5 at b=3000 s/mm², near the Rician noise
floor).  A multi-TE smoke test confirms the S0-normalisation path for schemes
with multiple echo times.

Tests
-----
1. Single voxel: lambda_iso within 15% of truth; fitted_sigma positive and in range.
2. 30-voxel stability: no NaN, all sigmas within declared bounds.
3. Median lambda_iso < 15% bias over 30 voxels.
4. MSE and Rician give consistent median lambda_iso (within 5% at σ=0.05).
5. Multi-TE smoke test: 3-TE HCP scheme completes without crash or NaN.
"""

import pytest
import numpy as np

jax = pytest.importorskip('jax')

from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.core.spherical_mean_framework import MultiCompartmentSphericalMeanModel
from dmipy_fit.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues


# ---------------------------------------------------------------------------
# Ground-truth parameters
# ---------------------------------------------------------------------------

GT_LAMBDA = 0.7e-9   # m²/s; gives E_sm(b=3000) ≈ 0.12, near σ floor
GT_SIGMA  = 0.08     # 1/SNR₀; per-direction SNR ≈ 1.5 at b=3000


def _make_noisy(E_clean, sigma, seed):
    """Generate Rice-distributed signal from noiseless E_clean."""
    rng = np.random.default_rng(seed)
    n1 = rng.normal(0, sigma, E_clean.shape)
    n2 = rng.normal(0, sigma, E_clean.shape)
    return np.sqrt((E_clean + n1) ** 2 + n2 ** 2)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def hcp():
    return wu_minn_hcp_acquisition_scheme()


@pytest.fixture(scope='module')
def mc_ball(hcp):
    return MultiCompartmentModel(models=[G1Ball()])


@pytest.fixture(scope='module')
def sm_ball():
    return MultiCompartmentSphericalMeanModel(models=[G1Ball()])


@pytest.fixture(scope='module')
def scheme_3te(hcp):
    """3-TE HCP scheme: 864 measurements, 12 SM shells.

    Note: the SM framework normalises each shell by its TE-specific b=0 signal,
    which absorbs T2 decay.  After normalisation the 12 SM shells reduce to 3
    copies of the same 4 normalised b-value signals, so per-TE S0 normalisation
    makes the multi-TE case equivalent to the single-TE case for diffusion
    parameter estimation.  The multi-TE test is therefore a smoke test (no
    crash / no NaN) rather than a quantitative recovery test.
    """
    TEs = [0.060, 0.080, 0.100]
    bv    = np.tile(hcp.bvalues,            3)
    gd    = np.tile(hcp.gradient_directions, (3, 1))
    delta = np.tile(hcp.delta,               3)
    Delta = np.tile(hcp.Delta,               3)
    TE    = np.concatenate([np.full(len(hcp.bvalues), t) for t in TEs])
    return acquisition_scheme_from_bvalues(bv, gd, delta=delta, Delta=Delta, TE=TE)


# ---------------------------------------------------------------------------
# Test 1: single-voxel recovery at near-noise-floor SNR
# ---------------------------------------------------------------------------

class TestSingleVoxelRecovery:
    """Single-voxel Rician SM fit at near-noise-floor SNR (σ=0.08)."""

    def test_lambda_iso_within_15_percent(self, hcp, mc_ball, sm_ball):
        E = mc_ball.simulate_signal(
            hcp, mc_ball.parameters_to_parameter_vector(
                G1Ball_1_lambda_iso=GT_LAMBDA))
        noisy = _make_noisy(E, GT_SIGMA, seed=1)
        result = sm_ball.fit(hcp, noisy[None], solver='jax',
                             sigma_x0=GT_SIGMA, sigma_range=(0.01, 0.20))
        lam = float(result.fitted_parameters['G1Ball_1_lambda_iso'].squeeze())
        assert abs(lam - GT_LAMBDA) / GT_LAMBDA < 0.15, (
            f"lambda_iso: fitted={lam:.3e}, true={GT_LAMBDA:.3e}")

    def test_fitted_sigma_positive_in_range(self, hcp, mc_ball, sm_ball):
        E = mc_ball.simulate_signal(
            hcp, mc_ball.parameters_to_parameter_vector(
                G1Ball_1_lambda_iso=GT_LAMBDA))
        noisy = _make_noisy(E, GT_SIGMA, seed=2)
        result = sm_ball.fit(hcp, noisy[None], solver='jax',
                             sigma_x0=GT_SIGMA, sigma_range=(0.01, 0.20))
        s = float(result.fitted_sigma.squeeze())
        assert not np.isnan(s), "fitted_sigma is NaN"
        assert 0.01 <= s <= 0.20, f"sigma out of declared range: {s:.4f}"


# ---------------------------------------------------------------------------
# Test 2 & 3: multi-voxel stability
# ---------------------------------------------------------------------------

class TestMultiVoxelStability:
    """30-voxel robustness: no NaN, all sigmas in declared range, bias < 15%."""

    @pytest.mark.slow
    def test_no_nan_values(self, hcp, mc_ball, sm_ball):
        E = mc_ball.simulate_signal(
            hcp, mc_ball.parameters_to_parameter_vector(
                G1Ball_1_lambda_iso=GT_LAMBDA))
        noisy = _make_noisy(np.tile(E, (30, 1)), GT_SIGMA, seed=10)
        result = sm_ball.fit(hcp, noisy, solver='jax',
                             sigma_x0=GT_SIGMA, sigma_range=(0.01, 0.20))
        lam = result.fitted_parameters['G1Ball_1_lambda_iso'].flatten()
        sig = result.fitted_sigma.flatten()
        assert not np.any(np.isnan(lam)), "NaN in lambda_iso"
        assert not np.any(np.isnan(sig)), "NaN in fitted_sigma"
        assert np.all(sig >= 0.01) and np.all(sig <= 0.20), (
            f"sigma outside [0.01, 0.20]: [{sig.min():.3f}, {sig.max():.3f}]")

    @pytest.mark.slow
    def test_median_lambda_within_15_percent(self, hcp, mc_ball, sm_ball):
        """Median lambda_iso < 15% bias over 30 independent noisy realisations."""
        E = mc_ball.simulate_signal(
            hcp, mc_ball.parameters_to_parameter_vector(
                G1Ball_1_lambda_iso=GT_LAMBDA))
        noisy = _make_noisy(np.tile(E, (30, 1)), GT_SIGMA, seed=11)
        result = sm_ball.fit(hcp, noisy, solver='jax',
                             sigma_x0=GT_SIGMA, sigma_range=(0.01, 0.20))
        lam_med = float(np.median(
            result.fitted_parameters['G1Ball_1_lambda_iso']))
        assert abs(lam_med - GT_LAMBDA) / GT_LAMBDA < 0.15, (
            f"Median lambda_iso={lam_med:.3e}, true={GT_LAMBDA:.3e}")


# ---------------------------------------------------------------------------
# Test 4: MSE vs Rician consistency
# ---------------------------------------------------------------------------

class TestMseVsRicianConsistency:
    """At moderate SNR, MSE and Rician SM give similar median lambda estimates.

    Rice(ν_sm, σ_eff) reduces to Gaussian when ν_sm >> σ_eff.  For HCP
    (90 dirs/shell), σ_eff ≈ σ/9.5 ≈ 0.005 at σ=0.05, while the lowest
    SM signal (b=3000) is ≈ 0.12 — a factor of 24 above σ_eff.  Both losses
    are therefore in the Gaussian regime, and their median estimates must agree
    within 5% of the true lambda.
    """

    @pytest.mark.slow
    def test_mse_and_rician_medians_agree(self, hcp, mc_ball, sm_ball):
        sigma_mod = 0.05  # moderate noise: well above σ_eff Rician floor
        E = mc_ball.simulate_signal(
            hcp, mc_ball.parameters_to_parameter_vector(
                G1Ball_1_lambda_iso=GT_LAMBDA))
        noisy = _make_noisy(np.tile(E, (30, 1)), sigma_mod, seed=20)

        result_mse = sm_ball.fit(hcp, noisy, solver='jax')
        result_ric = sm_ball.fit(hcp, noisy, solver='jax',
                                 sigma_x0=sigma_mod, sigma_range=(0.01, 0.15))

        lam_mse = float(np.median(
            result_mse.fitted_parameters['G1Ball_1_lambda_iso']))
        lam_ric = float(np.median(
            result_ric.fitted_parameters['G1Ball_1_lambda_iso']))
        assert abs(lam_mse - lam_ric) / GT_LAMBDA < 0.05, (
            f"MSE={lam_mse:.3e} and Rician={lam_ric:.3e} disagree by "
            f">{abs(lam_mse - lam_ric)/GT_LAMBDA:.1%} of true {GT_LAMBDA:.3e}")


# ---------------------------------------------------------------------------
# Test 5: multi-TE smoke test
# ---------------------------------------------------------------------------

class TestMultiTeSmoke:
    """3-TE HCP scheme: no crash, no NaN in fitted_sigma.

    After per-TE S0 normalisation, T2 is absorbed and the 12 SM shells reduce
    to 3 copies of the same 4 normalised diffusion signals.  Quantitative
    sigma and lambda recovery are therefore not asserted here.
    """

    @pytest.mark.slow
    def test_3te_scheme_no_crash(self, scheme_3te):
        mc = MultiCompartmentModel(models=[G1Ball()])
        sm = MultiCompartmentSphericalMeanModel(models=[G1Ball()])
        E = mc.simulate_signal(
            scheme_3te,
            mc.parameters_to_parameter_vector(
                G1Ball_1_lambda_iso=GT_LAMBDA, G1Ball_1_T2=0.08))
        noisy = _make_noisy(E, GT_SIGMA, seed=30)
        result = sm.fit(scheme_3te, noisy[None], solver='jax',
                        sigma_x0=GT_SIGMA, sigma_range=(0.01, 0.20))
        assert result.fitted_sigma is not None, "fitted_sigma is None for 3-TE scheme"
        s = float(result.fitted_sigma.squeeze())
        assert not np.isnan(s), "fitted_sigma is NaN for 3-TE scheme"
        assert s > 0, "fitted_sigma must be positive"
        lam = float(result.fitted_parameters['G1Ball_1_lambda_iso'].squeeze())
        assert not np.isnan(lam), "lambda_iso is NaN for 3-TE scheme"
