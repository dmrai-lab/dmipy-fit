"""Γ_lm fast-eigenmode path tests using pre-computed MC signal fixtures.

Tests compare ``C4CylinderGaussianPhaseApproximation`` analytical output against
MC reference signals stored in::

    benchmarks/cylinder_fixtures/fixtures_R2.0um_gamma_lm_validation.npz

That file is generated once by::

    cd benchmarks/cylinder_fixtures && python generate_gamma_lm_fixtures.py

It contains pre-computed signals (17 kB) derived from the R=2µm trajectory
substrate.  Tests never load the raw trajectory file (3.47 GB).

The Γ_lm path is triggered when the acquisition scheme stores the full waveform
tensor G(t) (``AcquisitionScheme.from_waveform``).  It uses a fast-eigenmode
expansion valid when α₁ × σ >> 1, where α₁ = µ₁²D/R² and σ is the effective
gradient duration.

For R = 2 µm, D = 1.7×10⁻⁹ m²/s, σ = 40 ms:
    α₁ = (1.8412)² × 1.7×10⁻⁹ / (2×10⁻⁶)² = 1441 s⁻¹
    τ₁ = 1/α₁ = 0.69 ms
    α₁ × σ = 57.6  >>  1  ✓

All tests skip automatically when the fixture file is absent.
"""

from pathlib import Path
import warnings

import numpy as np
import numpy.testing as npt
import pytest
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
import os
# Dev-only MC fixtures (not shipped). Set DMIPY_FIXTURE_DIR to run; else the tests skip.
FIXTURE_DIR     = Path(os.environ.get("DMIPY_FIXTURE_DIR", "cylinder_fixtures_unavailable"))
FIXTURE_NPZ     = FIXTURE_DIR / "fixtures_R2.0um_gamma_lm_validation.npz"

D    = 1.7e-9         # m²/s
R_M  = 2.0e-6         # m
DIAM = 2.0 * R_M

# Fast-eigenmode validity: α₁σ >> 1 requires σ >> τ₁ ≈ 0.69 ms.
SIGMA_VALID   = 40e-3   # 40 ms  →  α₁σ ≈ 57.6  (well into fast-eigenmode regime)
SIGMA_INVALID = 0.5e-3  # 0.5 ms →  α₁σ ≈ 0.72  (borderline — should fail)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_fixture():
    if not FIXTURE_NPZ.exists():
        pytest.skip(f"Γ_lm fixture not found: {FIXTURE_NPZ}\n"
                    "Run: cd benchmarks/cylinder_fixtures && "
                    "python generate_gamma_lm_fixtures.py")


def _load_fixture():
    _require_fixture()
    return np.load(FIXTURE_NPZ)


def _build_rotating_scheme(G_amplitudes, sigma=SIGMA_VALID, n_t=400):
    """x→y rotating gradient: G(t) = G × (cos θ(t) x̂ + sin θ(t) ŷ), θ: 0→π/2.

    Returns (scheme, G_arr float32 (n_meas, n_t, 3), dt float).
    """
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme

    dt    = sigma / (n_t - 1)
    t     = np.linspace(0.0, sigma, n_t)
    theta = (np.pi / 2.0) * t / sigma   # 0 → π/2

    G_list = []
    for G_amp in G_amplitudes:
        Gx = G_amp * np.cos(theta)
        Gy = G_amp * np.sin(theta)
        G_list.append(np.stack([Gx, Gy, np.zeros(n_t)], axis=-1))

    G_arr     = np.stack(G_list, axis=0).astype(np.float32)
    grad_dirs = np.tile(
        np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]]),
        (len(G_amplitudes), 1)
    )
    scheme = AcquisitionScheme.from_waveform(G_arr, dt, grad_dirs, allow_unrefocused=True)
    return scheme, G_arr, float(dt)


def _c4():
    from dmipy_fit.signal_models.cylinder_models import C4CylinderGaussianPhaseApproximation
    return C4CylinderGaussianPhaseApproximation(diffusion_perpendicular=D, diameter=DIAM)


# ---------------------------------------------------------------------------
# 1. PGSE path consistency: signal_from_gamma_lm == perpendicular_attenuation
#    for a collinear waveform — pure analytical comparison, no fixture needed.
# ---------------------------------------------------------------------------

def test_gamma_lm_matches_standard_pgse_path():
    """Γ_lm path and standard C4.perpendicular_attenuation agree for PGSE.

    For a collinear PGSE waveform the Γ_lm is a fast-eigenmode approximation
    of the exact Van Gelderen GPA.  The two paths produce consistent signals
    within the known Γ_lm approximation error (~1–2% absolute at moderate b).
    Tolerance: 0.02 (consistent with rotating-waveform MC tests).
    """
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    from dmipy_fit.signal_models.cylinder_models import C4CylinderGaussianPhaseApproximation

    delta = 10e-3    # 10 ms
    DELTA = 30e-3    # 30 ms
    G_amps = [0.1, 0.2, 0.3, 0.5]
    n_t   = 401

    T_total = DELTA + delta
    dt      = T_total / (n_t - 1)
    n_pulse = max(1, round(delta / dt))
    n_DELTA = round(DELTA / dt)

    G_list = []
    for G_amp in G_amps:
        G_1d = np.zeros((n_t, 3), dtype=np.float32)
        G_1d[:n_pulse, 0]                  =  G_amp
        G_1d[n_DELTA:n_DELTA + n_pulse, 0] = -G_amp
        G_list.append(G_1d)

    G_arr     = np.stack(G_list, axis=0)
    grad_dirs = np.tile(np.array([[1., 0., 0.]]), (len(G_amps), 1))
    scheme    = AcquisitionScheme.from_waveform(G_arr, dt, grad_dirs, allow_unrefocused=True)

    c4 = C4CylinderGaussianPhaseApproximation(diffusion_perpendicular=D, diameter=DIAM)

    E_standard = np.array([
        c4.perpendicular_attenuation(G, delta, DELTA, DIAM)
        for G in G_amps
    ])
    # Explicit Γ_lm call — collinear waveform doesn't auto-dispatch in __call__
    mu_sph     = np.array([0.0, 0.0])   # z-axis in spherical coords
    E_gamma_lm = c4.signal_from_gamma_lm(scheme, mu=mu_sph, lambda_par=D)

    npt.assert_allclose(
        E_gamma_lm, E_standard, atol=0.02,
        err_msg=(
            "Γ_lm path and standard PGSE path disagree for collinear waveform.\n"
            f"G_amps: {G_amps}\n"
            f"Standard:  {E_standard}\n"
            f"Γ_lm path: {E_gamma_lm}\n"
            f"Diff:      {np.abs(E_gamma_lm - E_standard)}"
        )
    )


# ---------------------------------------------------------------------------
# 2. Rotating waveform: Γ_lm path vs MC fixture (R = 2 µm, σ = 40 ms)
# ---------------------------------------------------------------------------

def test_gamma_lm_rotating_waveform_vs_mc_fixture():
    """Γ_lm path matches MC for x→y rotating waveform on R = 2 µm substrate.

    Fast-eigenmode condition: α₁ × σ = 57.6 >> 1 (well-satisfied).
    Tolerance: |ΔE| < 0.02 (MC noise floor σ_MC ≈ 0.0014 + GPD residual).
    """
    fix = _load_fixture()
    E_mc   = fix["rot_sigma40ms_signals"]    # (3,) float64
    G_amps = fix["rot_sigma40ms_G_amps"].tolist()
    G_arr  = fix["rot_sigma40ms_G_arr"]      # (3, n_t, 3) float32
    dt_wf  = float(fix["rot_sigma40ms_dt_wf"])

    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    grad_dirs = np.tile(
        np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]]),
        (len(G_amps), 1)
    )
    scheme = AcquisitionScheme.from_waveform(G_arr, dt_wf, grad_dirs, allow_unrefocused=True)

    c4     = _c4()
    E_anal = c4(scheme, mu=np.array([0., 0.]), lambda_par=D)

    diffs    = np.abs(E_anal - E_mc)
    failures = [
        f"G={G_amps[i]:.1f} T/m: Γ_lm={E_anal[i]:.4f}  MC={E_mc[i]:.4f}  |ΔE|={diffs[i]:.4f}"
        for i in range(len(G_amps)) if diffs[i] > 0.02
    ]
    assert not failures, (
        "Γ_lm vs MC fixture for R=2µm rotating waveform:\n" +
        "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 3. Cylindrical symmetry: G in x̂ = G in ŷ = G in (x̂+ŷ)/√2
# ---------------------------------------------------------------------------

def test_cylindrical_symmetry_across_transverse_directions():
    """For a z-axis cylinder, PGSE signal is equal in any transverse direction.

    The cylinder has circular cross-section: diffusion in x and y are
    statistically identical.  MC signals from G in x̂, ŷ, and (x̂+ŷ)/√2 must
    agree within MC noise (tolerance: 0.003, well above σ_MC ≈ 0.0014).
    """
    fix  = _load_fixture()
    E_x, E_y, E_xy = fix["symmetry_signals"]

    tol = 0.003
    assert abs(E_x - E_y) < tol, (
        f"Symmetry broken: E(x̂)={E_x:.4f} vs E(ŷ)={E_y:.4f}, diff={abs(E_x-E_y):.4f}"
    )
    assert abs(E_x - E_xy) < tol, (
        f"Symmetry broken: E(x̂)={E_x:.4f} vs E((x̂+ŷ)/√2)={E_xy:.4f}, "
        f"diff={abs(E_x-E_xy):.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Fast-eigenmode validity boundary: σ << τ₁ should fail the approximation
# ---------------------------------------------------------------------------

def test_fast_eigenmode_valid_at_long_sigma():
    """At σ = 40 ms >> τ₁ = 0.69 ms (α₁σ ≈ 57.6), Γ_lm matches MC well.

    Uses the same fixture as test_gamma_lm_rotating_waveform_vs_mc_fixture
    (first G amplitude, G=0.1 T/m).  Tolerance: 0.02.
    """
    fix  = _load_fixture()
    E_mc = float(fix["rot_sigma40ms_signals"][0])
    G_arr = fix["rot_sigma40ms_G_arr"][[0]]     # first amplitude only
    dt_wf = float(fix["rot_sigma40ms_dt_wf"])

    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    grad_dirs = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]])
    scheme    = AcquisitionScheme.from_waveform(G_arr, dt_wf, grad_dirs, allow_unrefocused=True)

    c4     = _c4()
    E_anal = float(c4(scheme, mu=np.array([0., 0.]), lambda_par=D)[0])

    assert abs(E_anal - E_mc) < 0.02, (
        f"σ=40ms (α₁σ=57.6): |ΔE|={abs(E_anal-E_mc):.4f} > 0.02 "
        f"(Γ_lm={E_anal:.4f}, MC={E_mc:.4f})"
    )


def test_fast_eigenmode_fails_at_short_sigma():
    """At σ = 0.5 ms ≈ τ₁ (α₁σ ≈ 0.72), Γ_lm significantly overestimates E.

    When σ is comparable to τ₁, the fast-eigenmode approximation breaks down.
    We verify |ΔE| > 0.02 to document this known failure mode.
    """
    fix  = _load_fixture()
    E_mc  = float(fix["rot_sigma05ms_signals"][0])
    G_arr = fix["rot_sigma05ms_G_arr"]           # (1, n_t, 3) float32
    dt_wf = float(fix["rot_sigma05ms_dt_wf"])

    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    grad_dirs = np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]])
    scheme    = AcquisitionScheme.from_waveform(G_arr, dt_wf, grad_dirs, allow_unrefocused=True)

    c4     = _c4()
    E_anal = float(c4(scheme, mu=np.array([0., 0.]), lambda_par=D)[0])

    diff = abs(E_anal - E_mc)
    assert diff > 0.02, (
        f"Expected Γ_lm to fail at σ=0.5ms (α₁σ≈0.72), but |ΔE|={diff:.4f} < 0.02. "
        f"Γ_lm={E_anal:.4f}, MC={E_mc:.4f}. "
        "Either the approximation is unexpectedly good here, or the waveform "
        "produced insufficient phase (check G and σ)."
    )


# ---------------------------------------------------------------------------
# 5. Dispersed signal: isotropic ODF → spherical mean, pure analytical test
# ---------------------------------------------------------------------------

def test_dispersed_isotropic_odf_equals_spherical_mean_fixture():
    """For a nearly isotropic Watson ODF (κ = 0.01), dispersed signal ≈ average
    of single-cylinder signals over all orientations.

    Purely analytical: signal_lm() + dispersed_signal_from_E_lm() vs 724-point spherical quadrature.
    No fixture file needed.  Tolerance: 0.02.
    """
    from dmipy_fit.signal_models.cylinder_models import (
        C4CylinderGaussianPhaseApproximation,
        dispersed_signal_from_E_lm,
        watson_odf_lm,
        _SPHERE_QUAD_PTS,
        _SPHERE_QUAD_W,
        _compute_C_geom,
    )

    G_amps = [0.1, 0.3]
    scheme, _, dt_wf = _build_rotating_scheme(G_amps, sigma=SIGMA_VALID)

    c4     = C4CylinderGaussianPhaseApproximation(diffusion_perpendicular=D, diameter=DIAM)
    kappa  = 0.01
    mu_z   = np.array([0., 0., 1.])
    odf_lm = watson_odf_lm(mu_z, kappa, l_max=4)

    S_disp = dispersed_signal_from_E_lm(
        c4.signal_lm(scheme, diameter=DIAM, lambda_par=D), odf_lm
    )

    roots  = C4CylinderGaussianPhaseApproximation._CYLINDER_TRASCENDENTAL_ROOTS
    C_geom = _compute_C_geom(R_M, D, roots)
    gam_lm = scheme.gamma_lm(l_max=4)
    Gamma00 = gam_lm[:, 0]
    Gamma2m = gam_lm[:, 1:]
    c0  = C_geom * (2. / 3.) * Gamma00 * np.sqrt(4 * np.pi)
    c2m = -C_geom * (8 * np.pi / 15.) * Gamma2m

    pts = _SPHERE_QUAD_PTS
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    Y2m_q = np.stack([
        np.sqrt(15. / (4 * np.pi)) * x * y,
        np.sqrt(15. / (4 * np.pi)) * y * z,
        np.sqrt(5. / (16 * np.pi)) * (2 * z ** 2 - x ** 2 - y ** 2),
        np.sqrt(15. / (4 * np.pi)) * x * z,
        np.sqrt(15. / (16 * np.pi)) * (x ** 2 - y ** 2),
    ], axis=-1)

    # Parallel phase: lambda_par * n̂^T B n̂ using the exact btensor.
    # The rotating waveform has off-diagonal btensor entries, so the
    # PGSE-like formula (b * (n_grad · n̂)^2) does not apply here.
    B = scheme.btensor()   # (n_m, 3, 3)

    for i in range(len(G_amps)):
        phi_perp_q = c0[i] + (Y2m_q * c2m[i]).sum(-1)
        phi_par_q  = D * np.einsum('qi,ij,qj->q', pts, B[i], pts)
        phi_q      = phi_perp_q + phi_par_q
        sph_mean   = float(np.sum(_SPHERE_QUAD_W * np.exp(-phi_q)))

        assert abs(float(S_disp[i]) - sph_mean) < 0.02, (
            f"G={G_amps[i]} T/m: dispersed (κ=0.01)={float(S_disp[i]):.4f} "
            f"vs spherical mean={sph_mean:.4f}, "
            f"|ΔE|={abs(float(S_disp[i])-sph_mean):.4f}"
        )
