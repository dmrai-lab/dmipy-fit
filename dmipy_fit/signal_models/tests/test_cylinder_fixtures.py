"""Regression tests: MC trajectory fixture battery vs analytical cylinder models.

Pre-computed Monte Carlo signal fixtures from
``benchmarks/cylinder_fixtures/`` are loaded and validated against the
C4CylinderGaussianPhaseApproximation (Van Gelderen 1994 GPD formula).

Each test is < 5 s (CPU-only, no MC re-runs).  Tests skip automatically when
fixture files are absent so the suite can run in CI without the large files.

Physics context
---------------
GPD approximation validity: the Van Gelderen formula integrates the
eigenfunction expansion of the Green's function for a cylinder assuming the
phase distribution is Gaussian.  It is exact in the low-b regime and
progressively overestimates E at high b where quantum-path interference
(diffraction) appears in the true signal.

Tolerance tiers (empirically validated, see reports/):
  - b < 1000 s/mm²:             |ΔE| < 0.003  (GPD essentially exact)
  - b ∈ [1000, 50000] s/mm²,
    R ≤ 2 µm:                   |ΔE| < 0.015  (GPD moderate regime)
  - b ∈ [1000, 50000] s/mm²,
    R = 3 µm:                   |ΔE| < 0.030  (GPD starts breaking down)
  - b > 50000 s/mm² or R = 5 µm with b > 1000: no analytical tolerance
    (diffraction / oscillatory regime; MC is ground truth)
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------

import os
# Dev-only MC fixtures (not shipped). Point DMIPY_FIXTURE_DIR at them to run these tests;
# otherwise the default path does not exist and every fixture test skips.
FIXTURE_DIR = Path(os.environ.get("DMIPY_FIXTURE_DIR", "cylinder_fixtures_unavailable"))

#: Mapping from radius (µm) to the stem used in fixture filenames.
RADIUS_TO_STEM = {
    0.25: "R0.25um",
    0.50: "R0.50um",
    0.75: "R0.75um",
    1.0:  "R1.0um",
    2.0:  "R2.0um",
    3.0:  "R3.0um",
    5.0:  "R5.0um",
}

ALL_RADII = sorted(RADIUS_TO_STEM)


def _load(stem, group):
    """Load fixture signals and waveform metadata. Skip if not found."""
    npz  = FIXTURE_DIR / f"fixtures_{stem}_{group}.npz"
    yml  = FIXTURE_DIR / f"fixtures_{stem}_{group}.yaml"
    if not npz.exists():
        pytest.skip(f"Fixture not found: {npz}")
    data = np.load(npz)
    with open(yml) as f:
        meta = yaml.safe_load(f)
    return data["signals"], meta["waveform_params"]


def _c4(R_um):
    """Instantiate C4 GPD cylinder model with the standard diffusivity."""
    from dmipy_fit.signal_models.cylinder_models import (
        C4CylinderGaussianPhaseApproximation,
    )
    return C4CylinderGaussianPhaseApproximation(
        diffusion_perpendicular=1.7e-9,
        diameter=2.0 * R_um * 1e-6,
    )


# ---------------------------------------------------------------------------
# 1. GPD exact regime: b < 1000 s/mm² — all 7 radii
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um", ALL_RADII)
def test_c4_gpd_exact_regime(R_um):
    """C4 GPD matches MC within 0.003 for b < 1000 s/mm² (all radii).

    This regime covers every physiologically relevant acquisition on clinical
    scanners (G ≤ 80 mT/m, δ ≤ 20 ms).  GPD should be essentially exact here.
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, params = _load(stem, "pgse_finite_delta")
    c4      = _c4(R_um)
    diam    = 2.0 * R_um * 1e-6
    tol     = 0.003

    failures = []
    for i, p in enumerate(params):
        b_smm2 = p["b_sm2"] * 1e-6
        if b_smm2 >= 1000.0:
            continue
        E_anal = c4.perpendicular_attenuation(
            p["G_Tm"], p["delta_s"], p["DELTA_s"], diam
        )
        diff = abs(signals[i] - E_anal)
        if diff > tol:
            failures.append(
                f"b={b_smm2:.0f} s/mm²: MC={signals[i]:.4f} "
                f"C4={E_anal:.4f}  |ΔE|={diff:.4f} > {tol}"
            )

    assert not failures, (
        f"R={R_um} µm — C4 GPD exact-regime failures:\n" +
        "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 2. GPD moderate regime: b ∈ [1000, 50000] s/mm², R ≤ 3 µm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um,tol,b_max", [
    (0.25, 0.003, 50_000),
    (0.50, 0.003, 50_000),
    (0.75, 0.004, 50_000),
    (1.0,  0.005, 50_000),
    (2.0,  0.015, 50_000),
    # R=3 µm: diffraction structure starts around b~40 000 s/mm²;
    # restrict to b < 20 000 where GPD error stays below 0.03.
    (3.0,  0.030, 20_000),
])
def test_c4_gpd_moderate_regime(R_um, tol, b_max):
    """C4 GPD matches MC within tolerance for b ∈ [1000, b_max] s/mm², R ≤ 3 µm.

    GPD error grows with R and b.  For R = 3 µm the upper b limit is reduced
    to 20 000 s/mm² because diffraction oscillations (missing in GPD) appear
    above that threshold.  R = 5 µm is excluded entirely: diffraction begins
    at b ≈ 3 000 s/mm².
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, params = _load(stem, "pgse_finite_delta")
    c4      = _c4(R_um)
    diam    = 2.0 * R_um * 1e-6

    failures = []
    for i, p in enumerate(params):
        b_smm2 = p["b_sm2"] * 1e-6
        if not (1000.0 <= b_smm2 <= b_max):
            continue
        E_anal = c4.perpendicular_attenuation(
            p["G_Tm"], p["delta_s"], p["DELTA_s"], diam
        )
        diff = abs(signals[i] - E_anal)
        if diff > tol:
            failures.append(
                f"b={b_smm2:.0f}: MC={signals[i]:.4f} "
                f"C4={E_anal:.4f}  |ΔE|={diff:.4f} > {tol}"
            )

    assert not failures, (
        f"R={R_um} µm — C4 GPD moderate-regime failures:\n" +
        "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 3. Diffraction dip at R = 5 µm: MC captures non-monotonic structure GPD misses
# ---------------------------------------------------------------------------

def test_diffraction_dip_r5um():
    """At the first diffraction minimum (R=5 µm, δ=5 ms, Δ=25 ms, G=0.6 T/m),
    MC signal is near zero while C4 GPD overestimates by > 0.04.

    This confirms that the MC fixtures correctly encode diffraction interference
    that the Gaussian Phase Approximation cannot reproduce, and documents the
    boundary of GPD validity for R=5 µm.
    """
    signals, params = _load("R5.0um", "pgse_finite_delta")
    c4   = _c4(5.0)
    diam = 10.0e-6

    # Find the δ=5 ms, G=0.6 T/m measurement
    idx = next(
        i for i, p in enumerate(params)
        if abs(p["delta_s"] - 5e-3) < 1e-5 and abs(p["G_Tm"] - 0.6) < 0.01
    )
    p      = params[idx]
    E_mc   = signals[idx]
    E_anal = c4.perpendicular_attenuation(p["G_Tm"], p["delta_s"], p["DELTA_s"], diam)

    # MC is near the diffraction dip (signal collapsed to ~0)
    assert E_mc < 0.04, (
        f"Expected diffraction dip (E_mc < 0.04), got {E_mc:.4f}"
    )
    # GPD overestimates because it cannot model coherence effects
    assert E_anal > E_mc + 0.04, (
        f"Expected GPD >> MC at diffraction minimum, but "
        f"E_anal={E_anal:.4f}, E_mc={E_mc:.4f}, diff={E_anal - E_mc:.4f}"
    )


def test_diffraction_echo_r5um():
    """At R=5 µm, the first diffraction echo occurs just above the dip.

    At G=1.0 T/m (δ=5 ms, Δ=25 ms) the MC signal is measurably above the
    preceding dip (G=0.6 T/m), confirming the Van Gelderen oscillatory
    structure is correctly reproduced by the MC trajectories.
    """
    signals, params = _load("R5.0um", "pgse_finite_delta")

    def _get(G_target):
        return signals[next(
            i for i, p in enumerate(params)
            if abs(p["delta_s"] - 5e-3) < 1e-5 and abs(p["G_Tm"] - G_target) < 0.01
        )]

    E_dip  = _get(0.6)   # first diffraction minimum
    E_echo = _get(1.0)   # first diffraction echo (signal rebounds)

    assert E_echo > E_dip + 0.01, (
        f"Expected diffraction echo E(G=1.0)={E_echo:.4f} > E_dip(G=0.6)={E_dip:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Physical ordering: smaller R → more restricted signal (fixed waveform)
# ---------------------------------------------------------------------------

def test_smaller_radius_more_restricted():
    """For a fixed PGSE waveform (δ=5 ms, Δ=25 ms, G=0.3 T/m) restricted
    diffusion increases monotonically with R: E(R=0.25) > E(R=0.5) > … > E(R=5).

    A smaller cylinder confines walkers more tightly; more restriction means
    lower net displacement and therefore HIGHER signal at fixed b.

    Wait — this is the *opposite* direction: a SMALLER radius means walkers
    are MORE confined to the perpendicular plane.  Confined walkers have
    SMALLER mean squared displacement in the transverse direction, which means
    LESS phase dispersion and a HIGHER signal.  Equivalently, effective D_perp
    decreases as R decreases → E(small R) > E(large R).
    """
    signals_by_R = {}
    for R_um, stem in RADIUS_TO_STEM.items():
        signals, params = _load(stem, "pgse_finite_delta")
        idx = next(
            i for i, p in enumerate(params)
            if abs(p["delta_s"] - 5e-3) < 1e-5 and abs(p["G_Tm"] - 0.3) < 0.01
        )
        signals_by_R[R_um] = signals[idx]

    radii_sorted = sorted(signals_by_R)
    for r_small, r_large in zip(radii_sorted, radii_sorted[1:]):
        E_small = signals_by_R[r_small]
        E_large = signals_by_R[r_large]
        assert E_small > E_large - 0.002, (
            f"Expected E(R={r_small}µm)={E_small:.4f} > "
            f"E(R={r_large}µm)={E_large:.4f} but ordering violated"
        )


# ---------------------------------------------------------------------------
# 5. Physical ordering: increasing G/b → lower E within non-diffraction regime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um", [0.25, 0.50, 0.75, 1.0, 2.0])
def test_signal_decreases_with_gradient_strength(R_um):
    """Signal decreases monotonically with G for b < 10 000 s/mm².

    In the pre-diffraction regime, greater gradient strength → more dephasing
    → lower signal.  Diffraction oscillations for R ≤ 2 µm only appear at
    b >> 10 000 s/mm² so the ordering must be strict up to that threshold.
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, params = _load(stem, "pgse_finite_delta")

    # Within each δ-group, sort by G and check monotone decrease in E
    delta_groups: dict[float, list] = {}
    for i, p in enumerate(params):
        d = round(p["delta_s"] * 1e3, 1)
        delta_groups.setdefault(d, []).append((p["G_Tm"], p["b_sm2"] * 1e-6, signals[i]))

    for delta_ms, group in delta_groups.items():
        group_sorted = sorted(group, key=lambda x: x[0])
        prev_E = 1.0
        for G, b_smm2, E in group_sorted:
            if b_smm2 > 10_000:
                break
            assert E <= prev_E + 0.003, (
                f"R={R_um}µm, δ={delta_ms}ms: "
                f"signal not monotone at G={G:.2f} T/m "
                f"(E={E:.4f} > prev={prev_E:.4f})"
            )
            prev_E = E


# ---------------------------------------------------------------------------
# 6. Physical ordering: longer diffusion time → lower E at fixed G
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um", [1.0, 2.0, 3.0, 5.0])
def test_longer_diffusion_time_lower_signal(R_um):
    """Increasing δ and Δ together (δ=5→20 ms) at fixed G=0.3 T/m lowers E.

    As the gradient block grows, both b increases and walkers have more time
    to probe the cylinder wall → more attenuation.  R ≥ 1 µm is used where
    the effect is large enough to be clearly above MC noise.
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, params = _load(stem, "pgse_finite_delta")
    G_target = 0.3

    # Collect (δ_ms, E) for G ≈ 0.3 T/m
    pairs = []
    for i, p in enumerate(params):
        if abs(p["G_Tm"] - G_target) < 0.01:
            pairs.append((p["delta_s"] * 1e3, signals[i]))

    pairs_sorted = sorted(pairs)   # ascending δ
    for (d1, E1), (d2, E2) in zip(pairs_sorted, pairs_sorted[1:]):
        assert E2 <= E1 + 0.002, (
            f"R={R_um}µm, G={G_target} T/m: "
            f"E(δ={d2:.0f}ms)={E2:.4f} not < E(δ={d1:.0f}ms)={E1:.4f}"
        )


# ---------------------------------------------------------------------------
# 7. OGSE frequency ordering: higher f → less restriction at fixed G
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um", [1.0, 2.0, 3.0, 5.0])
def test_ogse_higher_freq_less_restricted(R_um):
    """At fixed G and T, higher OGSE frequency gives HIGHER signal (less restriction).

    Higher f → shorter effective diffusion time T_eff = 1/(2f) → walkers
    sample less of the cylinder → less phase dispersion → higher E.

    Tested for G = 0.3 T/m where signals span a clear dynamic range across
    frequencies for R ≥ 1 µm.
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, params = _load(stem, "ogse_cosine")
    G_target = 0.3

    # Collect (freq, E) for G ≈ 0.3 T/m
    pairs = sorted(
        [(p["freq_Hz"], signals[i])
         for i, p in enumerate(params) if abs(p["G_Tm"] - G_target) < 0.01]
    )

    # Signal should increase with frequency (less restriction at shorter T_eff)
    for (f1, E1), (f2, E2) in zip(pairs, pairs[1:]):
        assert E2 >= E1 - 0.005, (
            f"R={R_um}µm, G={G_target} T/m: "
            f"E(f={f2}Hz)={E2:.4f} not ≥ E(f={f1}Hz)={E1:.4f}"
        )


# ---------------------------------------------------------------------------
# 8a. High-frequency OGSE approaches free diffusion for LARGE R
# ---------------------------------------------------------------------------

def test_ogse_1000hz_approaches_free_diffusion_large_r():
    """At 1000 Hz OGSE (T_eff = 0.5 ms), R = 5 µm cylinder approaches free
    diffusion: τ_c = R²/D = 14.7 ms >> T_eff, so walkers barely feel the wall
    within one half-cycle.

    f_c(R=5 µm) = µ₁²D/(2πR²) ≈ 37 Hz, so 1000 Hz >> f_c → short-time regime
    where D_eff ≈ D_free.  E ≈ exp(-b_OGSE × D) within 0.015 for G ≤ 1 T/m.
    """
    D_free  = 1.7e-9
    signals, params = _load("R5.0um", "ogse_cosine")

    failures = []
    for i, p in enumerate(params):
        if p["freq_Hz"] != 1000 or p["G_Tm"] > 1.01:
            continue
        b_sm2  = p["b_sm2"]
        E_free = np.exp(-b_sm2 * D_free)
        diff   = abs(signals[i] - E_free)
        if diff > 0.015:
            failures.append(
                f"G={p['G_Tm']:.2f} T/m  b={b_sm2*1e-6:.1f} s/mm²  "
                f"MC={signals[i]:.4f}  exp(-bD)={E_free:.4f}  |ΔE|={diff:.4f}"
            )

    assert not failures, (
        "R=5 µm OGSE 1000 Hz not approaching free diffusion:\n" +
        "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 8b. High-frequency OGSE: fully-restricted regime for small R
# ---------------------------------------------------------------------------

def test_ogse_1000hz_fully_restricted_small_r():
    """At 1000 Hz OGSE (T_eff = 0.5 ms), R = 0.25 µm is fully restricted:
    τ_c = R²/D = 37 µs << T_eff, so walkers explore the full cylinder many
    times per half-cycle → D_eff → 0 → E → 1 regardless of G.

    We verify that E > 0.97 for G ≤ 1 T/m, confirming nearly zero effective
    perpendicular diffusivity at this radius and frequency.
    """
    signals, params = _load("R0.25um", "ogse_cosine")

    for i, p in enumerate(params):
        if p["freq_Hz"] != 1000 or p["G_Tm"] > 1.01:
            continue
        assert signals[i] > 0.97, (
            f"R=0.25 µm 1000 Hz: expected full restriction (E > 0.97) "
            f"but E={signals[i]:.4f} at G={p['G_Tm']:.2f} T/m"
        )


# ---------------------------------------------------------------------------
# 9. Trapezoidal OGSE: more lobes → less restriction at same G and δ/Δ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um", [2.0, 3.0, 5.0])
def test_trap_ogse_more_lobes_less_restricted(R_um):
    """Trapezoidal OGSE with more lobes per block gives HIGHER signal at same G.

    More lobes = shorter effective diffusion time = less restriction.
    Checked for G = 0.6 T/m where the dynamic range is clear.
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, params = _load(stem, "ogse_trap")
    G_target = 0.6

    # Collect (N_lobes, E) for G ≈ 0.6 T/m
    pairs = sorted(
        [(int(p["N_lobes"]), signals[i])
         for i, p in enumerate(params) if abs(p["G_Tm"] - G_target) < 0.01]
    )

    for (N1, E1), (N2, E2) in zip(pairs, pairs[1:]):
        assert E2 >= E1 - 0.005, (
            f"R={R_um}µm, G={G_target} T/m: "
            f"E(N={N2})={E2:.4f} not ≥ E(N={N1})={E1:.4f}"
        )


# ---------------------------------------------------------------------------
# 10. All signals in valid physical range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um", ALL_RADII)
@pytest.mark.parametrize("group", [
    "pgse_finite_delta", "pgse_short", "ogse_cosine", "ogse_trap"
])
def test_signals_in_physical_range(R_um, group):
    """All fixture signals lie in [-0.005, 1.005].

    Values below -0.005 or above 1.005 indicate a bug in trajectory
    generation or waveform application (phase integral sign error, etc.).
    The narrow tolerance of 0.005 accounts for MC noise at N=500k walkers
    (σ ≈ 0.0014 per measurement).
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, _ = _load(stem, group)

    out_of_range = [
        (i, float(s)) for i, s in enumerate(signals)
        if s < -0.005 or s > 1.005
    ]
    assert not out_of_range, (
        f"R={R_um}µm, group={group}: signals out of range: {out_of_range}"
    )


# ---------------------------------------------------------------------------
# 11. Low-gradient limit: E → 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R_um", ALL_RADII)
def test_low_gradient_signal_near_unity(R_um):
    """At G = 0.03 T/m and the shortest δ (δ = 5 ms, b ≈ 38 s/mm²), E > 0.99.

    This is the minimum-b measurement in the PGSE finite-delta battery and
    validates that the fixture pipeline has the correct sign convention (E → 1
    as b → 0).  Only the shortest-δ group is tested; longer pulses at R = 5 µm
    have b ≈ 172–859 s/mm² which is physically correctly attenuated.
    """
    stem    = RADIUS_TO_STEM[R_um]
    signals, params = _load(stem, "pgse_finite_delta")

    # Find the minimum-b measurement at G = 0.03 T/m (shortest δ group)
    candidates = [
        (p["b_sm2"], signals[i])
        for i, p in enumerate(params)
        if abs(p["G_Tm"] - 0.03) < 0.001
    ]
    b_min, E_min_b = min(candidates, key=lambda x: x[0])

    assert E_min_b > 0.99, (
        f"R={R_um}µm: at minimum b = {b_min*1e-6:.0f} s/mm² "
        f"(G=0.03 T/m), signal={E_min_b:.4f} < 0.99"
    )
