# Changelog

## 2.3.0

**Relaxation–exchange on GPU, b-tensor encoding for restricted models, and a full-spectrum
compatibility test.** Relaxation and surface relaxivity are now consistently opt-in
`OccupancyGatedModel` factors across the whole model zoo, and a registry-driven stress test
exercises every compartment × scheme × framework × wrapper combination.

### Added
- **Coupled compartment-wise T2/T1 relaxation–exchange** in `X0GeneralizedKarger` on the JAX
  backend — a dimension-agnostic (N-pool-ready) matrix-exponential SE/STE propagator; relaxation is
  supplied per compartment via `OccupancyGatedModel`. Fast scalar path retained for the
  no-relaxation case; finite-RF combinations are refused rather than silently approximated.
- **Gaussian-phase models on multidimensional b-tensor encoding** — `S4SphereGaussianPhaseApproximation`
  (per-Cartesian-component GPA, validated to the analytic PGSE colinear limit) and
  `C4CylinderGaussianPhaseApproximation` (existing gamma_lm path, unblocked) now evaluate STE/PTE and
  arbitrary waveforms on both the NumPy and JAX engines.
- **Gamma-averaged sphere surface relaxivity** `IntraSphereSurfaceRelaxivity` (+ `white_matter.surface.b_hat_sphere`),
  the sphere sibling of `IntraPoreSurfaceRelaxivity`.
- **Geometry-aware `SurfaceRelaxivity`** — S/V follows the base compartment (sphere 6/d, cylinder 4/d,
  plane 2/d), auto-bound via `OccupancyGatedModel`; `exterior_surface_to_volume` now takes a required
  `geometry`.
- **Robust multi-modal scheme concatenation** — `AcquisitionScheme.concatenate` resamples every
  waveform onto a common `dt`, so PGSE + PGSTE + OGSE + b-tensor STE/PTE combine without corrupting
  the b-tensor blocks.
- **Fittable capped-cylinder `length`**; `tau` on the spherical-mean scheme; full-spectrum
  initialization + nesting stress tests (`core/tests/test_full_spectrum_*`).

### Fixed
- Spherical-mean crashes for isotropic restricted models: `S4`/`S3` (missing `shell_delta`/`tau`) and
  `DD1Gamma`/`DD2Poisson` around a sphere/dot (`mu_param`); latent `shell_delta` reach in `C4`.
- `CC3CappedCylinderCallaghanApproximation` construction (stale `n_roots` kwarg); zero-parameter models
  (e.g. a lone `S1Dot`) now simulate.
- `AcquisitionScheme.btensor()` uses the exact analytic rank-1 b for colinear PGSE/PGSTE (removes an
  O(1/n_t) quadrature error that mis-scaled anisotropic Gaussian models).

### Changed (breaking)
- **Relaxation/relaxivity are opt-in factors only.** `surface_relaxivity` (and `T2`/`T1`) are no longer
  baked into bare compartments (`S2/S3/S4`, `C2/C3/C4`) or `X0GeneralizedKarger`; add them via
  `OccupancyGatedModel([... , TransverseRelaxation()/SurfaceRelaxivity()])`.
- `_S3SphereCallaghanApproximation` → **`S3SphereCallaghanApproximation`** (underscore dropped).
- `DD2Poisson` mean-diameter parameter renamed `mu` → **`mean_diameter`** (avoids colliding with the
  fibre orientation, so Poisson distributions work on anisotropic compartments).
- `SurfaceRelaxivity` with neither a base `diameter` nor an explicit `surface_to_volume` now raises
  (was a silent no-op).

## 2.2.0

**Compartment-wise T1 (gated longitudinal relaxation) + PGSTE** — the analytical, occupancy-gated
sibling of `TransverseRelaxation`, coordinated with dmipy-sim 2.2.0.

### Added
- **`LongitudinalRelaxation`** factor (`signal_models/attenuation.py`, + pure-JAX builder) —
  `exp(−τ∥/T1)` with `τ∥ = scheme.TM`, and the identity on a plain spin echo (no TM). During a
  stimulated-echo mixing time the magnetisation is stored along the field, so T2 and surface
  relaxivity gate off and only T1 acts. T1 range 1e-2…10 s.
- **`AcquisitionScheme.from_pgste`** — stimulated-echo scheme constructor (`Δ = δ + TM`, transverse
  `TE` default `2·δ`, hard pulses only); the spherical-mean scheme carries per-shell `TM` so the
  factor flows through like `TE` does for the transverse factors.
- **Per-compartment T1** in `white_matter/composition.py` (`OccupancyGatedModel_<n>_T1`).

Magnetisation is treated as fully transverse on a spin echo (ideal instantaneous pulses); the fit
path applies no constant stimulated-echo amplitude (degenerate with `S0`).

## 2.1.0

Coordinated release alongside dmipy-sim 2.1.0 (mesh substrates). Fit-side changes are model
corrections and agent ergonomics.

### Fixed
- **`_S3SphereCallaghanApproximation`** — corrected sphere Neumann roots and the finite-time
  short-gradient-pulse (SGP) series.
- **Multi-TE models** — expose per-compartment T2 in the `mte_*` models; dropped the redundant
  `X2NEXIModel`.

### Added
- **Agent guide** (`CLAUDE.md`) — how to drive the analytical inverse engine efficiently.

## 2.0.0

First coordinated public release of the dmrai ecosystem: dmipy-fit, dmipy-sim, and the
`dmipy` umbrella version in lockstep (shipped as `2.0.0`; tag `v2.0.0`).

Public-release overhaul focused on the white-matter physics and real-data ergonomics.

### Added
- **Bundled HCP 3T/7T example slice.** A matched coronal slice of HCP subject 191841 at
  both field strengths, loadable in one line:
  `dmipy_fit.data.saved_data.hcp_191841_coronal_slice(field="3T"|"7T")`. The returned scheme
  carries per-measurement `TE`, `b0_direction` and `B0_magnitude`, ready for the relaxation /
  susceptibility factors. Includes brain masks and T1w underlays. See the HCP acknowledgement
  in `dmipy_fit/data/hcp_191841/NOTICE`.
- **Cached Monte Carlo reference arrays** (`dmipy_fit.data.saved_data.mc_reference(...)`) for
  surface relaxivity, the `sin⁴θ` susceptibility law and the cross-term, so the parity figures
  reproduce with no simulation (GPU-generated from dmipy-sim, shipped as cached arrays).
- **Sequence-agnostic forward modelling**: tutorial `01.3` showing `AcquisitionScheme`
  factories (`from_pgse/pgste/gre/cpmg/ogse/waveform/btensor_*`) composed into one scheme
  with `+`, one forward model over mixed sequence families, and the EPG lowering — the same
  scheme object drives both dmipy-fit and dmipy-sim.
- **Susceptibility-aware CPMG / myelin water fraction** (`dmipy_fit.white_matter.cpmg`):
  inject a diffusion FOD into a multi-echo $T_2$ fit to remove the orientation-dependent
  ($\sin^4\theta$) extra-axonal susceptibility bias in MWF; Monte Carlo ground truth
  `dmipy_sim...UnifiedWhiteMatterModel.simulate_cpmg_susceptibility`. Tutorial `08.6`.
- **New tutorial sections**:
  - `08_white_matter_pathways/` — occupancy-gated models, the unified white-matter model,
    surface relaxivity, the susceptibility `sin⁴θ`/B0² law, the diffusion×susceptibility
    cross-term, and susceptibility-aware MWF.
  - `09_real_data_3T_7T/` — a tour of the bundled slices and susceptibility-aware fitting in
    vivo (the cross-field test).
  - `10_reproductions/` — reproducing the Monte Carlo parity figures from the cached arrays.
- **Documentation site.** Jupyter Book config (`_config.yml`, `_toc.yml`) deployed to GitHub
  Pages by `.github/workflows/docs.yml`; lightweight notebooks are re-executed in CI (nbmake).

### Changed
- README overhauled around the coherence-pathway signal model (six physical effects) with a
  10-line quick-start on the bundled slice and a hero figure.
- `pyproject.toml` now ships `data/hcp_191841/*` and `data/mc_reference/*`.
- Slimmed the bundled data to real in-vivo examples only. Removed the cat spinal-cord
  histology (`tanguy_cat_spinal_cord`), the ISBI-2015 and de Santis challenge data, and the
  **Camino synthetic signals** (+ the Monte-Carlo equivalence test) — dmipy-sim is the
  toolbox's own, superior simulator. Dropped the b=10000 shell from the MGH-1010 slice
  (27 → 14.5 MB; shells 0/1000/3000/5000 retained). Kept `sos_constraints` (CSD optimizer)
  and `gradient_tables` (scheme helpers). External datasets are documented in the README rather than bundled.

### Removed
- The broken `wu_minn_hcp_coronal_slice()` stub (it pointed at data that was never bundled and
  required AWS credentials). Replaced by `hcp_191841_coronal_slice`.

## 2.0.0

- GPU fitting via JAX, Rician noise model, b-tensor / free-waveform schemes, T2/T1 relaxation,
  Monte Carlo integration, lighter install. See the README.
