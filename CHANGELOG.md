# Changelog

## Unreleased

- **Reference catalogue, batch 2: the obvious mistakes fixed** (audit in `docs/catalog_audit.md`, "Batch 2"; documented
  design choices stay).
  - `active_ax()` is the minimal model of white matter diffusion: the Callaghan cylinder and the Zeppelin share one
    orientation and one parallel diffusivity, and the Zeppelin's perpendicular diffusivity is the tortuosity
    D_par v_ec / (v_ic + v_ec). Parameters `G2Zeppelin_1_mu`, `G2Zeppelin_1_lambda_par` and
    `G2Zeppelin_1_lambda_perp` are gone (two free scalars and one orientation fewer). At the parity harness's
    midpoint parameters the signal moves by up to 9.0e-2 of S0 (25 % relative); a fit that used the free Zeppelin
    returned a hindered tensor unrelated to the axon direction and its own diffusivities.
  - `sandi()` is the direction-averaged model of Palombo et al. 2020, Eq. 10: a spherical-mean model (no
    `C1Stick_1_mu`), with the soma diffusivity fixed at the free-water value 3e-9 m^2/s (was 1.58e-9) and the neurite
    diffusivity `C1Stick_1_lambda_par` free (was fixed at 1.7e-9; five free scalars). At soma diameter 8-16 um and
    b = 0.5-6 ms/um^2 the soma signal moves by up to 0.12 of S0 (0.59 to 0.71 at 8 um, b = 6) from the diffusivity
    alone. The spherical-mean JAX forward gains the Gaussian-phase sphere, so `solver="jax"` still works; the parity
    record's `ref.sandi` JAX cases move to `sm_forward/sandi`.
  - `noddida_mte()` is `noddida()` with a T2 per compartment (stick, zeppelin): the Ball and its diffusivity and T2 are
    gone, the model is Watson-dispersed (`odi`), the diffusivities are free in [0, 4] um^2/ms, and there is no
    tortuosity or equality link. Fitted values of the former aligned Stick + Zeppelin + Ball model have no
    counterpart. No paper defines the combination; the docstring says so.
  - Docstrings only: `two_fascicle_noddi()` is not defined by Behrens 2007 (a ball-and-sticks paper); `axcaliber()`'s
    and `DD1Gamma`'s Gamma is on the radius (the mean diameter is 2 alpha beta), as in Assaf 2008 Eq. 5, not on the
    diameter as the docstring said; `charmed()`, `verdict()` and `impulsed()` state what could and could not be
    verified.
  - Parity harness: `rebaseline --only` also drops a matching recorded case that is no longer registered.

- **Reference catalogue, batch 1: the obvious mistakes fixed** (audit in `docs/catalog_audit.md`; documented design
  choices stay).
  - `ball_and_stick()` has one diffusivity shared by stick and ball (Behrens 2003 Eqs. 12-13): the parameter
    `G1Ball_1_lambda_iso` is gone and `C1Stick_1_lambda_par` is d. Fits had two free diffusivities; the ball can no
    longer be slower or faster than the stick.
  - `noddida()` is NODDIDA (Jelescu, Veraart, Fieremans & Novikov 2016, Eqs. 2-5): a Watson-dispersed Stick + Zeppelin
    with free D_a, D_e,par and D_e,perp in [0, 4] um^2/ms, no tortuosity link, no Ball. Its parameter names, count and
    meaning change (`SD1WatsonDistributed_1_*`, plus `odi` and `mu`); fitted values of the former aligned
    Stick + Zeppelin + Ball model have no counterpart (no dispersion, no CSF fraction, tied diffusivities).
  - `bingham_noddi()`: kappa is bounded by 64 (Tariq 2016), the odi lower bound is 0.00994 instead of 0.02.
    Voxels of coherent tissue (kappa between 31.8 and 64) can now fit; elsewhere the signal is unchanged.
  - Docstrings only: `stick_tortuous_zeppelin()` is the dispersion-free, tortuosity-constrained special case of the
    Standard Model, not the Standard Model; `free_water_elimination()` states that its tissue is a Zeppelin, not the
    full tensor of Pasternak 2009 (no full-tensor Gaussian compartment exists in the package); `noddi()`,
    `bingham_noddi()` and `noddida()` state that their dispersed extracellular signal is the ODF average of zeppelin
    signals, which differs from the exponential of the averaged tensor (Zhang 2012 Eq. 4) by 4.5e-3 to 7.3e-3 of S0
    (NODDI), 7.3e-3 to 7.4e-3 (Bingham-NODDI) and 6.0e-3 to 9.8e-3 (NODDIDA) in the tested cases. No fitted value of
    `noddi()` changes.
  - Citations: Le Bihan 1988 is Radiology 168; Kaden, Kelm, Carson, Does & Alexander 2016; NODDIDA is Jelescu,
    Veraart, Fieremans & Novikov, NMR Biomed 29 (2016).
  - Parity record: re-baselined with `rebaseline --only`: `ref.bingham_noddi` (pgse, legacy; the default parameter
    vector moves with the new odi lower bound; max abs 2.9e-4 NumPy, 3.9e-4 JAX) and `ref.noddida` (pgse; a different
    model; max abs 2.7e-1 NumPy, 2.2e-1 JAX). `ref.ball_and_stick` keys are unchanged.

- **Parity harness: `rebaseline --only KEY...` and `rename-case OLD NEW`.** `rebaseline PREFIX...` refused whenever any
  unnamed key differed, which on a machine with another Python, NumPy or JAX than the record's is dozens of keys of
  environment drift. `rebaseline --only` takes case ids or `fnmatch` patterns (each must match a registered case),
  evaluates and rewrites only those, ignores the drift of every other key, leaves the manifest's `environment` block
  as it was and records the written keys under the environment they were written in (`rewrites`). `rename-case`
  renames a case's keys in the manifest and the `.npz` files, array bytes untouched, and verifies each digest under
  the new name. No effect on any model or recorded value.

- **`T1_tortuosity` normalises both fractions (wrong lambda_perp with a third compartment).** Called with two
  explicit fractions, the link divided the intra fraction by `vf_intra + vf_extra` but used the raw extra fraction, so
  whenever the two did not sum to one it returned the wrong perpendicular diffusivity; it now uses the intra fraction of
  the intra+extra bundle, `vf_intra / (vf_intra + vf_extra)` (Szafer et al. 1995), with the S0 weighting unchanged.
  Affected: the NumPy path of any model with a compartment besides the linked intra and extra pair, i.e. `noddida()`
  and `noddida_mte()` (their Ball takes part of the volume); at fractions (0.4, 0.35, 0.25) `noddida()` set
  lambda_perp = 0.396 D_a instead of 0.467 D_a (18 % low), moving the signal by up to 2e-2. Two-compartment models
  (`stick_tortuous_zeppelin()`, `charmed()`, `mcsmt()`, `wmti()`, NODDI) have fractions that sum to one and are
  unchanged; the JAX forward already computed the correct value, so the two paths now agree. The unit test compared
  values of order 1e-9 at 7 decimals and could not fail; it now uses a relative tolerance, fractions summing below one,
  and unequal S0 responses.

- **`reference_models.stick_zeppelin_karger()` removed.** The reference factory holds published models only, each
  faithful to its paper. This one was the former `nexi()`: an orientation-resolved Stick plus a tortuous Zeppelin with
  a one-way Karger rate. No paper defines that composition (Karger et al. 1988 covers the exchange equation, not
  the pairing), so it leaves the factory, with no alias. Tests of the generic Karger machinery build
  `X0GeneralizedKarger(C1Stick(), G2Zeppelin(), parameter_links=[...])` themselves; the parity case is now
  `karger.stick_zeppelin_tortuous`, a rename with the recorded values and hashes unchanged.

- **Published NEXI.** `reference_models.nexi()` is a fittable spherical-mean Stick + isotropic Ball with Karger
  exchange (Jelescu et al. 2022, Eqs. 3--6), with free f, Di, De and t_ex in seconds, no orientation or tortuosity,
  and a JAX twin. The PGSE effective-time
  approximation and compartment-signal floor are unchanged.

- **`brute2fine` local stage runs to convergence.** `Brute2FineOptimizer` refines the coarse stage's best grid point
  with L-BFGS-B on the objective divided by its value there, with both stopping thresholds at the double-precision
  epsilon. L-BFGS-B's thresholds are absolute for an objective below one, so on the mean squared residual of a signal
  attenuation (order 1e-3) with an exchange rate in s^-1 the refinement stopped at the first iterate: a Stick + Zeppelin Karger composition on a
  multi-Delta PGSE protocol returned the nearest grid rate (kappa 50 for a true 20) instead of the global minimum.

- **Filter-exchange imaging.** `AcquisitionScheme.from_fexi` (dmipy-replay's `fexi_protocol`: filter off/on x mixing
  times x detection b-values, one sequence per mixing time); `signal_models.filter_exchange.FEXIApparentExchange`, the
  AXR analysis (`ADC_eq`, `sigma`, `AXR`, one unweighted signal per series, the series eliminated in closed form) whose
  equation is `dmipy_analytic.exchange.fexi_signal`, every row's filter and detection b-value and mixing time read off
  the sequences (`fexi_measurements`). The Kärger models (`X0GeneralizedKarger`, `fexi`) evaluate a FEXI
  sequence by the generalised Kärger equation along its own effective gradient
  (`dmipy_analytic.exchange.karger_waveform`), each pool's diffusivity along the row's direction; a scheme mixing FEXI
  sequences with others is refused. On the two-site signal the AXR fit converges to `k_12 + k_21` at first order in the
  detection b.

## 3.0.0 - 2026-11-02

dmipy-fit fits multi-compartment diffusion-MRI models to data, voxel by voxel, and returns interpretable
microstructure maps. It is the analytical inverse of the dmipy ecosystem: the equations are
[dmipy-analytic](https://github.com/dmrai-lab/dmipy-analytic)'s, the signal side
[dmipy-replay](https://github.com/dmrai-lab/dmipy-replay), the forward engine
[dmipy-sim](https://github.com/dmrai-lab/dmipy-sim).

- **Models.** Sticks, cylinders, spheres, planes, Gaussian compartments and tissue responses, with Watson and Bingham
  dispersion and gamma diameter distributions.
- **The equations are dmipy-analytic's.** The signal models keep their parameters, ranges, orientation handling and
  composition and call dmipy-analytic's forms (`gaussian`, `restricted`, `exchange`, `relaxation`, `orientation`,
  `harmonics`, `constants`); the JAX optimisers call its twins. No equation is left in dmipy-fit, and no re-export stands in for one.
- **Restricted diffusion for any waveform.** The exact matrix-method compartments `P5PlaneMatrixMethod`,
  `C5CylinderMatrixMethod` and `S5SphereMatrixMethod` (the Callaghan 1997 / Grebenkov 2007 multiple-correlation-function
  solution) are exact for an arbitrary gradient waveform (PGSE, OGSE, fixed direction) up to eigenmode truncation; the
  eigenmode solver is dmipy-analytic's (`dmipy_analytic.restricted.matrix_method`), once. The Gaussian-phase compartments
  `P4PlaneGaussianPhaseApproximation` (the finite-pulse approximation for a 1-D slab, Balinov et al. 1993), `C4`
  (Van Gelderen) and `S4` (Murday-Cotts) are their approximations where the Gaussian-phase assumption holds, and depart
  at high b and short pulses. `use_jax=True` on `C5`/`S5`/`P5` evaluates a differentiable `jax.lax.scan` Strang
  propagator, so gradients flow with respect to diameter, orientation and diffusivity and a whole slice fits with
  `solver="jax"` (it needs the `jax` extra and a stored waveform; a scalar-timing scheme falls back to NumPy).
- **Relaxation.** T2, T1 (PGSTE mixing time) and surface relaxivity are occupancy-gated factors that attach to any
  compartment.
- **Replay compartments.** `C6MonteCarloReplayCylinder`, `S6MonteCarloReplaySphere` and `P6MonteCarloReplayPlane` are
  the generalized pore: the signal is the replay of a stored Monte-Carlo walk (a replay pack from
  [SubstrateCommons](https://huggingface.co/SubstrateCommons)), exact to the walk's floor for any gradient waveform,
  with the fit parameters of the analytic cylinders and spheres (`mu`, `diameter`; `diameter` interpolates across the
  family's packs). `MonteCarloReplay(pack)` poses any one pack as a compartment by the lab direction of its axis. A pack
  is its walk as bands on one basis (continuous sines over each storage window); the acquisition is compiled onto the
  bands once, with dmipy-replay's kernel, so each replay is a single matmul. A pack written on a save grid or by an older
  producer is refused by name: re-walk it. Surface relaxivity is exact, not an `exp(-TE rho S/V)` tag-on: a
  `surface_relaxivity` factor activates the pack's boundary-local-time replay knob, optionally gated by a transverse
  gate on the acquisition's own grid.
- **Signal look-up form.** For single-diffusion-encoding PGSE at fixed timing, `build_pgse_kernel` lowers a replay family
  once to a KB-scale kernel of `E(diameter, b, cos_theta[, rho])` (`ReplayKernel`), and fitting interpolates it per
  measurement. It is a fit accelerator, interpolation-grade (the signed signal is interpolated so that diffraction zeros
  do not inflate); the replay compartments stay exact, and other waveforms use them.
- **Fixes.** The capped cylinders (`CC2`, `CC3`) attenuate along the axis alike for a gradient and its reverse: the
  signed projection of the wave-vector reaches the forms, which are even in it, where a gradient against the axis read the
  cross-section's attenuation alone. The JAX forward of the Stick + Zeppelin Karger composition reads the parameters the model exposes (it raised a
  `KeyError`), and a pair the closed form is not wired for is refused by name. `S3` and `S4` sum over dmipy-analytic's shared
  Neumann roots, with no private copy of a root left (their values move by at most 2e-10).
- **Deleted.** `white_matter.surface.b_hat_ea_short`, whose exponent was not dimensionless and which nothing called, and
  with it the module `white_matter.surface`, which held nothing else.
- **Fitting on the device.** `solver="jax"` fits a whole slice on the GPU; batched DTI, Tournier 2007 CSD and
  multi-shell multi-tissue CSD run on JAX or PyTorch. The JAX spherical mean of the zeppelin covers the full range of its
  diffusivities, prolate, spherical and oblate tensors alike (the oblate average is dmipy-analytic's Gauss-Legendre twin).
  dipy is not a runtime dependency (the `oracle` extra only).
- **Acquisition.** `AcquisitionScheme` reads a `ScannerSequence` (dmipy-replay's acquisition object, built by the
  `pgse`, `pgste`, `ogse`, `cpmg`, `ste` and `pte` builders) or a `Protocol` of them.
- **Pairs with** dmipy-analytic `>=3,<4` (`[special]`, and `[jax]` in the `jax` and `jax-gpu` extras), dmipy-sim `>=3,<4` and
  dmipy-replay `>=0.2,<0.3`. Python >= 3.10, NumPy >= 2.0, SciPy >= 1.11;
  extras `jax`, `jax-gpu`, `torch`, `speed`, `data`, `hub`, `viz`, `legacy`, `oracle` and `all`.
- **Licence** AGPL-3.0-only or commercial.

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
