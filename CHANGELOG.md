# Changelog

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

> Note: the `## 1.0.0` heading below predates the `v2.0.0` tag/`pyproject` version and is
> retained as historical record; the shipping version line is 2.x.

## 1.0.0

Re-baselined from the inherited `2.0.0` to `1.0.0` for the first coordinated dmrai ecosystem
release: dmipy-fit, dmipy-sim, and dmrai now version in lockstep.

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
