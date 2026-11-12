# Working in dmipy-fit

dmipy-fit is the analytical inverse engine for diffusion microstructure MRI:
it fits multi-compartment tissue models voxel by voxel and returns parameter maps.
Diffusion and occupancy-gated relaxation/surface-relaxivity factors compose on
compartments. It is not the Monte-Carlo forward simulator or the sequence-design
engine. Dependency direction is fit → sim, never sim → fit; closed-form equations
belong in dmipy-analytic, and acquisition objects/builders in dmipy-replay.

Read `CLAUDE.md` for model examples and operational details, then `README.md`,
`CONTRIBUTING.md`, and `.github/workflows/tests.yml` for the current test contract.
Report stale guidance against code/CI rather than silently propagating it.

## Setup and checks

Use Python 3.11 or 3.12 (the CI matrix) in an isolated environment, from the repo
root. CI currently installs the three public dependency revisions named by
`DMIPY_SIM_REF`, `DMIPY_REPLAY_REF`, and `DMIPY_ANALYTIC_REF` in `tests.yml`;
do not substitute unpinned releases while that workflow pins unreleased versions.
Set those three variables to the workflow's values, then install its dependencies
and this checkout (the interpreter must have pip):

```bash
python -m pip install \
  "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim@${DMIPY_SIM_REF}" \
  "dmipy-replay @ git+https://github.com/dmrai-lab/dmipy-replay@${DMIPY_REPLAY_REF}" \
  "dmipy-analytic[special,jax] @ git+https://github.com/dmrai-lab/dmipy-analytic@${DMIPY_ANALYTIC_REF}"
python -m pip install --no-deps -e .
python -m pip install "jax[cpu]>=0.6.2" "jaxopt>=0.8" "numba>=0.57" \
  "nibabel>=5.0" "matplotlib>=3.5" "cvxpy>=1.3" "pathos>=0.3" \
  "huggingface_hub>=0.20" "requests>=2.28" "safetensors>=0.4" \
  "dipy>=1.7" "scipy>=1.10" "packaging>=23.0" pytest pytest-cov nbmake jupytext
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run the regular CI test command before hand-back:

```bash
pytest dmipy_fit/ tests/ -q --tb=short -m "not slow"
```

The slow suite includes costly recovery fits and Monte-Carlo validation. Leave
the full battery to an adequately provisioned CI runner rather than a laptop-size
memory budget. CI runs it on CPU, on Python 3.12 for main pushes or manual dispatch:

```bash
pytest dmipy_fit/ tests/ -q --tb=short -m slow
```

CI also executes these nine lightweight notebooks on Python 3.12:

```bash
JAX_PLATFORMS=cpu pytest --nbmake -q \
  examples/flagship_canonical_wm/canonical_wm_parity.ipynb \
  examples/01_acquisition_schemes/waveform_first_acquisition_scheme.ipynb \
  examples/01_acquisition_schemes/loading_schemes_from_files_and_dipy.ipynb \
  examples/02_signal_models/build_multicompartment_model.ipynb \
  examples/03_fitting_and_optimization/simulating_and_fitting.ipynb \
  examples/03_fitting_and_optimization/ivim.ipynb \
  examples/05_distributed_models/noddi_watson.ipynb \
  examples/06_gpu_acceleration/jax_gpu_fitting.ipynb \
  examples/11_acquisition_families/btensor_microscopic_anisotropy.ipynb
```

For signal arithmetic changes, the separate CI parity job records the base on
the same runner using `tests/parity_analytic/record_base.sh`, installs the pinned
`tests/parity_analytic/constraints.txt` environment without numba, then runs:

```bash
python tests/parity_analytic/harness.py check --reference base/tests/parity_analytic/record \
  --base-manifest "$RUNNER_TEMP/base_manifest.json"
```

Here `base/` and the saved manifest are CI-prepared inputs, not another checkout
for a test to create. Do not rebaseline records just to hide arithmetic drift.
Reusable guard/integration workflows provide additional CI gates; release build
and publishing workflows are for maintainers. Report checks you could not run.
Run one local Python job at a time under a small memory cap; allow subprocesses
the tests themselves spawn. Never add a second sandbox to change test behavior.

## Layout and model contracts

- `dmipy_fit/core/`: acquisition schemes, full/spherical-mean/SH fitting, fit results.
- `dmipy_fit/signal_models/`: compartment classes and occupancy-gated factors;
  equations call dmipy-analytic. Matrix-method adapters connect exact restricted
  signals to the shared eigenmode solver; distinguish these from low-b GPA forms.
- `dmipy_fit/custom_optimizers/reference_models.py`: named literature factories;
  use these for NODDI, NEXI, SANDI, etc., rather than hand-assembling replacements.
- `dmipy_fit/distributions/`, `optimizers/`, `optimizers_fod/`: dispersion and solvers.
- `dmipy_fit/jax/`, `dmipy_fit/torch/`: accelerated fitting and signal backends.
- `dmipy_fit/white_matter/`, `tissue_response/`, `audit/`, `utils/`: composition,
  MWF, response estimation, citations, and shared helpers.
- Tests live beside modules in `tests/` subdirectories and in top-level `tests/`;
  `tests/parity_analytic/` holds signal identity records and their harness.
- `examples/` contains tutorials; paired Markdown is the editable notebook source.
  `dmipy_fit/data/` and signal-model fixture directories contain reference assets.

Use SI: b-values s/m², diffusivity m²/s, lengths m, times s. Read physical
constants through public accessors, not copied literals. Get parameter names
from `model.parameter_names` or `fit.fitted_parameters`; do not guess indices.
In a model's per-measurement call use `delta`/`Delta`; `shell_delta`/`shell_Delta`
belong only to per-shell paths. Spherical-mean fits do not estimate orientation
or dispersion; use the full or SH framework for those quantities.

## Implementation rules

- Unify representations by removing the old spelling and converting every caller;
  no compatibility shims, aliases, or legacy backdoors.
- Required input must be present: missing means an error, or `None` with a named
  flag. No numeric lookup defaults, default arguments, or clamps substituting
  for missing required values. Do not swallow failures or silently switch paths.
- Never read another module's private attributes; use its public function/record.
- Write declarative docstrings describing behavior, inputs, units, and limitations.
  Implement a published model as its primary paper defines it; read the source
  and cite it, or state exactly what could not be verified.
- Check correctness in float64 first; float32 is for production speed only when
  its error against float64 is below the physical noise floor. Preserve
  `conftest.py`'s per-test x64 policy: production float32 baseline, explicit
  float64 modules for parity/overflow-sensitive models, reset after each test.
  Do not globally enable x64 to fix one test; respect model precision declarations.
- New behavior needs a test that fails without it; new physics also needs an exact
  reference/limiting-case validation. Tests must work in CI and on supported
  runners without git, network, or input files outside this repository.
- Keep assertions for known defects; use `pytest.mark.xfail(strict=True,
  reason="<issue>: ...")` for an open issue, never weaken expectations to pass.
- Derive numerical bounds from error models. Flag changes to thresholds,
  definitions, scope, or units explicitly as decisions in the hand-back.

## Boundaries and hand-back

Never push, merge, tag, release, publish, send external messages, delete data, or
read/change secrets or credentials. The lead reviews and commits. If commits are
explicitly requested, make new commits; never amend or force-push a pushed branch.
Public code, docs, tests, commits, and PR text must name no private repository,
issue, method, machine, host, or provider. Prefer free CPU/reference checks;
paid compute requires a stated estimate and explicit authorization.
Wait only on recorded PIDs; never use process-pattern kill/wait commands.
Read the named remote ref when comparing with a base, not a stale local branch.

Hand back the change summary, numbered requirements with status and evidence,
exact commands/outcomes (including failures, skips, and unrun checks), decisions,
unverified claims, PR link if any, and compute spend. Include first-person friction
with minutes lost, workarounds, and the one change that would help most. When the
brief supplies a structured hand-back schema/checker, write and validate that
artifact as instructed; keep operational/private details out of public artifacts.
