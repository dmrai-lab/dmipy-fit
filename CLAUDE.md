# dmipy-fit — Agent Guide

**Read this file, not the whole tree.** dmipy-fit is built to be *operated by agents* (any
vendor): this guide is the operational contract — what the package does, the entry points,
the common tasks with copy-paste code, and where to look for the rest.

dmipy-fit is the **analytical inverse** of the dmipy framework: given an acquisition scheme
and a multi-compartment tissue model, it fits parameters voxel-by-voxel and returns
interpretable microstructure maps (volume fractions, diffusivities, axon radius/dispersion,
T2, …). Its forward counterpart is the Monte-Carlo simulator
[dmipy-sim](https://github.com/dmrai-lab/dmipy-sim) (see its `CLAUDE.md`). **You describe the
tissue once**; both engines consume the same `AcquisitionScheme`. The dependency is
one-directional: **fit → sim** (fit may import sim for parity; sim never imports fit).

## Mental model

$$S = S_0 \sum_i f_i\, E^{\text{diff}}_i(b)\, e^{-\mathrm{TE}/T_{2,i}}\, \hat B^{\text{surf}}_i$$

A model is a list of **compartments** (sticks, cylinders, spheres, Gaussians/ball-zeppelin,
planes), optionally wrapped in **occupancy-gated factors** (T2, surface relaxivity) that
attach to *any* compartment and compose by listing more factors. Fitting estimates the
`f_i` (partial volumes) and each compartment's parameters. Magnetisation is treated as fully
transverse (ideal instantaneous pulses).

## Environment

```bash
pip install "dmipy-fit[jax]"          # + GPU fitting (solver="jax"); [all] for everything
pytest -q -m "not slow"               # fast suite; drop the filter for the heavy GPU battery
```

GPU: fitting whole slices uses JAX (`solver="jax"`). If a CUDA jaxlib is installed but
`jax.devices()` shows only CPU, export `LD_LIBRARY_PATH` to the venv's `nvidia/*/lib` dirs.
Correctness is defined by the **test suite** — analytical results, dipy/MISST references, and
**analytic ↔ Monte-Carlo parity** against dmipy-sim; keep it green.

## Common tasks (copy-paste)

**Build an acquisition scheme** (shared with dmipy-sim; b-values in SI, s/m²):
```python
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
scheme = AcquisitionScheme.from_pgse(bvalues, gradient_directions, delta=0.01, Delta=0.03)
# also: from_pgste, from_cpmg, from_ogse, from_waveform, from_btensor_{ste,pte,waveform}
```

**Fit a model** → maps:
```python
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.core.modeling_framework import MultiCompartmentModel

model = MultiCompartmentModel([G1Ball(), C1Stick()])
fit = model.fit(scheme, data, solver="jax")          # whole slice on GPU
f_intra = fit.fitted_parameters["partial_volume_1"]  # per-compartment params by name
```

**Use a named literature model** (don't hand-assemble NODDI etc.):
```python
from dmipy_fit.custom_optimizers.reference_models import noddi, nexi, sandi, verdict, impulsed
fit = noddi().fit(scheme, data, solver="jax")
```
Available: `ball`, `zeppelin`, `ball_and_stick`, `ball_and_zeppelin`, `stick_tortuous_zeppelin`,
`noddi`, `bingham_noddi`, `noddida`, `mcsmt`, `charmed`, `axcaliber`, `active_ax`, `verdict`,
`sandi`, `impulsed`, `nexi`, `ivim`, `free_water_elimination`, `two_fascicle_noddi`.

**Myelin-water fraction from a multi-echo T2 decay** (e.g. a dmipy-sim CPMG train):
```python
from dmipy_fit.white_matter.mwf import t2_spectrum_mwf
mwf, T2_grid, spectrum = t2_spectrum_mwf(signal / signal[0], echo_times)   # NNLS T2 spectrum
```

**Canonical white-matter model** (diffusion-only, T2 + surface-relaxivity gated):
```python
from dmipy_fit.white_matter.composition import build_white_matter_model
model = build_white_matter_model(include_csf=False)
```

## Module map (`dmipy_fit/`)

| Path | Role |
|------|------|
| `core/modeling_framework.py` | `MultiCompartmentModel` + `.fit(scheme, data, solver=…, mask=…)` |
| `core/spherical_mean_framework.py`, `core/spherical_harmonics_framework.py` | spherical-mean & SH (CSD/FOD) frameworks |
| `core/fitted_modeling_framework.py` | fit result: `.fitted_parameters`, multi-tissue fractions |
| `core/acquisition_scheme.py` | `AcquisitionScheme.from_*` (PGSE/PGSTE/CPMG/OGSE/waveform/b-tensor) |
| `signal_models/` | `cylinder_models` (stick/cylinder/axcaliber), `gaussian_models` (ball/zeppelin), `sphere_models`, `plane_models`, `capped_cylinder_models`, `tissue_response_models`, `exchange_models` |
| `signal_models/attenuation.py` | `OccupancyGatedModel` + `TransverseRelaxation`, `IntraPoreSurfaceRelaxivity`, `ExteriorSurfaceRelaxivity` |
| `distributions/` | Watson / Bingham dispersion, Gamma diameter distribution |
| `optimizers/`, `optimizers_fod/` | brute2fine, MIX, multi-tissue NNLS; CSD (Tournier / cvxpy / OSQP-JAX) |
| `custom_optimizers/reference_models.py` | named literature models (NODDI, NEXI, SANDI, VERDICT, IMPULSED, …) |
| `jax/` | GPU signal models, `vmap_fit`, DTI/CSD/fractions — the `solver="jax"` backend |
| `white_matter/` | `build_white_matter_model()`, `mwf.t2_spectrum_mwf()` |
| `tissue_response/`, `audit/` (`biophysical_constants`), `utils/`, `_gpu_config.py` | responses, cited constants, helpers, GPU mem cap |

## Where to look for X

- **Orientation dispersion** → `distributions/` (Watson/Bingham) + SH framework.
- **CSD / fibre ODFs** → `core/spherical_harmonics_framework.py`, `optimizers_fod/`.
- **T2 / surface relaxivity on a compartment** → `signal_models/attenuation.py`
  (`OccupancyGatedModel`).
- **GPU / whole-slice speed** → `solver="jax"`, `jax/`.
- **A standard model by name** → `custom_optimizers/reference_models.py`.
- **Cross-checking against ground truth** → build the same tissue in dmipy-sim, fit the MC
  signal; parity tolerance is roughly `max(0.02, 1/√N)`.

## Gotchas

- **SI units.** b-values in s/m², diffusivities in m²/s, lengths in m, times in s.
- **Parameter names** are positional: `partial_volume_0/1/…` and
  `<CompartmentName>_1_<param>` — read them off `model.parameter_names` /
  `fit.fitted_parameters`, don't guess.
- **Spherical-mean vs full**: the spherical-mean framework fits orientation-invariant
  parameters (no dispersion/orientation); use the full or SH framework when you need those.
- **`solver="jax"` needs the `[jax]` extra**; without a GPU it runs on CPU JAX (slower, still
  correct).
- Don't re-implement NODDI/NEXI/SANDI — call `reference_models`.
