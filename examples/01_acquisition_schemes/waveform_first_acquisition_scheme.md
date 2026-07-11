---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Waveform-first acquisition schemes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/01_acquisition_schemes/waveform_first_acquisition_scheme.ipynb)

**Learning objective:** construct a dmipy `AcquisitionScheme` from PGSE parameters and from a
free gradient waveform `G(t)`, and see that b-values are a *derived* quantity, not the primary
representation — the design that lets the same scheme drive both the analytical models and the
Monte-Carlo simulator.

**Units:** b-values in s/m² (multiply s/mm² by 1e6), gradient strengths in T/m, timing in seconds.

```{code-cell} ipython3
# On Google Colab this installs the engines (public on GitHub); locally it is a no-op.
# Once dmipy is on PyPI this becomes simply:  pip install dmipy
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## 1. Why waveform-first?

A physical MRI sequence is defined by its gradient waveform `G(t)` — the time-varying gradient
vector in T/m. PGSE, OGSE and PGSTE are all just specific waveform *shapes*, and the b-value,
q-value and gradient strength are all *derived* from the waveform by integration.

`AcquisitionScheme` stores `G(t)` as its primary state:

- `G`: `(n_measurements, n_t, 3)` float32 T/m — the raw gradient waveform,
- `dt`: float seconds — the uniform timestep,

and exposes `bvalues`, `qvalues`, `gradient_strengths` as *computed* properties. This is what
lets a scheme feed both the analytical fit (which reads `.bvalues`) and the Monte-Carlo
simulator in `dmipy-sim` (which reads `.waveform.G` directly).

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from dmipy_fit.core.acquisition_scheme import (
    AcquisitionScheme,
    acquisition_scheme_from_bvalues,   # legacy function — still works
)
```

## 2. `AcquisitionScheme.from_pgse` — from PGSE parameters

For standard PGSE experiments use `from_pgse`. The interface mirrors the legacy
`acquisition_scheme_from_bvalues`, but the returned object carries the full waveform.

```{code-cell} ipython3
rng = np.random.default_rng(42)

# b=0 measurements + three DWI shells (HCP-like), b in s/m²
bvals = np.concatenate([np.zeros(5), np.repeat([1e9, 2e9, 3e9], 20)])
dirs  = np.vstack([np.tile([0., 0., 1.], (5, 1)), rng.standard_normal((60, 3))])
dirs[5:] /= np.linalg.norm(dirs[5:], axis=1, keepdims=True)

delta, Delta = 0.0106, 0.0431          # 10.6 / 43.1 ms (HCP)
scheme = AcquisitionScheme.from_pgse(bvals, dirs, delta, Delta)
print("type:", type(scheme).__name__, "| n_measurements:", len(scheme.bvalues))
```

## 3. The `.waveform` view

`.waveform` returns a named tuple `(G, dt, ...)` — the raw representation the simulator eats.

```{code-cell} ipython3
wv = scheme.waveform
print("G shape (n_meas, n_t, 3) T/m:", wv.G.shape, "| dt:", round(wv.dt * 1e3, 3), "ms")

t = np.arange(wv.G.shape[1]) * wv.dt * 1e3            # ms
m = 5                                                 # first DWI measurement
fig, axes = plt.subplots(3, 1, figsize=(7, 3.5), sharex=True)
for ax, ch, lab in zip(axes, range(3), ["Gx", "Gy", "Gz"]):
    ax.plot(t, wv.G[m, :, ch]); ax.set_ylabel(lab + " [T/m]", fontsize=9)
axes[-1].set_xlabel("time [ms]"); axes[0].set_title(f"PGSE waveform, measurement {m}")
plt.tight_layout()
```

## 4. b-values are derived

`.bvalues` is computed from the waveform integral
$b = \gamma^2 \int_0^{TE} |\int_0^t G(t')\,dt'|^2\,dt$. For PGSE this carries a tiny (~0.16%)
systematic error vs the analytic formula at `n_t=1000` — well within the MRI noise floor.

```{code-cell} ipython3
b_input, b_derived = 3e9, scheme.bvalues[scheme.shell_indices == scheme.shell_indices.max()].mean()
print(f"b=3000 s/mm²: input {b_input:.0f} vs derived {b_derived:.0f} s/m²"
      f"  ({abs(b_derived - b_input) / b_input * 100:.3f}% error)")
```

## 5. `AcquisitionScheme.from_waveform` — from an arbitrary `G(t)`

For OGSE / PGSTE / scanner waveforms, or anything produced by `dmipy-sim`, build from the raw
array. Pass `delta`/`Delta` for models that need q-values or gradient strengths.

```{code-cell} ipython3
n_t, d, D, G_mag = 1000, 0.010, 0.030, 0.040        # 10/30 ms, 40 mT/m
dt = (D + d) / n_t
bvecs = np.array([[1., 0., 0.], [0., 1., 0.]])
G = np.zeros((2, n_t, 3), np.float32); tt = np.arange(n_t) * dt
for i, bv in enumerate(bvecs):                       # two-lobe PGSE trapezoid
    G[i, (tt >= 0) & (tt < d)]        =  bv * G_mag
    G[i, (tt >= D) & (tt < D + d)]    = -bv * G_mag
scheme_wf = AcquisitionScheme.from_waveform(G, dt, bvecs, delta=d, Delta=D)
print("from_waveform ->", type(scheme_wf).__name__, "| n_meas:", len(scheme_wf.bvalues))
```

## 6. Any scheme drives the analytical models identically

```{code-cell} ipython3
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.modeling_framework import MultiCompartmentModel

bv, gd = np.array([0., 1e9, 2e9]), np.tile([0., 0., 1.], (3, 1))
legacy = acquisition_scheme_from_bvalues(bv, gd, 0.010, 0.030)
new    = AcquisitionScheme.from_pgse(bv, gd, 0.010, 0.030)
model  = MultiCompartmentModel(models=[G1Ball()])
E_legacy = model(legacy, G1Ball_1_lambda_iso=2e-9)
E_new    = model(new,    G1Ball_1_lambda_iso=2e-9)
print("max |E_legacy - E_new| =", float(np.max(np.abs(E_legacy - E_new))))
```

## Summary

- `AcquisitionScheme` is waveform-first: `G(t)` is primary; b/q/gradient-strength are derived.
- Use `from_pgse` for standard PGSE, `from_waveform` for arbitrary/free waveforms.
- Every analytical model works unchanged (they read `.bvalues`), and the same scheme bridges to
  Monte-Carlo simulation in `dmipy-sim` via `.waveform.G`.

Next: **loading real acquisition parameters** (bvals/bvecs files, DIPY interop).
