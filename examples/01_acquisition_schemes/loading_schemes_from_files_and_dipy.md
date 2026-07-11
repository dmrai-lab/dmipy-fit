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

# Loading acquisition parameters (files, DIPY, Camino)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/01_acquisition_schemes/loading_schemes_from_files_and_dipy.ipynb)

**Learning objective:** build an `AcquisitionScheme` from `.bval`/`.bvec` text files, from a DIPY
`GradientTable`, and from a Camino schemefile — and read back the derived shells, q-values, and
diffusion times.

**Units:** dmipy uses SI — b-values in **s/m²**. `.bval` files are conventionally in s/mm², so
multiply by `1e6`.

```{code-cell} ipython3
# On Colab this installs the engines (public on GitHub); locally it is a no-op.
# Once dmipy is on PyPI:  pip install dmipy
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## From `.bval` / `.bvec` text files

We load the WU-Minn Human Connectome Project acquisition parameters bundled with dmipy-fit.

```{code-cell} ipython3
import os
import numpy as np
import dmipy_fit
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues

acq_path = os.path.join(os.path.dirname(dmipy_fit.__file__), "data", "gradient_tables")

bvalues = np.loadtxt(os.path.join(acq_path, "bvals_hcp_wu_minn.txt"))   # s/mm^2
bvalues_SI = bvalues * 1e6                                             # -> s/m^2
gradient_directions = np.loadtxt(os.path.join(acq_path, "bvecs_hcp_wu_minn.txt"))

delta, Delta = 0.0106, 0.0431                                          # HCP: 10.6 / 43.1 ms
acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)
acq_scheme.print_acquisition_info
```

The scheme automatically separates shells and detects b0s. The derived quantities are all
available as attributes:

```{code-cell} ipython3
print("n measurements :", len(acq_scheme.bvalues))
print("shells (s/m^2) :", np.round(np.unique(acq_scheme.shell_bvalues), 0))
print("q-values (1/m) :", np.round(np.unique(acq_scheme.shell_qvalues), 1)[:4], "...")
print("tau = Δ-δ/3 (s):", round(float(np.unique(acq_scheme.tau)[0]), 4))
```

## When δ and Δ are unknown (clinical data)

Omit `delta`/`Delta` — shell separation still works; only models that need q-values or gradient
strengths will complain (with a clear error).

```{code-cell} ipython3
acq_scheme_no_timing = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions)
acq_scheme_no_timing.print_acquisition_info
```

## From a DIPY `GradientTable`

```{code-cell} ipython3
from dipy.core.gradients import gradient_table
from dmipy_fit.core.acquisition_scheme import gtab_dipy2dmipy

gtab = gradient_table(bvalues, gradient_directions, big_delta=Delta, small_delta=delta)
acq_from_dipy = gtab_dipy2dmipy(gtab)
print("shells match the file-loaded scheme:",
      np.allclose(np.unique(acq_from_dipy.shell_bvalues),
                  np.unique(acq_scheme.shell_bvalues), atol=1e6))
```

(The conversion requires both `big_delta` and `small_delta` on the DIPY table.)

## From / to a Camino schemefile

```{code-cell} ipython3
import tempfile
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_schemefile

acq_from_scheme = acquisition_scheme_from_schemefile(
    os.path.join(acq_path, "schemefile_hcp_wu_minn.scheme1"))
print("loaded from schemefile:", len(acq_from_scheme.bvalues), "measurements")

# round-trip: write to a temp schemefile and reload
out = os.path.join(tempfile.mkdtemp(), "my_scheme.scheme1")
acq_scheme.to_schemefile(out)
print("wrote:", out)
```

## Summary

- b-values in files are s/mm² — multiply by `1e6` for dmipy's SI (s/m²).
- Build from `acquisition_scheme_from_bvalues` (files), `gtab_dipy2dmipy` (DIPY), or
  `acquisition_scheme_from_schemefile` (Camino); read back shells / q-values / `tau` as attributes.
- δ/Δ are optional; models that need them raise a clear error if missing.
