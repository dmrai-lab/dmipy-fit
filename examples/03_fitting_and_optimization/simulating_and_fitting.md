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

# Simulating and fitting: the core workflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/03_fitting_and_optimization/simulating_and_fitting.ipynb)

**Learning objective:** the end-to-end dmipy loop — build a model, simulate a signal from known
parameters, then fit the model back to recover them. The same three-line pattern (model →
`simulate_signal` → `fit`) works for *every* model in the catalog.

```{code-cell} ipython3
# On Colab this installs the engines (public on GitHub); locally it is a no-op.
# Once dmipy is on PyPI:  pip install dmipy
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## 1. A model and an acquisition scheme

We use a bundled multi-shell scheme (MGH-USC Connectom) and the simplest microstructure model —
a single `C1Stick` (a zero-radius cylinder: intra-axonal diffusion along one direction).

```{code-cell} ipython3
import numpy as np
from dmipy_fit.signal_models import cylinder_models
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data import saved_acquisition_schemes

acq_scheme = saved_acquisition_schemes.mgh_1010_acquisition_scheme()
stick_model = MultiCompartmentModel(models=[cylinder_models.C1Stick()])
print("parameters:", stick_model.parameter_cardinality)
```

The name `C1Stick_1_mu` reads as *model C1Stick, instance 1, parameter mu*. `mu` is the
orientation `(theta, phi)` on the sphere (cardinality 2); `lambda_par` is the parallel
diffusivity (cardinality 1).

## 2. Simulate a signal from known parameters

`parameters_to_parameter_vector` packs named parameters into the vector the model expects.

```{code-cell} ipython3
mu = (np.pi / 2., np.pi / 2.)      # orientation in radians
lambda_par = 1.7e-9                # m^2/s
truth = stick_model.parameters_to_parameter_vector(
    C1Stick_1_mu=mu, C1Stick_1_lambda_par=lambda_par)

E = stick_model.simulate_signal(acq_scheme, truth)
print("simulated signal:", E.shape, "attenuations in", (round(float(E.min()), 3), round(float(E.max()), 3)))
```

## 3. Fit the model back

`model.fit(scheme, data)` is the whole inversion. The default `brute2fine` optimizer does a
global brute-force search then a local refine — no GPU needed.

```{code-cell} ipython3
res = stick_model.fit(acq_scheme, E)
fitted = res.fitted_parameters_vector.ravel()
print("recovered:", np.round(fitted, 4))
print("truth:    ", np.round(truth, 4))

# A stick is antipodally symmetric: mu and -mu describe the same axis, so the orientation
# may come back "flipped" (e.g. phi -> phi ± pi). The physically meaningful check is that the
# recovered model reproduces the signal, and that lambda_par matches.
E_fit = stick_model.simulate_signal(acq_scheme, fitted)
np.testing.assert_allclose(E_fit, E, atol=1e-4)
np.testing.assert_allclose(fitted[2], lambda_par, rtol=0.02)
print("\nfitted model reproduces the signal (max |ΔE| = "
      f"{float(np.max(np.abs(E_fit - E))):.2e}); lambda_par recovered.")
```

## Where to go next

- The *same* pattern fits any model: swap `C1Stick()` for a `MultiCompartmentModel` of
  ball + zeppelin + stick (see the [model catalog](https://dmipy.org)).
- For a whole brain in one vectorised GPU call, pass `solver="jax"` to `.fit`.
- Add Rician noise before fitting to test robustness; fit with `Nsamples`/noise-aware options.
