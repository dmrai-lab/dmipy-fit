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

# Building a multi-compartment model (Ball & Stick)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/02_signal_models/build_multicompartment_model.ipynb)

**Learning objective:** compose basic compartments into a `MultiCompartmentModel`, understand the
parameter-naming convention and volume fractions, then simulate and fit. This is the grammar the
whole model catalog (NODDI, SMT, VERDICT, …) is built from.

```{code-cell} ipython3
# On Colab this installs the engines (public on GitHub); locally it is a no-op.
# Once dmipy is on PyPI:  pip install dmipy
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## Compose Ball + Stick

A `MultiCompartmentModel` is a volume-fraction-weighted sum of its compartments:
$E = f_0\,E_\text{Ball} + f_1\,E_\text{Stick}$, with $f_0 + f_1 = 1$.

```{code-cell} ipython3
import numpy as np
from dmipy_fit.signal_models import cylinder_models, gaussian_models
from dmipy_fit.core.modeling_framework import MultiCompartmentModel
from dmipy_fit.data import saved_acquisition_schemes

acq_scheme = saved_acquisition_schemes.mgh_1010_acquisition_scheme()
ball_and_stick = MultiCompartmentModel(models=[gaussian_models.G1Ball(), cylinder_models.C1Stick()])
print("parameters:", ball_and_stick.parameter_cardinality)
```

The `MultiCompartmentModel` joins the compartments' parameters (each prefixed by its model) and
adds `partial_volume_0`, `partial_volume_1`. **Order matters:** `partial_volume_0` is the first
model listed (the ball here).

## Simulate, then fit

```{code-cell} ipython3
truth = ball_and_stick.parameters_to_parameter_vector(
    G1Ball_1_lambda_iso=3e-9,
    C1Stick_1_lambda_par=1.7e-9,
    C1Stick_1_mu=(np.pi / 2., np.pi / 2.),
    partial_volume_0=0.5,
    partial_volume_1=0.5)
E = ball_and_stick.simulate_signal(acq_scheme, truth)

fit = ball_and_stick.fit(acq_scheme, E)
p = fit.fitted_parameters
print("recovered f_ball =", round(float(p['partial_volume_0']), 3),
      "| lambda_iso =", f"{float(p['G1Ball_1_lambda_iso']):.2e}",
      "| lambda_par =", f"{float(p['C1Stick_1_lambda_par']):.2e}")
```

```{code-cell} ipython3
# The fitted model reproduces the signal, and the scalar parameters match ground truth
# (the stick orientation mu is recovered only up to its antipodal symmetry).
E_fit = ball_and_stick.simulate_signal(acq_scheme, fit.fitted_parameters_vector.ravel())
np.testing.assert_allclose(E_fit, E, atol=1e-3)
np.testing.assert_allclose(float(p['partial_volume_0']), 0.5, atol=0.05)
np.testing.assert_allclose(float(p['G1Ball_1_lambda_iso']), 3e-9, rtol=0.05)
np.testing.assert_allclose(float(p['C1Stick_1_lambda_par']), 1.7e-9, rtol=0.05)
print("fit reproduces the signal and recovers the ground-truth parameters.")
```

## Takeaway

- Compose any compartments with `MultiCompartmentModel([...])`; parameters auto-namespace and
  volume fractions are added (summing to 1).
- The `simulate_signal` → `fit` loop is identical regardless of how many compartments you stack —
  this is the composition grammar behind every model in the catalog.
