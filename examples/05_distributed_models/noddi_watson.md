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

# NODDI-Watson: orientation dispersion + neurite density

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/05_distributed_models/noddi_watson.ipynb)

NODDI (Zhang et al. 2012) estimates neurite **orientation dispersion** (ODI) and **density**
from a multi-shell acquisition. It is a `MultiCompartmentModel` built from primitives you have
already met, plus one new idea — a **distribution** wrapping a bundle:

- a **Watson-dispersed** stick+zeppelin bundle (`SD1WatsonDistributed`) — intra-axonal `C1Stick`
  and extra-axonal `G2Zeppelin`, dispersed on the sphere with concentration → ODI,
- an isotropic **CSF** `G1Ball`,
- the **tortuosity** and **equal-parallel-diffusivity** constraints that make it identifiable.

```{code-cell} ipython3
# On Colab this installs the engines (public on GitHub); locally it is a no-op.
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## Build the NODDI-Watson model

```{code-cell} ipython3
import numpy as np
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_fit.signal_models.cylinder_models import C1Stick
from dmipy_fit.distributions.distribute_models import SD1WatsonDistributed
from dmipy_fit.core.modeling_framework import MultiCompartmentModel

bundle = SD1WatsonDistributed(models=[C1Stick(), G2Zeppelin()])
# tortuosity: extra-axonal lambda_perp is tied to lambda_par and the intra fraction
bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

noddi = MultiCompartmentModel([G1Ball(), bundle])
noddi.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)          # CSF
print("free parameters:", list(noddi.parameter_cardinality))
```

The free parameters are the fibre orientation `mu`, the dispersion `odi`, the intra-axonal
fraction within the bundle (`SD1WatsonDistributed_1_partial_volume_0`), and the CSF / bundle
volume fractions.

## Simulate a voxel and fit it back

```{code-cell} ipython3
rng   = np.random.default_rng(0)
bvals = np.concatenate([[0.], np.full(60, 1e9), np.full(60, 2e9)])   # HCP-like 2-shell HARDI
d = rng.standard_normal((120, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
scheme = acquisition_scheme_from_bvalues(bvals, np.vstack([[0, 0, 1.], d]),
                                         delta=0.0106, Delta=0.0431)

truth = noddi.parameters_to_parameter_vector(
    SD1WatsonDistributed_1_SD1Watson_1_mu=[np.pi / 2, 0.],
    SD1WatsonDistributed_1_SD1Watson_1_odi=0.3,     # orientation dispersion index
    SD1WatsonDistributed_1_partial_volume_0=0.6,    # intra-axonal fraction (bundle)
    partial_volume_0=0.1,                           # CSF fraction
    partial_volume_1=0.9)
E = noddi.simulate_signal(scheme, truth)

fit = noddi.fit(scheme, E)
sc = lambda a: float(np.asarray(a).reshape(-1)[0])
odi   = sc(fit.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi'])
f_in  = sc(fit.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'])
f_csf = sc(fit.fitted_parameters['partial_volume_0'])
print(f"ODI          {odi:.3f}  (truth 0.300)")
print(f"intra frac   {f_in:.3f}  (truth 0.600)")
print(f"CSF frac     {f_csf:.3f}  (truth 0.100)")
np.testing.assert_allclose([odi, f_in, f_csf], [0.3, 0.6, 0.1], atol=0.05)
```

## On real data

The recipe is identical for a real volume — pass a 4-D array and a brain mask:

```python
fit = noddi.fit(scheme, data, mask=data[..., 0] > 0)   # fits every masked voxel
odi_map = fit.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
```

For a whole brain, `solver="jax"` fits every voxel in one vectorised GPU call (see the
[GPU-fitting tutorial](../06_gpu_acceleration/jax_gpu_fitting.ipynb)).
