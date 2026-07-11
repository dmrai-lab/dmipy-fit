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

# IVIM: separating perfusion from diffusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/03_fitting_and_optimization/ivim.ipynb)

Intravoxel incoherent motion (IVIM; Le Bihan 1988) models the DWI signal as **two isotropic
pools**: fast "pseudo-diffusion" from blood micro-circulation (perfusion fraction $f$, coefficient
$D^*$) and ordinary tissue diffusion ($D$):

$$S/S_0 = f\,e^{-b D^*} + (1-f)\,e^{-b D},\qquad D^* \gg D.$$

Because $D^*$ is large, the perfusion term decays away by $b \sim 200$ s/mm² — so **dense low-b
sampling** is what makes $f$ and $D^*$ estimable. In dmipy it is just two `G1Ball`s.

```{code-cell} ipython3
# On Colab this installs the engines (public on GitHub); locally it is a no-op.
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## Build the IVIM model (fixed $D^*$)

The common "$D^*$-fixed" IVIM: fix the blood pseudo-diffusivity (here $7\times10^{-9}$ m²/s,
after Gurney-Champion 2016) and bound the tissue $D$, which stabilises the fit.

```{code-cell} ipython3
import numpy as np
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.modeling_framework import MultiCompartmentModel

# b-values in s/m^2 (1e6 = 1 s/mm^2), dense at low b to catch the perfusion decay
bvals = np.array([0, 10, 20, 40, 60, 80, 100, 150, 200, 400, 600, 800], float) * 1e6
bvecs = np.tile([0, 0, 1.], (len(bvals), 1))
scheme = acquisition_scheme_from_bvalues(bvals, bvecs)

ivim = MultiCompartmentModel([G1Ball(), G1Ball()])   # ball 0 = tissue D, ball 1 = blood D*
ivim.set_fixed_parameter('G1Ball_2_lambda_iso', 7e-9)               # fix D*
ivim.set_parameter_optimization_bounds('G1Ball_1_lambda_iso', [0.5e-9, 3e-9])
print("free parameters:", [n for n, c in ivim.parameter_cardinality.items()])
```

## Simulate a voxel and fit it back

```{code-cell} ipython3
F_TRUE, D_TRUE = 0.15, 1.0e-9                        # perfusion fraction, tissue diffusivity
truth = ivim.parameters_to_parameter_vector(
    G1Ball_1_lambda_iso=D_TRUE, partial_volume_0=1 - F_TRUE, partial_volume_1=F_TRUE)
E = ivim.simulate_signal(scheme, truth)

fit = ivim.fit(scheme, E)
sc = lambda a: float(np.asarray(a).reshape(-1)[0])
D_fit = sc(fit.fitted_parameters['G1Ball_1_lambda_iso'])
f_fit = sc(fit.fitted_parameters['partial_volume_1'])
print(f"tissue D          : {D_fit:.2e}  (truth {D_TRUE:.2e})")
print(f"perfusion fraction: {f_fit:.3f}   (truth {F_TRUE:.3f})")
np.testing.assert_allclose(D_fit, D_TRUE, rtol=0.1)
np.testing.assert_allclose(f_fit, F_TRUE, atol=0.05)
```

## Notes

- **Tissue $D$** is recovered tightly; the **perfusion fraction $f$** is the hard parameter —
  it rides on the few low-b points, so it is sensitive to SNR and b-sampling. This is why IVIM
  acquisitions pack many low-b shells.
- "$D^*$-free" IVIM (let $D^*$ vary too) is the same model without the `set_fixed_parameter` — but
  $D^*$ is poorly conditioned and often needs a two-step or segmented fit.
- Everything else is the standard loop: build the model, `simulate_signal`, `fit`.
