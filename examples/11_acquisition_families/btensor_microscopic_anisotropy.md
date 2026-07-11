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

# b-tensor encoding: microscopic anisotropy (LTE vs STE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/11_acquisition_families/btensor_microscopic_anisotropy.ipynb)

A single linear (PGSE) shell cannot separate **microscopic anisotropy** (anisotropic
compartments) from **orientation dispersion** (anisotropic compartments pointing every which
way): both flatten the powder-averaged signal. **Tensor-valued encoding** breaks the degeneracy.
The b-matrix becomes a *tensor* with a shape $b_\Delta$:

- **LTE** (linear, $b_\Delta=1$) — ordinary PGSE; sensitive to orientation.
- **STE** (spherical, $b_\Delta=0$) — encodes isotropically, so it is **blind to orientation** and
  sees only the *mean* diffusivity.

dmipy carries the full b-tensor per measurement, so LTE and STE compose into one model. This
example is fully synthetic (no data download): we simulate a microscopically-anisotropic Gaussian,
show the LTE↔STE gap, and fit back its $(\lambda_\parallel,\lambda_\perp)$.

```{code-cell} ipython3
# On Colab this installs the engines (public on GitHub); locally it is a no-op.
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## A microscopically-anisotropic compartment

`G2Zeppelin` is an axially-symmetric Gaussian with $\lambda_\parallel \neq \lambda_\perp$ — a
single micro-anisotropic "pore". Its isotropic mean is $\bar D=(\lambda_\parallel+2\lambda_\perp)/3$.

```{code-cell} ipython3
import numpy as np
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.gaussian_models import G2Zeppelin

LPAR, LPERP = 1.7e-9, 0.5e-9                 # ground-truth micro-diffusivities (m^2/s)
Dbar = (LPAR + 2 * LPERP) / 3
zep = G2Zeppelin(mu=[0., 0.], lambda_par=LPAR, lambda_perp=LPERP)
shells = np.array([1e9, 2e9, 3e9])           # s/m^2  (1e9 = 1000 s/mm^2)

# A FIXED Fibonacci set of directions for the powder average (deterministic, so the fit below
# sees a smooth objective rather than a re-randomised one each call).
_N = 60; _i = np.arange(_N) + 0.5
_phi = np.arccos(1 - 2 * _i / _N); _th = np.pi * (1 + 5 ** 0.5) * _i
DIRS = np.c_[np.sin(_phi) * np.cos(_th), np.sin(_phi) * np.sin(_th), np.cos(_phi)]

def lte_powder(model, b):
    """Powder-averaged linear-encoding signal at b (mean over the fixed directions)."""
    sch = AcquisitionScheme.from_pgse(np.r_[0., np.full(_N, b)],
                                      np.vstack([[0, 0, 1.], DIRS]), delta=0.012, Delta=0.030)
    return float(model(sch)[1:].mean())

def ste(model, b):
    """Spherical (isotropic) encoding signal at b — orientation-blind."""
    sch = AcquisitionScheme.from_btensor_ste(np.array([0., b]), delta=0.012, Delta=0.030)
    return float(np.asarray(model(sch)).reshape(-1)[-1])
```

## The LTE↔STE gap *is* microscopic anisotropy

```{code-cell} ipython3
E_lte = np.array([lte_powder(zep, b) for b in shells])
E_ste = np.array([ste(zep, b) for b in shells])
for b, l, s in zip(shells, E_lte, E_ste):
    print(f"b={b/1e9:.0f}e9  LTE_powder={l:.4f}   STE={s:.4f}   gap={l - s:+.4f}")
print(f"\nSTE tracks exp(-b*Dbar) exactly (isotropic mean); LTE decays slower — "
      f"the powder gap is the micro-anisotropy signature.")
assert np.all(E_lte > E_ste)                 # LTE always above STE for a micro-anisotropic pore
```

## Fit: STE pins the mean, LTE pins the variance

STE alone fixes $\bar D$ but is blind to anisotropy; the LTE gap supplies the missing variance.
Fitting the *joint* LTE+STE signal recovers $(\lambda_\parallel,\lambda_\perp)$ — which a single
LTE shell cannot (dispersion would mimic it).

```{code-cell} ipython3
from scipy.optimize import least_squares

# "measured" = truth + a little noise
meas = np.concatenate([E_lte, E_ste]) + 0.003 * np.random.default_rng(1).standard_normal(2 * len(shells))

def residual(x):
    lp, lperp = x
    m = G2Zeppelin(mu=[0., 0.], lambda_par=lp, lambda_perp=lperp)
    pred = np.concatenate([[lte_powder(m, b) for b in shells],
                           [ste(m, b) for b in shells]])
    return pred - meas

sol = least_squares(residual, x0=[1.0e-9, 1.0e-9],
                    bounds=([0.1e-9, 0.1e-9], [3e-9, 3e-9]))
print(f"recovered lambda_par  = {sol.x[0]:.2e}  (truth {LPAR:.2e})")
print(f"recovered lambda_perp = {sol.x[1]:.2e}  (truth {LPERP:.2e})")
np.testing.assert_allclose(sol.x, [LPAR, LPERP], rtol=0.2)
```

## Takeaway

- **STE is orientation-blind**; the powder-averaged **LTE↔STE gap is microscopic anisotropy**
  ($\mu$FA), which a single PGSE shell cannot separate from orientation dispersion.
- dmipy consumes the **b-tensor per measurement**, so `from_pgse` (LTE), `from_btensor_pte`
  (PTE) and `from_btensor_ste` (STE) compose into one scheme/model. The joint fit recovers the
  microscopic $(\lambda_\parallel,\lambda_\perp)$.
- Point dmipy at any real b-tensor dataset with the same recipe (e.g. Szczepankiewicz et al.,
  *Data in Brief* 2019): load per-shell b-values and b-tensor shapes, build the scheme, fit.
