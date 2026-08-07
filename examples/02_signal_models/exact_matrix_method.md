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

# Exact matrix-method restricted compartments (P5 / C5 / S5)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/02_signal_models/exact_matrix_method.ipynb)

**Learning objective:** use the exact matrix / multiple-correlation-function (MCF) restricted-diffusion
compartments — `C5CylinderMatrixMethod`, `S5SphereMatrixMethod`, `P5PlaneMatrixMethod` — and see *why*
they exist: they solve the Bloch–Torrey equation exactly for the **actual gradient waveform**, so they stay
correct at high b / strong gradients / short pulses where the Gaussian-phase approximation (GPA) models
`C4`/`S4` (their low-b limit) break down. References: Callaghan (1997), Grebenkov (2007).

```{code-cell} ipython3
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## Agreement at low b, divergence at high b

The matrix model reproduces the Gaussian-phase model where the GPA is valid (low b), then departs as the
GPA breaks down — the matrix model is the exact one, so it attenuates *more*.

```{code-cell} ipython3
import numpy as np
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.cylinder_models import (
    C5CylinderMatrixMethod, C4CylinderGaussianPhaseApproximation)

b = np.array([0.0, 5e8, 1e9, 2e9, 5e9, 1e10])          # s/m^2  (up to 10 000 s/mm^2)
directions = np.tile([0., 0., 1.], (len(b), 1))         # perpendicular to the cylinder axis (x)
scheme = AcquisitionScheme.from_pgse(b, directions, delta=10e-3, Delta=40e-3)

mu, lam_par, diameter = [np.pi / 2, 0.0], 1.7e-9, 8e-6  # axis along x, 8 um cylinder
matrix = C5CylinderMatrixMethod(mu=mu, lambda_par=lam_par, diameter=diameter)(scheme)
gpa    = C4CylinderGaussianPhaseApproximation(mu=mu, lambda_par=lam_par, diameter=diameter)(scheme)

for bb, m, g in zip(b, matrix, gpa):
    print(f"b={bb/1e6:6.0f} s/mm^2 | matrix={m:.4f} | GPA={g:.4f} | diff={abs(m-g):.1e}")
```

At `b ≤ 1000 s/mm²` the two agree to `~2e-4`; by `10 000 s/mm²` the GPA overestimates the signal by `~1 %`
(≈1e-2 absolute) — the matrix model is the exact one there.

```{code-cell} ipython3
low, high = b <= 1e9, b == 1e10
np.testing.assert_allclose(matrix[low], gpa[low], atol=5e-4)   # agree where the GPA is valid
assert abs(matrix[b == 1e10][0] - gpa[b == 1e10][0]) > 1e-3    # real departure at high b
assert matrix[b == 1e10][0] < gpa[b == 1e10][0]                # exact model attenuates more
assert np.all((matrix > 0) & (matrix <= 1 + 1e-9))
print("matrix method: exact where the GPA is valid, and correct where it is not.")
```

## Sphere and plane

`S5SphereMatrixMethod` (isotropic, `diameter` only) and `P5PlaneMatrixMethod` (a 1-D slab; the exact
sibling of `P2`/`P3`) follow the same pattern. All three **consume the stored gradient waveform** `_G`
directly, so they also handle OGSE and other fixed-direction waveforms — not just PGSE.

```{code-cell} ipython3
from dmipy_fit.signal_models.sphere_models import S5SphereMatrixMethod
from dmipy_fit.signal_models.plane_models import P5PlaneMatrixMethod

E_sphere = S5SphereMatrixMethod(diameter=10e-6)(scheme)
E_plane  = P5PlaneMatrixMethod(diameter=10e-6)(scheme)     # gradient taken along the restricting axis
print("sphere E:", np.round(E_sphere, 4))
print("plane  E:", np.round(E_plane, 4))
# smaller pores are more restricted -> higher signal
assert S5SphereMatrixMethod(diameter=4e-6)(scheme)[-1] > S5SphereMatrixMethod(diameter=12e-6)(scheme)[-1]
```

## Accuracy knob and cost

Each model has an `n_modes` argument (Laplacian eigenmodes kept). The default is converged well below
`1e-4` for typical b; raise it for very high q or short pulses. The compartments are drop-in for
`MultiCompartmentModel([...])` like any other, fittable with the standard `simulate_signal` → `fit` loop.

## Takeaway

- `C5`/`S5`/`P5` are the **exact** restricted-diffusion compartments for an arbitrary gradient waveform;
  the Gaussian-phase `C4`/`S4` are their low-b limit.
- Use them when the acquisition pushes into the high-b / short-pulse / OGSE regime where the GPA is no
  longer accurate; use the (cheaper, closed-form) GPA models when it is.
