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

# GPU fitting with JAX

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-fit/blob/main/examples/06_gpu_acceleration/jax_gpu_fitting.ipynb)

**Learning objective:** use dmipy's JAX fitting path — `model.fit(..., solver="jax")` with a
pluggable loss, and `build_vmap_fitter` to fit a whole batch of voxels in one vectorised
(GPU-able) call. On a GPU runtime (Colab: *Runtime → Change runtime type → GPU*) the batch fit is
a single kernel; on CPU it still works, just slower. The first call JIT-compiles (~tens of s).

```{code-cell} ipython3
# On Colab this installs the engines (public on GitHub); locally it is a no-op.
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## Scheme, model, and a noisy signal

```{code-cell} ipython3
import numpy as np
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_fit.core.modeling_framework import MultiCompartmentModel

rng   = np.random.default_rng(42)
bvals = np.r_[0.0, np.full(25, 1e9), np.full(25, 2e9)]
bvecs = np.zeros((51, 3)); v = rng.standard_normal((50, 3))
bvecs[1:] = v / np.linalg.norm(v, axis=1, keepdims=True)
scheme = acquisition_scheme_from_bvalues(bvals, bvecs, delta=0.010, Delta=0.030)

model = MultiCompartmentModel([G1Ball()])
TRUE, SIGMA = 2.0e-9, 0.03
E = model(scheme, G1Ball_1_lambda_iso=TRUE)
E_noisy = np.abs(E + SIGMA * (rng.standard_normal(E.shape) + 1j * rng.standard_normal(E.shape)))
```

## Single-voxel fit with a JAX loss

`loss_fn` is pluggable — MSE, Rician NLL, or non-central-χ (parallel imaging). Here Rician, which
is the correct magnitude-MRI likelihood.

```{code-cell} ipython3
from dmipy_fit.jax.losses_jax import rician_nll

fit = model.fit(scheme, E_noisy, solver="jax", loss_fn=rician_nll(sigma=SIGMA))
lam = float(np.asarray(fit.fitted_parameters['G1Ball_1_lambda_iso']).reshape(-1)[0])
print(f"fitted lambda_iso = {lam:.3e}  (truth {TRUE:.3e}, err {abs(lam-TRUE)/TRUE*100:.1f}%)")
```

## Batch fit: `build_vmap_fitter` (one call over many voxels)

This is the GPU path — the optimiser is `vmap`-ed across voxels and JIT-compiled once, then the
whole batch is one kernel. On CPU it still avoids the Python per-voxel loop.

```{code-cell} ipython3
import jax.numpy as jnp
from dmipy_fit.jax.optimizers_jax import JaxOptimizer
from dmipy_fit.jax.vmap_fit import build_vmap_fitter

jax_opt   = JaxOptimizer(model, scheme, loss_fn=rician_nll(sigma=SIGMA),
                         maxiter=200, Ns=5, N_sphere_samples=15)
fit_batch = build_vmap_fitter(jax_opt, dtype=jnp.float32)

true_lams = np.array([1.5e-9, 2.0e-9, 2.5e-9, 3.0e-9])
data = np.stack([np.abs(model(scheme, G1Ball_1_lambda_iso=l)
                        + SIGMA * (rng.standard_normal(E.shape) + 1j*rng.standard_normal(E.shape)))
                 for l in true_lams])
x0 = np.tile(model.parameters_to_parameter_vector(G1Ball_1_lambda_iso=2e-9), (len(true_lams), 1))
out = np.asarray(fit_batch(jnp.asarray(data), jnp.asarray(x0)))
print("recovered lambda_iso:", np.round(out.ravel(), 3) * 1e9, "x1e-9")
print("truth             :", np.round(true_lams, 3) * 1e9, "x1e-9")
```

## When to use which

- **< ~100 voxels or a quick look:** `brute2fine` (the scipy default) — no JIT overhead.
- **Many voxels / a whole brain, or Rician/χ likelihoods, or a GPU:** `solver="jax"` /
  `build_vmap_fitter` — one vectorised kernel, autodiff gradients, GPU-accelerated.

The first JAX call pays a one-time JIT compile; subsequent fits reuse it.
