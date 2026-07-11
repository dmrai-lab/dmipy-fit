"""Pytest session configuration: per-test JAX float64 (x64) policy.

``jax_enable_x64`` is a *process-global* JAX flag, but the suite genuinely needs both settings:

* **float32 (x64 off)** is the production default (GPU speed) and is what most tests — and in
  particular the CSD/OSQP path — expect. jaxopt's OSQP mixes float32/float64 internals under
  x64 and raises a dtype error, and some behavioural thresholds are calibrated for float32.
* **float64 (x64 on)** is required by the parity-vs-numpy tests and by the cylinder/plane
  models, whose Van Gelderen / Bessel sums have alpha^6 terms (~1e48) that overflow float32.

Because x64 is global, a fit that enables it (cylinder models do, in production) leaks into
later work; that made the suite order-dependent (CSD passed alone, failed after a cylinder fit).
Here we set x64 explicitly per test — on for the float64 modules below, off otherwise — and
reset to the float32 baseline afterwards, so the suite is deterministic regardless of order.
(Production robustness for the cylinder->CSD case is additionally handled in
CsdOsqpOptimizer.fit_batch, which forces x64 off around its own solve.)
"""
import pytest
import jax

# Test modules that must run in float64: JAX-vs-numpy parity checks and the cylinder/plane
# models that overflow float32. Everything else runs in the float32 production default.
_FLOAT64_MODULES = {
    "test_phase5_complex_models",   # C2/C3/C4 cylinder JAX forwards (Van Gelderen/Bessel)
    "test_plane_dot_jax",           # P2/P3 plane, parity vs numpy
    "test_spherical_mean_jax",      # spherical-mean fit parity
    "test_vmap_fixed_param_dot",    # vmap fixed-param, parity vs traced
    "test_occupancy_gated_jax",     # occupancy-gated (T2 / surface relaxivity) parity vs numpy
}


@pytest.fixture(autouse=True)
def _jax_x64_policy(request):
    mod = request.node.module.__name__.rsplit(".", 1)[-1] if request.node.module else ""
    want_x64 = mod in _FLOAT64_MODULES
    jax.config.update("jax_enable_x64", want_x64)
    yield
    jax.config.update("jax_enable_x64", False)
