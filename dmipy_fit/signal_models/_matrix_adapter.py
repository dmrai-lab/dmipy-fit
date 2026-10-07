"""Glue between C5/S5/P5 (:mod:`cylinder_models`, :mod:`sphere_models`, :mod:`plane_models`) and the
shared eigenmode solver :mod:`dmipy_sim.math.matrix_method`: the model-side concerns sim's module has no
reason to carry -- a model's single ``n_modes`` knob, and the acquisition-scheme glue (a scalar-timing
PGSE waveform reconstructed from ``gradient_strengths``/``delta``/``Delta``, a waveform projected onto its
dominant direction for the plane and for a rotating encoding's fixed-direction approximation).
"""
import numpy as np

from dmipy_sim.math.matrix_method import MatrixPore, matrix_restricted_batch, matrix_restricted_signal

__all__ = [
    "modes_for", "project_fixed_direction", "pgse_waveform",
    "matrix_signal", "matrix_signal_batch",
]


def modes_for(shape, n_modes):
    """``dmipy_sim.math.matrix_method``'s ``n_modes`` for a model's single integer knob: the count of cosine
    modes itself for the plane (one mode index), and the ``(angular, radial)`` pair ``(n, max(4, n // 2 + 2))``
    for the cylinder and the sphere."""
    n = int(n_modes)
    return n if shape == "plane" else (n, max(4, n // 2 + 2))


def project_fixed_direction(G_m, dt):
    """The fixed gradient direction ``d`` (the unit direction at peak |G|) and the signed 1-D magnitude
    schedule ``g(t) = G_m . d`` for a single measurement's waveform ``G_m`` of shape (n_t, 3). Exact for
    fixed-direction waveforms (PGSE, standard OGSE); a rotating/b-tensor waveform is projected onto its
    dominant direction (an approximation for those, flagged in the model docstrings)."""
    G_m = np.asarray(G_m, dtype=np.float64)
    mag = np.linalg.norm(G_m, axis=1)
    k = int(np.argmax(mag))
    if mag[k] == 0.0:
        return np.array([1.0, 0.0, 0.0]), np.zeros(len(G_m))
    d = G_m[k] / mag[k]
    return d, G_m @ d


def pgse_waveform(gradient_strength, delta, Delta, direction, n_t=1024):
    """A rectangular bipolar PGSE waveform (n_t, 3) + dt reconstructed from scalar timing, for a scheme
    that carries only (gradient_strength, delta, Delta) and no stored ``_G``."""
    dt = (Delta + delta) / n_t
    nd = max(1, int(round(delta / dt)))
    ng = int(round(Delta / dt))
    G = np.zeros((n_t, 3))
    u = np.asarray(direction, dtype=np.float64)
    G[:nd] = gradient_strength * u
    G[ng:ng + nd] = -gradient_strength * u
    return G, dt


def matrix_signal(shape, g_axis, dt, D, size_m, n_modes):
    """``matrix_restricted_signal`` for a model's single ``n_modes`` knob (mapped through
    :func:`modes_for`); ``size_m`` is sim's own convention (the plate separation for the plane, the
    DIAMETER for the cylinder and the sphere)."""
    return matrix_restricted_signal(shape, g_axis, dt, D, size_m, n_modes=modes_for(shape, n_modes))


def matrix_signal_batch(shape, g_axes, dt, D, size_m, n_modes, *, use_jax=False):
    """``matrix_restricted_batch`` for a model's single ``n_modes`` knob, routed to sim's NumPy
    exact/Strang path or (``use_jax=True``) the differentiable JAX batch twin, built from the SAME pore's
    (size-scaled) eigenvalues and position-operator eigendecomposition (:class:`MatrixPore`'s public
    ``lam``/``B``; the decomposition itself done once here, per pore)."""
    pair = modes_for(shape, n_modes)
    if not use_jax:
        return matrix_restricted_batch(shape, g_axes, dt, D, size_m, n_modes=pair)
    import jax.numpy as jnp

    from ..jax.signal_models_jax import matrix_restricted_signal_jax_batch
    pore = MatrixPore(shape, size_m, D, n_modes=pair)
    beta, U = np.linalg.eigh(pore.B)
    out = matrix_restricted_signal_jax_batch(
        jnp.asarray(g_axes), float(dt), float(D),
        jnp.asarray(pore.lam), jnp.asarray(beta), jnp.asarray(U))
    return np.asarray(out, dtype=float)
