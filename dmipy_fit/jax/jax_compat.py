"""JAX compatibility utilities for dmipy.

Provides thin wrappers so that downstream modules can use JAX without
hard-depending on it at import time.
"""

import numpy as np

try:
    import jax
    # Use float32 by default: on GPU, float32 throughput is ~100x float64, so
    # float32 is the standard unless the caller explicitly needs float64.
    # jax_enable_x64 is intentionally NOT set here; callers that need float64
    # can set jax.config.update("jax_enable_x64", True) themselves.
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    jax_available = False
    jnp = None


def _require_jax():
    if not jax_available:
        raise ImportError(
            "JAX is required for this functionality. Install it with:\n"
            "  pip install 'jax[cpu]>=0.4.20' jaxopt>=0.8"
        )


def scheme_to_jax(acquisition_scheme):
    """Extract acquisition scheme arrays as JAX device arrays.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme

    Returns
    -------
    dict with keys: bvalues, gradient_directions, and optionally
    qvalues, delta, Delta, tau — all as jnp arrays.
    """
    _require_jax()
    s = acquisition_scheme
    out = {
        'bvalues': jnp.array(s.bvalues),
        'gradient_directions': jnp.array(s.gradient_directions),
    }
    if s.qvalues is not None:
        out['qvalues'] = jnp.array(s.qvalues)
    if s.delta is not None:
        out['delta'] = jnp.array(s.delta)
    if s.Delta is not None:
        out['Delta'] = jnp.array(s.Delta)
    if s.tau is not None:
        out['tau'] = jnp.array(s.tau)
    if hasattr(s, 'gradient_strengths') and s.gradient_strengths is not None:
        out['gradient_strengths'] = jnp.array(s.gradient_strengths)
    if s.TE is not None:
        out['TE'] = jnp.array(s.TE)
        # transverse occupancy time for the T2 / surface-relaxivity factors
        # (reuse the validated numpy gating): scheme.tau_perp when set (STE encoding
        # = 2*delta), else TE (spin echo, where the whole echo is transverse).
        from ..signal_models.attenuation import _tau_perp
        out['tau_perp'] = jnp.array(_tau_perp(s))
    # longitudinal (storage) occupancy time for the T1 factor: TM during a
    # stimulated-echo mixing time, else None (plain spin echo -> identity factor).
    from ..signal_models.attenuation import _tau_par
    _tau_par_val = _tau_par(s)
    if _tau_par_val is not None:
        out['tau_par'] = jnp.array(_tau_par_val)
    # Waveform fields (AcquisitionScheme only, not PGSEAcquisitionScheme)
    if hasattr(s, '_G') and s._G is not None:
        out['G_waveform'] = jnp.array(s._G)  # (n_m, n_t, 3)
        out['dt'] = float(s._dt)
    return out


def unitsphere2cart_1d_jax(mu):
    """Convert 1-D unit-sphere coordinates (theta, phi) to Cartesian (x, y, z).

    JAX equivalent of dmipy.utils.utils.unitsphere2cart_1d.

    Parameters
    ----------
    mu : array-like of shape (2,)

    Returns
    -------
    jnp.array of shape (3,)
    """
    _require_jax()
    theta, phi = mu[0], mu[1]
    sintheta = jnp.sin(theta)
    return jnp.stack([
        sintheta * jnp.cos(phi),
        sintheta * jnp.sin(phi),
        jnp.cos(theta),
    ])
