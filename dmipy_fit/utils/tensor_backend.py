"""The diffusion tensor's FA and principal direction on the device of a backend, for the response estimators:
``backend='torch'`` is :func:`~dmipy_fit.torch.dti_torch.build_dti_fitter_torch` (WLS, S0 fitted), ``backend='jax'``
is :func:`~dmipy_fit.jax.dti_jax.build_dti_fitter` (OLS on the signal normalised by its mean b = 0 signal). The
names are the ones of the batched CSD solvers (``solver='csd_tournier07_torch'`` / ``'csd_tournier07_jax'``)."""
import numpy as np

__all__ = ['BACKENDS', 'check_backend', 'tensor_fa_and_direction']

BACKENDS = ('torch', 'jax')


def check_backend(backend):
    """``backend`` when it is one of :data:`BACKENDS`, else ValueError naming them."""
    if backend not in BACKENDS:
        raise ValueError("backend must be one of {}, got {!r}".format(BACKENDS, backend))
    return backend


def tensor_fa_and_direction(acquisition_scheme, data, *, backend, b_max=None, float64=False):
    """``(fa (N,), direction (N, 3))`` as float64 numpy arrays for ``data (N, N_meas)`` (numpy, or a torch tensor
    for ``backend='torch'``). ``float64`` runs the torch fit in float64 (for a few hundred voxels whose eigenvectors
    enter a response); the JAX fit runs in the dtype JAX is configured for."""
    check_backend(backend)
    if backend == 'torch':
        import torch
        from ..torch.dti_torch import build_dti_fitter_torch
        fit = build_dti_fitter_torch(acquisition_scheme, b_max=b_max,
                                     dtype=torch.float64 if float64 else torch.float32)(data)
        return (fit.fa.double().cpu().numpy(), fit.principal_direction.double().cpu().numpy())
    import jax.numpy as jnp
    from ..jax.dti_jax import build_dti_fitter
    data = np.asarray(data, float)
    s0 = np.maximum(data[:, acquisition_scheme.b0_mask].mean(-1, keepdims=True), 1e-6)
    mu, fa = build_dti_fitter(acquisition_scheme, b_max=b_max)(jnp.asarray(data / s0, jnp.float32))
    theta, phi = np.asarray(mu, float).T
    direction = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
    return np.asarray(fa, float), direction
