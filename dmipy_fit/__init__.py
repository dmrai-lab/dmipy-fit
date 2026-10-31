"""dmipy-fit — analytical diffusion-MRI signal models, fitting and tissue response.

The package does not re-export a flat API; import the subpackage you need, e.g.::

    from dmipy_fit.signal_models import ...
    from dmipy_fit.core.modeling_framework import MultiCompartmentModel

Public subpackages: ``core``, ``signal_models``, ``distributions``, ``optimizers``,
``optimizers_fod``, ``custom_optimizers``, ``algorithms``, ``tissue_response``,
``data``, ``utils``, ``jax``.

Analytical diffusion-MRI signal models + fitting; consumes the shared free-waveform
``dmipy_sim`` sequence/substrate interface (fit → sim, one-directional).
"""
from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version("dmipy-fit")
except _PackageNotFoundError:
    __version__ = "unknown"

# The GPU memory cap (DMIPY_GPU_MEM_GB) and its Python spelling ``dmipy_sim.configure`` live in dmipy-sim, which
# applies them when it is imported, before JAX initialises; ``dmipy_fit.jax`` imports it first.
__all__ = ["__version__"]


def __getattr__(name):
    if name == "configure":
        raise AttributeError("dmipy_fit.configure is removed: the GPU memory cap is dmipy_sim.configure(max_gpu_gb=...)")
    raise AttributeError(f"module 'dmipy_fit' has no attribute {name!r}")
