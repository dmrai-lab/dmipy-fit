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

# Apply the GPU memory cap (DMIPY_GPU_MEM_GB) before JAX initialises. JAX is
# imported lazily by the fitting path, so this is effective as long as dmipy_fit
# (or configure()) is imported before the first jax import.
from ._gpu_config import apply_gpu_mem_cap as _apply_gpu_mem_cap, configure  # noqa: E402
_apply_gpu_mem_cap()

# Subpackages are imported explicitly (see the module docstring); the
# intentional top-level names are __version__ and configure.
__all__ = ["__version__", "configure"]
