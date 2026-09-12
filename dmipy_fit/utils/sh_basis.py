"""The real spherical-harmonic basis and the positivity grid the spherical-deconvolution optimisers use:
dmipy-sim's orthonormal real basis (``dmipy_sim.replay.so3.real_sh``, the ``tournier07`` convention with
``legacy=False``, RPH.md 4.1) and a stored hemisphere, so no shipped code reads dipy at runtime."""
import importlib
import os

import numpy as np

_SPHERES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "spheres")


def sh_degrees(sh_order):
    """The degree ``l`` of every coefficient of the even, symmetric basis up to ``sh_order``: ``(n_coef,)``."""
    return np.concatenate([np.full(2 * l + 1, l) for l in range(0, int(sh_order) + 1, 2)])


def real_sh_tournier(sh_order, dirs):
    """The basis sampled at unit ``dirs`` ``(n, 3)``: ``(n, n_coef)``, orthonormal, ``tournier07`` order."""
    from dmipy_sim.replay import so3
    return np.asarray(so3.real_sh(int(sh_order), np.asarray(dirs, np.float64)), np.float64)


def positivity_hemisphere():
    """The 362 unit directions the non-negativity constraint is imposed on: one of each antipodal pair of the
    724-vertex symmetric sphere (``symmetric724`` of Garyfallidis et al., dipy, BSD), stored with the package."""
    z = np.load(os.path.join(_SPHERES, "symmetric724_hemisphere.npz"))
    return np.asarray(z["vertices"], np.float64)


def positivity_basis(sh_order):
    """The basis on the positivity hemisphere: ``(362, n_coef)``."""
    return real_sh_tournier(sh_order, positivity_hemisphere())


class _Missing:
    """Stands in for an optional package that is not installed: any attribute access says so, naming it."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportError(f"{self._name} is required here but is not installed: pip install {self._name}")

    def __bool__(self):
        return False


def optional_module(name):
    """``(module, True)`` when ``name`` imports, else ``(a stand-in that raises ImportError on use, False)``."""
    try:
        return importlib.import_module(name), True
    except ImportError:
        return _Missing(name), False
