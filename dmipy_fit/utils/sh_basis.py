"""The real spherical-harmonic basis and the positivity grid the spherical-deconvolution optimisers use:
dmipy-sim's orthonormal real basis (``dmipy_sim.replay.so3.real_sh``, the ``tournier07`` convention with
``legacy=False``, RPH.md 4.1) and a stored hemisphere, so no shipped code reads dipy at runtime."""
import functools
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


@functools.lru_cache(maxsize=1)
def positivity_hemisphere_neighbours():
    """The neighbours of every vertex of :func:`positivity_hemisphere` on the antipodally symmetric sphere:
    ``(362, k)`` vertex indices, each row padded by repeating its first neighbour. Two hemisphere vertices are
    neighbours when a vertex of one and a vertex or the antipode of the other share a face of the convex hull of the
    724 points, so the adjacency wraps across the equator (dipy's ``HemiSphere`` edges)."""
    from scipy.spatial import ConvexHull
    v = positivity_hemisphere()
    n = len(v)
    faces = ConvexHull(np.concatenate([v, -v])).simplices % n
    nb = [set() for _ in range(n)]
    for f in faces:
        for a in f:
            nb[a].update(int(b) for b in f if b != a)
    k = max(len(s) for s in nb)
    return np.array([sorted(s) + [min(s)] * (k - len(s)) for s in nb], dtype=np.int64)


def fod_peak_values(sh_coeff, sh_order, max_peaks=2, min_separation_angle=25.):
    """The amplitudes of the ``max_peaks`` largest FOD peaks of every voxel, descending, 0 where a voxel has fewer
    peaks: ``(N, max_peaks)`` for ``sh_coeff (N, n_coef)`` (numpy, or a torch tensor, computed on its device and
    returned as numpy).

    The FOD is sampled on :func:`positivity_hemisphere` and clipped at zero; a peak is a vertex at least as large as
    every neighbour (:func:`positivity_hemisphere_neighbours`) and larger than one; peaks within
    ``min_separation_angle`` degrees (axially) of a larger kept peak are dropped, and a peak of value zero is not a
    peak. This is dipy's ``peak_directions`` with ``relative_peak_threshold=0`` on the same hemisphere.
    """
    v = positivity_hemisphere()
    nb = positivity_hemisphere_neighbours()
    far = np.abs(v @ v.T) <= np.cos(np.deg2rad(min_separation_angle))                    # (362, 362)
    if type(sh_coeff).__module__.startswith('torch'):
        import torch
        put = lambda a: torch.as_tensor(a, device=sh_coeff.device)
        B, nb, far, sh = put(real_sh_tournier(sh_order, v)), put(nb), put(far), sh_coeff.double()
        xp = torch
    else:
        B, sh, xp = real_sh_tournier(sh_order, v), np.asarray(sh_coeff, float), np
    amp = xp.clip(B @ sh.T, 0, None)                                                        # (362, N)
    nb_max, nb_min = amp[nb[:, 0]], amp[nb[:, 0]]
    for k in range(1, nb.shape[1]):                                  # one neighbour column at a time, rows contiguous
        nb_max = xp.maximum(nb_max, amp[nb[:, k]])
        nb_min = xp.minimum(nb_min, amp[nb[:, k]])
    vals = xp.where((amp >= nb_max) & (amp > nb_min) & (amp > 0), amp, -1.0)
    cols = xp.arange(amp.shape[1]) if xp is np else torch.arange(amp.shape[1], device=amp.device)
    out = []
    for k in range(max_peaks):
        j = vals.argmax(0)                                            # the largest remaining peak of every voxel
        best = vals[j, cols]
        has = best > 0
        out.append(xp.where(has, best, 0.0))
        vals = xp.where(far[:, j] | ~has[None], vals, -1.0)          # drop it and the peaks within the angle
    out = xp.stack(out, 1)
    return out.cpu().numpy() if xp is not np else out


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
