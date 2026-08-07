"""Loader for the canonical Monte-Carlo *replay-pack* reference dataset (Substrate Commons).

A family of ``.rpk`` packs, one per diameter, for a single restricted shape (cylinder or sphere) at a
fixed intrinsic diffusivity D0. The ``C6MonteCarloReplayCylinder`` / ``S6MonteCarloReplaySphere``
compartment models load a family through here and interpolate the replayed signal across diameter.

Packs are large, so they are NOT bundled in the wheel: point ``dataset_dir`` at a local directory
(populated from the Substrate Commons Hugging Face dataset), or set ``$SUBSTRATE_COMMONS_DATA``.

Forward evaluation uses the compiled-scheme engine (:mod:`dmipy_fit.signal_models._replay_fit`): the
acquisition waveform is projected onto the pack's DCT temporal basis ONCE, after which each replay is a
single matmul — mathematically identical to ``dmipy_sim.bank.ReplayPack.replay`` but fast enough to fit.
``dmipy_sim`` is imported lazily (dmipy-fit stays importable without the simulator installed)."""
import os
import glob
import numpy as np

from ..core.constants import CONSTANTS
from ..signal_models._replay_fit import compile_scheme, replay_complex

_FAMILY_CACHE = {}
_GAMMA = CONSTANTS["water_gyromagnetic_ratio"]


def _data_root(dataset_dir=None):
    return (dataset_dir or os.environ.get("SUBSTRATE_COMMONS_DATA")
            or os.path.join(os.path.dirname(__file__), "mc_replay"))


def _pack_arrays(pack):
    """(dct_coeffs, spin_weights, K, blt_dct-or-None) from a ReplayPack, as float64 host arrays."""
    a = pack.arrays
    C = np.asarray(a["dct_coeffs"], np.float64)
    w = np.asarray(a.get("spin_weights", np.ones(C.shape[0])), np.float64)
    blt = a.get("blt_dct")
    return C, w, C.shape[1], (None if blt is None else np.asarray(blt, np.float64))


class ReplayFamily:
    """A diameter-sorted family of replay packs for one (shape, D0). Signals are interpolated linearly
    across diameter — the restricted signal is smooth in radius, so at the dataset's fine spacing this
    is accurate and gives a differentiable ``diameter`` for fitting.

    The waveform is compiled once (per scheme + rho) and cached on the instance; repeated calls with the
    same scheme (i.e. every voxel/iteration of a fit) reuse it."""

    def __init__(self, shape, diffusivity, diameters_m, packs):
        self.shape = shape
        self.diffusivity = float(diffusivity)
        self.diameters = np.asarray(diameters_m, float)         # ascending, metres
        self.packs = list(packs)
        self._pk = [_pack_arrays(p) for p in self.packs]        # cached host arrays
        p0 = self.packs[0]
        self.n_t = int(p0.n_t)
        self.dt = float(p0.dt)
        self.K = self._pk[0][2]
        self.T_max = float(p0.meta["walk_params"]["T_max"])

    @property
    def diameter_range(self):
        return float(self.diameters[0]), float(self.diameters[-1])

    def _signal_one(self, idx, W, rho_over_D, chi_hat):
        C, w, K, blt = self._pk[idx]
        return replay_complex(C, w, W, blt_dct=blt, rho_over_D=rho_over_D,
                              n_t=self.n_t, chi_hat=chi_hat)

    def replay_interpolated(self, G_pack, diameter, *, rho_over_D=0.0, chi_hat=None):
        """Replay a waveform ``G_pack`` (n_meas, n_t, 3) already resampled onto this family's save grid
        against the two packs bracketing ``diameter`` and linearly interpolate the complex signal.
        ``rho_over_D`` (+ optional ``chi_hat``) activates the exact coherence-gated surface-relaxivity
        replay via each pack's boundary local time. Returns magnitude."""
        W = compile_scheme(G_pack, self.dt, self.K, _GAMMA)     # once per scheme (+rho via chi_hat)
        d = float(np.clip(diameter, self.diameters[0], self.diameters[-1]))
        j = int(np.searchsorted(self.diameters, d))
        if j <= 0:
            S = self._signal_one(0, W, rho_over_D, chi_hat)
        elif j >= len(self.diameters):
            S = self._signal_one(len(self.diameters) - 1, W, rho_over_D, chi_hat)
        else:
            d_lo, d_hi = self.diameters[j - 1], self.diameters[j]
            f = (d - d_lo) / (d_hi - d_lo)
            S = ((1.0 - f) * self._signal_one(j - 1, W, rho_over_D, chi_hat)
                 + f * self._signal_one(j, W, rho_over_D, chi_hat))
        return np.abs(np.asarray(S))


def load_replay_family(shape, diffusivity, *, dataset_dir=None):
    """Load (and cache) the diameter family of packs for ``shape`` ('cylinder'|'sphere') at intrinsic
    diffusivity ``diffusivity`` (m²/s). Returns a :class:`ReplayFamily`.

    Layout searched: ``<root>/canonical/D0-<d.dd>e-9/<shape>/*.rpk``, falling back to any ``*.rpk`` under
    ``<root>/**/<shape>/``. Diameter is read from each pack's ``provenance.diameter_m`` (or the filename)."""
    root = _data_root(dataset_dir)
    key = (os.path.abspath(root), shape, round(float(diffusivity) * 1e9, 3))
    if key in _FAMILY_CACHE:
        return _FAMILY_CACHE[key]
    from dmipy_sim.bank import read_rpk
    subdir = os.path.join(root, "canonical", f"D0-{diffusivity*1e9:.2f}e-9", shape)
    paths = sorted(glob.glob(os.path.join(subdir, "*.rpk")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**", shape, "*.rpk"), recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"no replay packs for shape={shape!r} D0={diffusivity:.2e} under {root!r}; set dataset_dir "
            f"or $SUBSTRATE_COMMONS_DATA to the downloaded Substrate Commons dataset")
    diams, packs = [], []
    for p in paths:
        pk = read_rpk(p)
        prov = pk.meta.get("provenance") or {}
        d = prov.get("diameter_m")
        if d is None:
            d = float(os.path.basename(p).split("d")[-1].split("um")[0]) * 1e-6
        diams.append(float(d)); packs.append(pk)
    order = np.argsort(diams)
    fam = ReplayFamily(shape, diffusivity, np.asarray(diams)[order], [packs[i] for i in order])
    _FAMILY_CACHE[key] = fam
    return fam


def family_from_packs(shape, diffusivity, diameters_m, packs):
    """Build a :class:`ReplayFamily` directly from in-memory ReplayPack objects (tests / custom sets)."""
    order = np.argsort(np.asarray(diameters_m, float))
    return ReplayFamily(shape, diffusivity, np.asarray(diameters_m, float)[order],
                        [packs[i] for i in order])


def resample_waveform_to_grid(G_scheme, dt_scheme, n_t, dt):
    """Resample a scheme waveform ``G_scheme`` (n_m, n_t_s, 3) at ``dt_scheme`` onto a pack save grid
    (``n_t`` points, spacing ``dt``), zero-padding after the acquisition ends (refocused → G→0)."""
    G_scheme = np.asarray(G_scheme, float)
    n_m, n_t_s, _ = G_scheme.shape
    t_s = np.arange(n_t_s) * float(dt_scheme)
    t_p = np.arange(int(n_t)) * float(dt)
    out = np.zeros((n_m, int(n_t), 3), np.float64)
    for m in range(n_m):
        for c in range(3):
            out[m, :, c] = np.interp(t_p, t_s, G_scheme[m, :, c], left=G_scheme[m, 0, c], right=0.0)
    return out


def orient_to_z(mu):
    """Rotation R (3×3) mapping unit axis ``mu`` → +z, with a deterministic perpendicular gauge
    (rows = e1, e2, mu). ``G_canonical(t) = G(t) @ R.T`` expresses a lab waveform in the pack's
    canonical frame (cylinder axis = z), so replay against a z-axis pack yields the mu-oriented signal."""
    mu = np.asarray(mu, float); mu = mu / np.linalg.norm(mu)
    a = np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ mu) * mu; e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    return np.stack([e1, e2, mu], axis=0)
