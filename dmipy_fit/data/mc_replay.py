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


def _pack_arrays(pack, axes=None):
    """(position_coeffs, spin_weights, K, blt_dct-or-None) from a ReplayPack, float64 host arrays.

    ``K`` is the number of SINE BANDS, i.e. the stored width minus the two endpoint
    coefficients. Returning the width instead would hand ``K + 2`` to compile_scheme, whose
    output would then be the right shape to multiply and the wrong thing to multiply by.
    """
    from dmipy_sim.compression import read_position_coeffs
    a = pack.arrays
    C = read_position_coeffs(a, axes=axes, dtype=np.float64)
    w = np.asarray(a.get("spin_weights", np.ones(C.shape[0])), np.float64)
    blt = a.get("blt_dct")
    return C, w, C.shape[1] - 2, (None if blt is None else np.asarray(blt, np.float64))


DEFAULT_REPO = "SubstrateCommons/canonical-pores"


class _LazyPacks:
    """Sequence of packs materialised on first access.

    A family spans 200 diameters and tens of GiB; a fit touches the two packs bracketing the current
    diameter. Loading the family eagerly would read every pack (~8 GiB resident locally, and a full
    dataset download from the Hub) to answer a question about two of them.
    """

    def __init__(self, loaders):
        self._loaders = list(loaders)
        self._cache = {}

    def __len__(self):
        return len(self._loaders)

    def __getitem__(self, i):
        i = int(i)
        if i not in self._cache:
            self._cache[i] = self._loaders[i]()
        return self._cache[i]


def _hf_manifest(repo_id, revision=None):
    from huggingface_hub import hf_hub_download
    import json
    p = hf_hub_download(repo_id=repo_id, filename="manifest.json", repo_type="dataset",
                        revision=revision)
    return json.load(open(p))


def _hf_family(shape, diffusivity, repo_id, revision=None, eps=None, axes=None):
    """Diameters + per-pack codec from the manifest ALONE; packs fetched on demand, one file each.

    The manifest carries n_t and K per substrate, so the family is fully described without downloading
    a single pack -- which is what makes X6 usable straight from the Hub instead of after a 24 GiB pull.
    """
    from huggingface_hub import hf_hub_download
    man = _hf_manifest(repo_id, revision)
    rows = [r for r in man["substrates"] if r["shape"] == shape]
    if not rows:
        raise FileNotFoundError(f"{repo_id}: no substrates of shape {shape!r} in manifest.json")
    rows.sort(key=lambda r: r["d_um"])
    diams = np.asarray([r["d_um"] * 1e-6 for r in rows], float)
    n_t = np.asarray([int(r["n_t"]) for r in rows], int)
    K = np.asarray([int(r["K"]) for r in rows], int)
    T_max = float(man.get("T_max", 0.2))
    dt = T_max / (n_t - 1)

    def mk(row):
        path = row["path"]
        # the manifest records the MEASURED prefix length for each floor, so the tier is known without
        # touching the pack; None means "whole file" (hf_hub_download, which also caches on disk).
        tier = None
        if eps is not None:
            tiers = row.get("prefix_tiers") or {}
            tier = tiers.get(f"{eps:g}") or tiers.get(str(eps))
            if tier is None:
                raise ValueError(f"{path}: no measured prefix tier for eps={eps:g}; "
                                 f"available {sorted(tiers)}")
        def load():
            from dmipy_sim.bank import read_rpk
            if tier is None and axes is None:
                return read_rpk(hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset",
                                                revision=revision))
            return fetch_pack_prefix(repo_id, path, tier if tier is not None else row["n_walkers"],
                                     axes=axes, revision=revision)
        return load
    return diams, n_t, dt, K, T_max, _LazyPacks([mk(r) for r in rows])


class ReplayFamily:
    """A diameter-sorted family of replay packs for one (shape, D0). Signals are interpolated linearly
    across diameter — the restricted signal is smooth in radius, so at the dataset's fine spacing this
    is accurate and gives a differentiable ``diameter`` for fitting.

    The waveform is compiled once (per scheme + rho) and cached on the instance; repeated calls with the
    same scheme (i.e. every voxel/iteration of a fit) reuse it."""

    def __init__(self, shape, diffusivity, diameters_m, packs, *, n_t_all=None, dt_all=None,
                 K_all=None, T_max=None, axes=None):
        self.shape = shape
        self.diffusivity = float(diffusivity)
        self.diameters = np.asarray(diameters_m, float)         # ascending, metres
        self.packs = packs if isinstance(packs, _LazyPacks) else list(packs)
        # Per-pack host arrays are materialised on demand: touching _pk[i] eagerly would defeat the lazy
        # pack loading above, since it reads the coefficients.
        self._pk_cache = {}
        # Which position axes this family carries/uses. Set by the MODEL, not the user: C6 handles the
        # axial term in closed form and P6 the two in-plane terms, so those axes are never contracted --
        # fetching them would move bytes only to multiply them by zero.
        self.axes = None if axes is None else tuple(int(i) for i in axes)
        # PER-PACK codec. The canonical dataset picks n_t (and occasionally K) per substrate -- a
        # sub-micron pore needs a finer save grid than a 20 um one -- so a family is generally
        # heterogeneous (measured: n_t in {2000, 4000, 8000}, K in {128, 196}). Anything that compiles
        # one scheme from pack 0 and reuses it across the family is wrong: it raises a shape mismatch
        # when K differs, and silently replays a waveform sampled on the wrong time grid when n_t does.
        if n_t_all is None:      # local eager path: read each pack's header (cheap) not its arrays
            n_t_all = [int(p.n_t) for p in self.packs]
            dt_all = [float(p.dt) for p in self.packs]
            K_all = [int(p.meta["compression"]["K"]) for p in self.packs]
            T_max = float(self.packs[0].meta["walk_params"]["T_max"])
        self.n_t_all = np.asarray(n_t_all, int)
        self.dt_all = np.asarray(dt_all, float)
        self.K_all = np.asarray(K_all, int)
        self.homogeneous = bool(len({(int(a), int(b)) for a, b in zip(self.n_t_all, self.K_all)}) == 1)
        self.n_t = int(self.n_t_all[0])
        self.dt = float(self.dt_all[0])
        self.K = int(self.K_all[0])
        self.T_max = float(T_max)

    @property
    def _pk(self):
        return self._PkView(self)

    class _PkView:
        """``family._pk[i]`` -> (C, w, K, blt) for pack i, materialised and cached on first use."""
        def __init__(self, fam):
            self._fam = fam
        def __len__(self):
            return len(self._fam.packs)
        def __getitem__(self, i):
            i = int(i); fam = self._fam
            if i not in fam._pk_cache:
                fam._pk_cache[i] = _pack_arrays(fam.packs[i], axes=fam.axes)
            return fam._pk_cache[i]

    @property
    def diameter_range(self):
        return float(self.diameters[0]), float(self.diameters[-1])

    def _signal_one(self, idx, W, rho_over_D, chi_hat):
        C, w, K, blt = self._pk[idx]
        return replay_complex(C, w, W, blt_dct=blt, rho_over_D=rho_over_D,
                              n_t=int(self.n_t_all[idx]), chi_hat=chi_hat)

    def replay_interpolated_raw(self, G, dt_in, diameter, *, rho_over_D=0.0, chi_hat=None):
        """Replay a waveform given on ITS OWN grid ``(G, dt_in)``, resampling and compiling PER PACK.

        Use this rather than :meth:`replay_interpolated` whenever the family may be heterogeneous, which
        for the canonical dataset is the normal case. Orientation must already be applied to ``G``;
        rotation is per-measurement and so commutes with the time-grid resampling.
        """
        d = float(np.clip(diameter, self.diameters[0], self.diameters[-1]))
        j = int(np.searchsorted(self.diameters, d))

        def one(idx):
            Gp = resample_waveform_to_grid(np.asarray(G), float(dt_in),
                                           int(self.n_t_all[idx]), float(self.dt_all[idx]))
            if self.axes is not None:
                Gp = Gp[..., list(self.axes)]      # compile from exactly the stored components
            W = compile_scheme(Gp, float(self.dt_all[idx]), int(self.K_all[idx]), _GAMMA,
                               n_t=int(self.n_t_all[idx]))
            return self._signal_one(idx, W, rho_over_D, chi_hat)

        if j <= 0:
            S = one(0)
        elif j >= len(self.diameters):
            S = one(len(self.diameters) - 1)
        else:
            d_lo, d_hi = self.diameters[j - 1], self.diameters[j]
            f = (d - d_lo) / (d_hi - d_lo)
            S = (1.0 - f) * one(j - 1) + f * one(j)
        return np.abs(S)

    def replay_interpolated(self, G_pack, diameter, *, rho_over_D=0.0, chi_hat=None):
        """Replay a waveform ``G_pack`` (n_meas, n_t, 3) already resampled onto this family's save grid
        against the two packs bracketing ``diameter`` and linearly interpolate the complex signal.
        ``rho_over_D`` (+ optional ``chi_hat``) activates the exact coherence-gated surface-relaxivity
        replay via each pack's boundary local time. Returns magnitude."""
        W = compile_scheme(G_pack, self.dt, self.K, _GAMMA, n_t=int(self.n_t))
        # once per scheme (+rho via chi_hat)
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


# safetensors dtype tags -> numpy
_ST_DTYPE = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "float16",
             "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
             "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool"}


def _st_header(url, headers):
    """(header dict, data section offset) from two small range GETs -- no tensor bytes transferred."""
    import json, struct, requests
    r = requests.get(url, headers={**headers, "Range": "bytes=0-7"}, allow_redirects=True)
    r.raise_for_status()
    n = struct.unpack("<Q", r.content)[0]
    r = requests.get(url, headers={**headers, "Range": f"bytes=8-{8 + n - 1}"}, allow_redirects=True)
    r.raise_for_status()
    return json.loads(r.content), 8 + n


def fetch_pack_prefix(repo_id, filename, n_rows, *, axes=None, revision=None):
    """Build a ReplayPack from HTTP RANGE READS of the first ``n_rows`` walkers.

    Why this exists: a coarser floor needs fewer walkers (floor ~ 1/sqrt(n)), and walkers are the leading
    axis of every per-walker tensor, so the rows a target eps needs are a contiguous byte range. Combined
    with one-tensor-per-axis positions, (axes you need) x (walkers you need) is one range per tensor.
    ``hf_hub_download`` cannot express that -- it transfers whole files -- so at eps=1e-2 a consumer would
    move ~3x more bytes than the answer requires (measured across this dataset: 24.0 GiB whole-file vs
    7.96 GiB of needed rows; 4x on the packs above 50 MiB).

    ``axes`` selects position components by index; omit for all three (needed for rotating / b-tensor
    encodings, which depend on the joint trajectory).
    """
    import numpy as np
    from huggingface_hub import hf_hub_url, get_token
    import requests
    from dmipy_sim.replay import ReplayPack
    from dmipy_sim.compression import POSITION_AXES

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    tok = get_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    hdr, base = _st_header(url, headers)
    meta = __import__("json").loads((hdr.get("__metadata__") or {}).get("rpk", "{}"))
    n_full = int((meta.get("walk_params") or {}).get("n_walkers") or 0)
    n_rows = int(min(n_rows, n_full)) if n_full else int(n_rows)

    keep_pos = None if axes is None else {POSITION_AXES[i] for i in axes}
    arrays, moved = {}, 0
    for name, spec in hdr.items():
        if name == "__metadata__":
            continue
        if name in POSITION_AXES and keep_pos is not None and name not in keep_pos:
            continue
        shape = list(spec["shape"]); dt = np.dtype(_ST_DTYPE[spec["dtype"]])
        s0, s1 = spec["data_offsets"]
        if shape and shape[0] == n_full and n_rows < n_full:
            row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            nbytes = n_rows * row * dt.itemsize
            a, b = base + s0, base + s0 + nbytes - 1
            shape[0] = n_rows
        else:
            a, b = base + s0, base + s1 - 1
        r = requests.get(url, headers={**headers, "Range": f"bytes={a}-{b}"}, allow_redirects=True)
        r.raise_for_status()
        moved += len(r.content)
        arrays[name] = np.frombuffer(r.content, dtype=dt).reshape(shape)
    meta = dict(meta)
    meta["walk_params"] = dict(meta.get("walk_params", {}), n_walkers=n_rows)
    meta["fetched"] = dict(n_rows=n_rows, n_rows_available=n_full, bytes=moved,
                           axes=(None if axes is None else list(axes)))
    return ReplayPack(arrays, meta)


def _pack_header(path):
    """(diameter_m, n_t, K) from a pack's safetensors HEADER only -- no coefficient bytes read.

    Building a family must not read the packs. safetensors keeps its JSON header at the front of the
    file, so metadata and tensor shapes cost one small read; pulling `provenance.diameter_m` by opening
    the pack instead would materialise every pack in the family (200 x tens of MiB) to answer a question
    about two of them.
    """
    import json
    from safetensors import safe_open
    with safe_open(path, framework="np") as h:
        meta = json.loads((h.metadata() or {}).get("rpk", "{}"))
        K = int(h.get_slice("pos_x").get_shape()[1]) if "pos_x" in h.keys() else None
    prov = meta.get("provenance") or {}
    d = prov.get("diameter_m")
    if d is None:
        d = float(os.path.basename(path).split("d")[-1].split("um")[0]) * 1e-6
    wp = meta.get("walk_params") or {}
    n_t = int(wp.get("n_t") or 0)
    if K is None:
        K = int((meta.get("compression") or {}).get("K") or 0)
    return float(d), n_t, K


def load_replay_family(shape, diffusivity, *, dataset_dir=None, repo_id=None, revision=None,
                       eps=None, axes=None):
    """Load (and cache) the diameter family for ``shape`` at intrinsic diffusivity ``diffusivity``.

    Resolution order:
      1. ``dataset_dir`` / ``$SUBSTRATE_COMMONS_DATA`` -- a local copy, searched as
         ``<root>/canonical/D0-<d.dd>e-9/<shape>/*.rpk``.
      2. otherwise the Hub (``repo_id``, default ``SubstrateCommons/canonical-pores``): the manifest is
         downloaded once and individual packs are fetched ON DEMAND, so a fit that touches two diameters
         transfers two files rather than the whole dataset.

    ``eps`` requests a coarser floor: each pack is RANGE-READ down to the measured prefix that meets it
    (from the manifest's ``prefix_tiers``), so the transfer shrinks with the accuracy you actually need.
    ``axes`` restricts which position components are fetched -- 1 for a slab, 2 for a cylinder's
    transverse plane -- and must be omitted for rotating / b-tensor encodings, which need the joint
    trajectory. Both are Hub-only; a local copy is already on disk.

    Either way the family is lazy: constructing it reads no coefficients.
    """
    key = (os.path.abspath(dataset_dir) if dataset_dir else
           (os.environ.get("SUBSTRATE_COMMONS_DATA") or f"hf:{repo_id or DEFAULT_REPO}"),
           shape, round(float(diffusivity) * 1e9, 3),
           (None if eps is None else float(eps)), (None if axes is None else tuple(axes)))
    if key in _FAMILY_CACHE:
        return _FAMILY_CACHE[key]

    root = dataset_dir or os.environ.get("SUBSTRATE_COMMONS_DATA")
    if root:
        from dmipy_sim.bank import read_rpk
        subdir = os.path.join(root, "canonical", f"D0-{diffusivity*1e9:.2f}e-9", shape)
        paths = sorted(glob.glob(os.path.join(subdir, "*.rpk")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(root, "**", shape, "*.rpk"), recursive=True))
        if not paths:
            raise FileNotFoundError(
                f"no replay packs for shape={shape!r} D0={diffusivity:.2e} under {root!r}; unset "
                f"dataset_dir/$SUBSTRATE_COMMONS_DATA to fetch from the Hub instead")
        hdr = [_pack_header(pth) for pth in paths]          # header-only: no coefficients read
        diams = [h[0] for h in hdr]
        order = np.argsort(diams)
        ordered = [paths[i] for i in order]
        n_t = np.asarray([hdr[i][1] for i in order], int)
        K = np.asarray([hdr[i][2] for i in order], int)
        T_max = 0.2
        packs = _LazyPacks([(lambda q=q: read_rpk(q)) for q in ordered])
        fam = ReplayFamily(shape, diffusivity, np.asarray(diams)[order], packs,
                          n_t_all=n_t, dt_all=T_max / (n_t - 1), K_all=K, T_max=T_max, axes=axes)
    else:
        diams, n_t, dt, K, T_max, packs = _hf_family(shape, diffusivity,
                                                     repo_id or DEFAULT_REPO, revision,
                                                     eps=eps, axes=axes)
        fam = ReplayFamily(shape, diffusivity, diams, packs, n_t_all=n_t, dt_all=dt, K_all=K,
                           T_max=T_max, axes=axes)
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


def orient_to_x(mu):
    """Rotation R (3×3) mapping unit axis ``mu`` → +x (rows = mu, e1, e2). For a slab/plane pack whose
    restricted axis is x (``Box1D``): with the plane NORMAL ``mu``, ``G @ R.T`` sends the gradient's
    normal component to x (restricted) and the in-plane components to y,z (free)."""
    R = orient_to_z(mu)                       # rows e1, e2, mu
    return R[[2, 0, 1], :]                     # -> rows mu, e1, e2  (mu -> +x)
