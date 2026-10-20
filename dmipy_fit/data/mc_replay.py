"""Loader for the canonical Monte-Carlo *replay-pack* reference dataset (Substrate Commons).

A family of ``.rpk`` packs, one per diameter, for a single restricted shape (cylinder, sphere or plane) at a
fixed intrinsic diffusivity D0. The ``C6MonteCarloReplayCylinder`` / ``S6MonteCarloReplaySphere`` /
``P6MonteCarloReplayPlane`` compartment models load a family through here and interpolate the replayed signal
across diameter.

Packs are large, so they are NOT bundled in the wheel: point ``dataset_dir`` at a local directory
(populated from the Substrate Commons Hugging Face dataset), or set ``$SUBSTRATE_COMMONS_DATA``.

A pack is its walk as coefficients on ONE basis (RPK.md 9.5): per storage window and axis the two exact endpoints
and ``K`` sine bands of the bridge between them, accumulated while the walk ran; there is no save grid. Forward
evaluation is sim's compiled replay (:mod:`dmipy_sim.replay.replay`): the acquisition waveform is projected onto
each window's bands ONCE (:func:`dmipy_sim.replay.compile_scheme`), after which each replay is a single matmul --
the same kernel as ``dmipy_sim.replay.ReplayPack.replay`` (:func:`dmipy_sim.replay.replay_coefficients`), fast
enough to fit. Every pack enters through sim's reader, which refuses by name a pack written on a save grid
(``dmipy_sim.replay.replay.REFUSED_KEYS``); nothing is converted."""
import os
import glob
import numpy as np

from ..core.constants import CONSTANTS
from dmipy_sim.replay import compile_scheme, replay_coefficients, surface_logweight
from dmipy_sim.replay.compression import band_projection

_FAMILY_CACHE = {}
_GAMMA = CONSTANTS["water_gyromagnetic_ratio"]


def _data_root(dataset_dir=None):
    return (dataset_dir or os.environ.get("SUBSTRATE_COMMONS_DATA")
            or os.path.join(os.path.dirname(__file__), "mc_replay"))


class PackCoefficients:
    """A pack's stored coefficients as the fit contracts them, float64 host arrays.

    ``C`` is ``(n_walkers, n_windows (K + 2), n_axes)``: the windows' position coefficients (two endpoints then
    ``K`` bands each, :func:`dmipy_sim.replay.compression.read_position_coeffs`) concatenated along the coefficient
    axis in window order, so that one matmul against :meth:`compile` is the sum over the windows the spec's 4.3
    prescribes. ``w`` are the walkers' weights; ``windows`` the per-window arrays and the time each window starts
    at on the walk's clock, which the contact channel (C2) is read from; ``surface`` is that channel's metadata
    (``compression.channels.boundary_local_time``) or ``None`` when the pack carries none.

    ``K`` is the number of SINE BANDS, i.e. the stored width per window minus the two endpoint coefficients.
    Handing the width to compile_scheme instead would give a scheme of the right shape to multiply and the wrong
    thing to multiply by.
    """

    def __init__(self, pack, axes=None):
        from dmipy_sim.replay.compression import read_position_coeffs
        self.K = int(pack.K)
        self.T_window = float(pack.T_window)
        self.n_windows = int(pack.n_windows)
        self.windows = [(pack.window(i).arrays, i * self.T_window) for i in range(self.n_windows)]
        self.C = np.concatenate([read_position_coeffs(a, axes=axes, dtype=np.float64) for a, _ in self.windows], axis=1)
        self.w = np.asarray(pack.arrays.get("spin_weights", np.ones(self.C.shape[0])), np.float64)
        self.surface = ((pack.meta.get("compression", {}).get("channels", {}) or {}).get("boundary_local_time")
                        if pack.has_surface else None)

    @property
    def n_coeffs(self):
        """Stored coefficients per window and axis, ``K + 2``."""
        return self.K + 2

    def windows_reached(self, n_wf, dt_wf):
        """How many leading windows an acquisition of ``n_wf`` samples on the grid ``dt_wf`` reaches: the windows
        whose start lies before its readout, at least one. A window beyond the readout projects to zero and is not
        contracted."""
        T_acq = (int(n_wf) - 1) * float(dt_wf)
        n = int(np.ceil(T_acq / self.T_window - 1e-12))
        return int(min(self.n_windows, max(1, n)))

    def compile(self, G, dt_wf):
        """The waveform ``G`` ``(n_meas, n_wf, n_axes)`` on its own grid ``dt_wf`` compiled against the windows it
        reaches: ``(n_reached (K + 2) n_axes, n_meas)``, a one-window pack through
        :func:`dmipy_sim.replay.compile_scheme`, a windowed pack each window's bands from its own start
        (:func:`dmipy_sim.replay.compression.band_projection`), rows in the order :attr:`C` flattens to. Nothing is
        resampled: an edge anywhere carries exactly its b. A gradient beyond the walk is refused by name."""
        G = np.asarray(G, np.float64)
        if self.n_windows == 1:
            return compile_scheme(G, float(dt_wf), self.K, _GAMMA, T=self.T_window)
        e = np.arange(G.shape[1]) * float(dt_wf)
        beyond = e > self.n_windows * self.T_window * (1.0 + 1e-9)
        if beyond.any() and np.any(G[:, beyond, :] != 0.0):
            raise ValueError(f"the waveform has a non-zero gradient beyond the pack's T_max = "
                             f"{self.n_windows * self.T_window:.6g} s (its own extent is {e[-1] + float(dt_wf):.6g} s); "
                             "the stored path ends there")
        parts = []
        for i in range(self.windows_reached(G.shape[1], dt_wf)):
            Wp = band_projection(G, float(dt_wf), self.T_window, self.K, t0=i * self.T_window)
            parts.append((_GAMMA * Wp).reshape(Wp.shape[0], -1).T)
        return np.concatenate(parts, axis=0)

    def coefficients_for(self, W):
        """The slice of :attr:`C` a compiled scheme ``W`` contracts: its leading windows."""
        return self.C[:, :W.shape[0] // self.C.shape[2], :]

    def surface_logweight(self, rho_over_D, gate, dt_gate):
        """The per-walker surface log-weight ``(rho / D) int chi dB`` of the contact channel, the gate ``chi`` on its
        own grid ``dt_gate`` (its last sample the readout), summed over the windows it reaches
        (:func:`dmipy_sim.replay.surface_logweight`). Refused when the pack carries no C2 channel."""
        if self.surface is None:
            raise ValueError("surface relaxivity was asked of a pack that carries no boundary-local-time channel (C2)")
        gate = np.asarray(gate, np.float64).reshape(-1)
        n = self.windows_reached(gate.shape[0], dt_gate)
        return sum(surface_logweight(a, rho_over_D, self.surface, gate, float(dt_gate), t0=t0)
                   for a, t0 in self.windows[:n])

    def signal(self, W, *, rho_over_D=0.0, gate=None, dt_gate=None, complex_signal=True):
        """The replay of the compiled scheme ``W`` (:meth:`compile`): :func:`dmipy_sim.replay.replay_coefficients`,
        with the surface log-weight when ``rho_over_D`` is set (``gate`` on its grid ``dt_gate``)."""
        slw = self.surface_logweight(rho_over_D, gate, dt_gate) if rho_over_D else None
        return replay_coefficients(self.coefficients_for(W), self.w, W, surface_logw=slw, complex_signal=complex_signal)


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


class PackGrid:
    """What a family knows of a pack before reading its coefficients: its band ``K`` and the window the bands are on
    (``T_window`` s, ``n_windows`` of them, ``T_max = n_windows T_window``), and the block ``h`` (s) the walk
    accumulated over (``walk_params.block``, a constant of the pack, RPK.md 9.5.5)."""

    def __init__(self, K, T_window, n_windows, h):
        self.K, self.T_window, self.n_windows, self.h = int(K), float(T_window), int(n_windows), float(h)

    @property
    def T_max(self):
        return self.n_windows * self.T_window

    @classmethod
    def of_meta(cls, meta, where):
        """From a pack's header, through sim's refusal of a header this revision does not read."""
        from dmipy_sim.replay.replay import _refuse_predating
        _refuse_predating(meta, where)
        wp = meta["walk_params"]
        win = wp.get("windows") or {"n": 1, "T": wp["T_max"]}
        return cls(meta["compression"]["K"], win["T"], win.get("n") or 1, wp["block"]["h"])

    @classmethod
    def of_pack(cls, pack):
        return cls(pack.K, pack.T_window, pack.n_windows, pack.dt)

    @classmethod
    def of_row(cls, row, where):
        """From a manifest row (:func:`dmipy_sim.replay.publish.manifest_row`: ``K``, ``T_max``, ``block``,
        ``windows``). A row of a manifest written on a save grid names ``n_t`` or ``dt_traj`` and is refused by name,
        as sim's reader refuses the pack."""
        for key in ("n_t", "dt_traj"):
            if key in row:
                raise ValueError(f"{where}: this manifest row carries `{key}`, which this revision does not read; "
                                 "re-walk the pack with a conformant producer. A pack is never converted.")
        win = row.get("windows") or {"n": 1, "T": row["T_max"]}
        return cls(row["K"], win["T"], win.get("n") or 1, row["block"]["h"])


def _hf_manifest(repo_id, revision=None):
    from huggingface_hub import hf_hub_download
    import json
    p = hf_hub_download(repo_id=repo_id, filename="manifest.json", repo_type="dataset",
                        revision=revision)
    return json.load(open(p))


def _hf_family(shape, diffusivity, repo_id, revision=None, eps=None, axes=None):
    """Diameters + per-pack grid from the manifest ALONE; packs fetched on demand, one file each.

    The manifest carries ``K``, ``T_max``, ``block`` and ``windows`` per substrate, so the family is fully described
    without downloading a single pack -- which is what makes X6 usable straight from the Hub instead of after a
    24 GiB pull.
    """
    from huggingface_hub import hf_hub_download
    man = _hf_manifest(repo_id, revision)
    rows = [r for r in man["substrates"] if r["shape"] == shape]
    if not rows:
        raise FileNotFoundError(f"{repo_id}: no substrates of shape {shape!r} in manifest.json")
    rows.sort(key=lambda r: r["d_um"])
    diams = np.asarray([r["d_um"] * 1e-6 for r in rows], float)
    grids = [PackGrid.of_row(r, f"{repo_id}/{r['path']}") for r in rows]

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
            from dmipy_sim.replay import read_rpk
            if tier is None and axes is None:
                return read_rpk(hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset",
                                                revision=revision))
            return fetch_pack_prefix(repo_id, path, tier if tier is not None else row["n_walkers"],
                                     axes=axes, revision=revision)
        return load
    return diams, grids, _LazyPacks([mk(r) for r in rows])


class ReplayFamily:
    """A diameter-sorted family of replay packs for one (shape, D0). Signals are interpolated linearly
    across diameter -- the restricted signal is smooth in radius, so at the dataset's fine spacing this
    is accurate and gives a differentiable ``diameter`` for fitting.

    The waveform is compiled per pack onto the pack's own bands (:meth:`PackCoefficients.compile`, cheap next to
    the walkers' matmul); the family is generally heterogeneous in ``K`` and window, so nothing compiled for one
    pack is reused for another."""

    def __init__(self, shape, diffusivity, diameters_m, packs, *, grids=None, axes=None):
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
        # PER-PACK grid (:class:`PackGrid`). The canonical dataset picks K (and the walk length) per substrate, so
        # a family is generally heterogeneous. Anything that compiles one scheme from pack 0 and reuses it across
        # the family is wrong: it raises a shape mismatch when K differs, and silently projects onto the wrong
        # window when T_window does.
        if grids is None:      # local eager path: read each pack's header (cheap) not its arrays
            grids = [PackGrid.of_pack(p) for p in self.packs]
        self.grids = list(grids)
        self.K_all = np.asarray([g.K for g in self.grids], int)
        self.T_window_all = np.asarray([g.T_window for g in self.grids], float)
        self.n_windows_all = np.asarray([g.n_windows for g in self.grids], int)
        self.h_all = np.asarray([g.h for g in self.grids], float)
        self.homogeneous = len({(g.K, g.T_window, g.n_windows) for g in self.grids}) == 1
        self.K = int(self.K_all[0])
        self.T_window = float(self.T_window_all[0])
        self.T_max = float(min(g.T_max for g in self.grids))     # the walk length every pack of the family reaches

    @property
    def _pk(self):
        return self._PkView(self)

    class _PkView:
        """``family._pk[i]`` -> :class:`PackCoefficients` of pack i, materialised and cached on first use."""
        def __init__(self, fam):
            self._fam = fam
        def __len__(self):
            return len(self._fam.packs)
        def __getitem__(self, i):
            i = int(i); fam = self._fam
            if i not in fam._pk_cache:
                fam._pk_cache[i] = PackCoefficients(fam.packs[i], axes=fam.axes)
            return fam._pk_cache[i]

    @property
    def diameter_range(self):
        return float(self.diameters[0]), float(self.diameters[-1])

    def replay_interpolated_raw(self, G, dt_in, diameter, *, rho_over_D=0.0, chi=None):
        """Replay a waveform given on ITS OWN grid ``(G, dt_in)``, compiled PER PACK onto the pack's bands (no
        resampling: an edge anywhere carries exactly its b), against the two packs bracketing ``diameter``, the
        complex signal interpolated linearly. Orientation must already be applied to ``G``. ``rho_over_D``
        activates the exact surface-relaxivity replay; ``chi`` is the coherence gate on the acquisition's own grid
        (``(n_wf,)``, its last sample the readout) and defaults to 1 over the acquisition, so contact after the
        echo is not counted whatever the pack's length (a pack is walked longer than any one acquisition).
        Returns magnitude.
        """
        G = np.asarray(G, np.float64)
        d = float(np.clip(diameter, self.diameters[0], self.diameters[-1]))
        j = int(np.searchsorted(self.diameters, d))
        gate = np.ones(G.shape[1]) if chi is None else chi

        def one(idx):
            pk = self._pk[idx]
            Gk = G if self.axes is None else G[..., list(self.axes)]      # compile from exactly the stored components
            return pk.signal(pk.compile(Gk, dt_in), rho_over_D=rho_over_D, gate=gate, dt_gate=dt_in)
        if j <= 0:
            S = one(0)
        elif j >= len(self.diameters):
            S = one(len(self.diameters) - 1)
        else:
            d_lo, d_hi = self.diameters[j - 1], self.diameters[j]
            f = (d - d_lo) / (d_hi - d_lo)
            S = (1.0 - f) * one(j - 1) + f * one(j)
        return np.abs(S)


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
    encodings, which depend on the joint trajectory). The header is read first and refused as sim's reader
    refuses it (``dmipy_sim.replay.replay.REFUSED_KEYS``) before any tensor bytes move.
    """
    import numpy as np
    from huggingface_hub import hf_hub_url, get_token
    import requests
    from dmipy_sim.replay import ReplayPack
    from dmipy_sim.replay.replay import _refuse_predating
    from dmipy_sim.replay.compression import POSITION_AXES

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    tok = get_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    hdr, base = _st_header(url, headers)
    meta = __import__("json").loads((hdr.get("__metadata__") or {}).get("rpk", "{}"))
    _refuse_predating(meta, f"{repo_id}/{filename}")
    n_full = int((meta.get("walk_params") or {}).get("n_walkers") or 0)
    n_rows = int(min(n_rows, n_full)) if n_full else int(n_rows)

    drop = () if axes is None else tuple(k for i, k in enumerate(POSITION_AXES) if i not in axes)
    arrays, moved = {}, 0
    for name, spec in hdr.items():
        if name == "__metadata__":
            continue
        channel = name.split("/", 1)[-1]                      # a later window's tensor sits under ``s{i}/``
        if any(channel == k or channel.startswith(f"{k}_") for k in drop):
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
    pack = ReplayPack(arrays, meta, source=url)
    pack.susceptibility_field                                 # the reader's check of the declared field
    return pack


def _pack_header(path):
    """``(diameter_m, PackGrid)`` from a pack's safetensors HEADER only -- no coefficient bytes read.

    Building a family must not read the packs. safetensors keeps its JSON header at the front of the
    file, so metadata costs one small read; pulling `provenance.diameter_m` by opening the pack instead
    would materialise every pack in the family (200 x tens of MiB) to answer a question about two of them.
    """
    import json
    from safetensors import safe_open
    with safe_open(path, framework="np") as h:
        meta = json.loads((h.metadata() or {}).get("rpk", "{}"))
    grid = PackGrid.of_meta(meta, path)
    prov = meta.get("provenance") or {}
    d = prov.get("diameter_m")
    if d is None:
        d = float(os.path.basename(path).split("d")[-1].split("um")[0]) * 1e-6
    return float(d), grid


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
        from dmipy_sim.replay import read_rpk
        subdir = os.path.join(root, "canonical", f"D0-{diffusivity*1e9:.2f}e-9", shape)
        paths = sorted(glob.glob(os.path.join(subdir, "*.rpk")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(root, "**", shape, "*.rpk"), recursive=True))
        if not paths:
            raise FileNotFoundError(
                f"no replay packs for shape={shape!r} D0={diffusivity:.2e} under {root!r}; unset "
                f"dataset_dir/$SUBSTRATE_COMMONS_DATA to fetch from the Hub instead")
        hdr = [_pack_header(pth) for pth in paths]          # header-only: no coefficients read
        order = np.argsort([h[0] for h in hdr])
        packs = _LazyPacks([(lambda q=paths[i]: read_rpk(q)) for i in order])
        fam = ReplayFamily(shape, diffusivity, [hdr[i][0] for i in order], packs,
                           grids=[hdr[i][1] for i in order], axes=axes)
    else:
        diams, grids, packs = _hf_family(shape, diffusivity, repo_id or DEFAULT_REPO, revision,
                                         eps=eps, axes=axes)
        fam = ReplayFamily(shape, diffusivity, diams, packs, grids=grids, axes=axes)
    _FAMILY_CACHE[key] = fam
    return fam


def family_from_packs(shape, diffusivity, diameters_m, packs):
    """Build a :class:`ReplayFamily` directly from in-memory ReplayPack objects (tests / custom sets)."""
    order = np.argsort(np.asarray(diameters_m, float))
    return ReplayFamily(shape, diffusivity, np.asarray(diameters_m, float)[order],
                        [packs[i] for i in order])


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
