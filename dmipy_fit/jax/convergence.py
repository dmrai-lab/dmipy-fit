"""Convergence analysis: data-driven optimizer iteration recommendations.

Fits a reference solution (maxiter=200) then sweeps faster configs,
measuring normalised MAE per signal-complexity stratum.  Returns the
fastest config that meets a user-specified MAE target.

Works on any tissue type — brain, muscle, tumour, spinal cord, phantom.
For brain data, ``sampler='brain'`` adds WM/GM tissue labels using the
JAX DTI fitter (no dipy dependency).

Usage
-----
    from dmipy_fit.jax.convergence import analyze_convergence

    report = analyze_convergence(
        optimizer,          # JaxOptimizer for your model + scheme
        data_flat,          # (N_vox, N_meas) float32, S/S0 normalised
        target_mae=0.01,
    )
    report.summary()
    # → Recommended: {'maxiter': 12}
"""

import time
import numpy as np
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class SDMStratifiedSampler:
    """Sample voxels stratified by Signal Decay Metric (SDM).

    SDM = mean_shells(log(S0 / S_shell)) — a scalar per voxel that measures
    how much diffusion attenuation the signal shows, independent of tissue
    type, organ, or species.

    Low SDM  → restricted diffusion (hard to fit — WM, dense tumour, etc.)
    High SDM → fast diffusion (easy to fit — CSF, necrosis, free water)

    Sampling is weighted toward low-SDM voxels so the convergence analysis
    is governed by the hardest cases, not the trivially easy ones.

    Works on any diffusion MRI dataset.

    Parameters
    ----------
    n_voxels : int
        Total number of voxels to sample.  Default 1000.
    strata_fractions : tuple of 3 floats summing to 1
        Fraction to draw from (low, mid, high) SDM strata.
        Default (0.5, 0.3, 0.2) — weight toward hard voxels.
    seed : int
        RNG seed.  Default 42.
    """

    def __init__(self, n_voxels=1000,
                 strata_fractions=(0.5, 0.3, 0.2), seed=42):
        assert abs(sum(strata_fractions) - 1.0) < 1e-6, \
            "strata_fractions must sum to 1"
        self.n_voxels = n_voxels
        self.strata_fractions = strata_fractions
        self.seed = seed

    @property
    def stratum_names(self):
        return ['low_sdm', 'mid_sdm', 'high_sdm']

    def _compute_sdm(self, data_flat, scheme):
        """Signal Decay Metric, shape (N_vox,).

        Inlined from three_tissue_response.signal_decay_metric so it works
        on flat (N_vox, N_meas) arrays without reshaping.
        """
        b0_mask = np.array(scheme.b0_mask, dtype=bool)
        mean_b0 = data_flat[:, b0_mask].mean(axis=1)           # (N_vox,)

        shells = np.array(scheme.unique_dwi_indices)
        shell_idx = np.array(scheme.shell_indices)
        shell_means = np.stack([
            data_flat[:, shell_idx == s].mean(axis=1)
            for s in shells
        ], axis=1)                                              # (N_vox, n_shells)

        sdm = np.zeros(data_flat.shape[0])
        ok  = (mean_b0 > 0) & (shell_means.min(axis=1) > 0)
        sdm[ok] = np.mean(
            np.log(mean_b0[ok, None] / shell_means[ok]), axis=1)
        return np.clip(sdm, 0, 10)

    def sample(self, data_flat, scheme):
        """Draw a stratified sample.

        Parameters
        ----------
        data_flat : (N_vox, N_meas) float32 — S/S0 normalised
        scheme    : acquisition scheme

        Returns
        -------
        indices : (n_sampled,) int  — row indices into data_flat
        labels  : (n_sampled,) int  — stratum id (0=low, 1=mid, 2=high)
        sdm     : (N_vox,) float    — SDM for all input voxels
        """
        rng = np.random.default_rng(self.seed)
        sdm = self._compute_sdm(data_flat, scheme)

        p33, p66 = np.percentile(sdm, [33, 66])
        strata   = np.digitize(sdm, [p33, p66])   # 0 / 1 / 2

        chosen, chosen_labels = [], []
        for s, frac in enumerate(self.strata_fractions):
            n_s  = max(1, int(self.n_voxels * frac))
            pool = np.where(strata == s)[0]
            if len(pool) == 0:
                continue
            idx = rng.choice(pool, size=min(n_s, len(pool)), replace=False)
            chosen.append(idx)
            chosen_labels.append(np.full(len(idx), s, dtype=np.int32))

        return np.concatenate(chosen), np.concatenate(chosen_labels), sdm


class ThreeTissueSampler(SDMStratifiedSampler):
    """Sample voxels stratified by WM / GM tissue class (brain data).

    Uses JAX DTI FA (no dipy) to identify WM (FA > 0.2) and
    SDM + Dhollander16 optimal threshold to separate GM from CSF.
    CSF is excluded — it contributes trivially easy voxels that mask
    the convergence difficulty for WM-targeted models.

    The hardest stratum is WM, which drives the headline recommendation
    for orientation models (NODDI, ball-stick, etc.).

    Falls back to SDM stratification if FA computation fails.

    Parameters
    ----------
    n_voxels : int.  Default 1000.
    wm_fraction : float.  Fraction from WM.  Default 0.7.
    gm_fraction : float.  Fraction from GM.  Default 0.3.
    seed : int.  Default 42.
    """

    def __init__(self, n_voxels=1000,
                 wm_fraction=0.7, gm_fraction=0.3, seed=42):
        assert abs(wm_fraction + gm_fraction - 1.0) < 1e-6, \
            "wm_fraction + gm_fraction must equal 1"
        super().__init__(
            n_voxels=n_voxels,
            strata_fractions=(wm_fraction, gm_fraction, 0.0),
            seed=seed)
        self._wm_frac = wm_fraction
        self._gm_frac = gm_fraction

    @property
    def stratum_names(self):
        return ['wm', 'gm', 'csf']

    def sample(self, data_flat, scheme):
        from .dti_jax import build_dti_fitter
        from ..tissue_response.three_tissue_response import optimal_threshold

        rng = np.random.default_rng(self.seed)
        sdm = self._compute_sdm(data_flat, scheme)

        # FA via JAX DTI (auto b_max for multi-shell acquisitions)
        b_max = 1.5e9 if np.array(scheme.bvalues).max() > 2e9 else None
        fit_dti = build_dti_fitter(scheme, b_max=b_max)

        b0_mask = np.array(scheme.b0_mask, dtype=bool)
        S0 = np.maximum(data_flat[:, b0_mask].mean(axis=1, keepdims=True), 1e-6)
        data_norm = (data_flat / S0).astype(np.float32)
        _, fa = fit_dti(jnp.array(data_norm))
        fa = np.array(fa)

        mask_wm = fa > 0.2

        # Separate GM / CSF: optimal threshold on SDM of non-WM voxels
        non_wm_sdm = sdm[(~mask_wm) & (sdm > 0)]
        if len(non_wm_sdm) > 10:
            try:
                opt = optimal_threshold(non_wm_sdm)
                mask_gm  = (~mask_wm) & (sdm <= opt)
            except Exception:
                mask_gm = ~mask_wm   # fallback: all non-WM → GM
        else:
            mask_gm = ~mask_wm

        chosen, chosen_labels = [], []
        for label, mask, frac in [
            (0, mask_wm, self._wm_frac),
            (1, mask_gm, self._gm_frac),
        ]:
            n_s  = max(1, int(self.n_voxels * frac))
            pool = np.where(mask)[0]
            if len(pool) == 0:
                continue
            idx = rng.choice(pool, size=min(n_s, len(pool)), replace=False)
            chosen.append(idx)
            chosen_labels.append(np.full(len(idx), label, dtype=np.int32))

        return np.concatenate(chosen), np.concatenate(chosen_labels), sdm


class MaskSampler:
    """Sample from an explicit index array or boolean mask.

    Use this for non-brain data where you already have a tissue ROI,
    or when you want full control over which voxels are analysed.

    Parameters
    ----------
    indices_or_mask : (N,) int array, or (N_vox,) bool array
    """

    def __init__(self, indices_or_mask):
        arr = np.asarray(indices_or_mask)
        self._indices = np.where(arr)[0] if arr.dtype == bool else arr.astype(np.int64)

    @property
    def stratum_names(self):
        return ['all']

    def sample(self, data_flat, scheme):
        labels = np.zeros(len(self._indices), dtype=np.int32)
        sdm    = np.zeros(data_flat.shape[0])
        return self._indices, labels, sdm


# ---------------------------------------------------------------------------
# Model–tissue affinity
# ---------------------------------------------------------------------------

def _detect_model_affinity(model):
    """Infer primary tissue type from model parameter structure.

    Returns 'wm', 'gm', or None (ambiguous / isotropic).
    """
    params = model.parameter_cardinality
    flags  = getattr(model, 'parameter_optimization_flags', {})

    if any(k.lower().endswith('_mu') and flags.get(k, True) for k in params):
        return 'wm'

    exchange_keywords = ('k_', 'tex', 't_ex', 'kex', 'exchange')
    if any(any(kw in k.lower() for kw in exchange_keywords) for k in params):
        return 'gm'

    return None   # isotropic, ambiguous


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class ConvergenceReport:
    """Results of a convergence sweep.

    Attributes
    ----------
    recommended : dict
        Kwargs for the fastest config meeting ``target_mae`` in the hardest
        stratum.  Pass directly to ``JaxOptimizer(..., **report.recommended)``.
    """

    def __init__(self, results, timing, target_mae,
                 stratum_names, n_per_stratum,
                 reference_iter, model_affinity):
        self._results      = results       # {label: {stratum: mae}}
        self._timing       = timing        # {label: sec/vox}
        self.target_mae    = target_mae
        self.stratum_names = stratum_names
        self.n_per_stratum = n_per_stratum
        self.reference_iter = reference_iter
        self.model_affinity = model_affinity
        self.recommended   = self._find_recommended()

    # The hardest stratum is always index 0 — low_sdm / wm / all.
    # Samplers are constructed so the most restricted voxels come first.
    @property
    def hardest_stratum(self):
        return self.stratum_names[0]

    def _find_recommended(self):
        hard = self.hardest_stratum
        for label, strata_mae in self._results.items():
            if strata_mae.get(hard, 1.0) <= self.target_mae:
                return _label_to_kwargs(label)
        # Nothing met target — return config with lowest MAE in hard stratum
        best = min(self._results,
                   key=lambda l: self._results[l].get(hard, 1.0))
        return _label_to_kwargs(best)

    def summary(self):
        hard = self.hardest_stratum
        active = [s for s in self.stratum_names if self.n_per_stratum.get(s, 0) > 0]

        print(f"\n{'='*62}")
        print(f"  Convergence analysis")
        if self.model_affinity:
            print(f"  Model affinity : {self.model_affinity.upper()}")
        print(f"  Target MAE     : {self.target_mae}")
        print(f"  Reference      : maxiter={self.reference_iter}")
        print(f"  Recommendation : driven by '{hard}' stratum")
        print(f"{'='*62}")

        col = 13
        header = f"  {'Config':<18}"
        for s in active:
            n = self.n_per_stratum.get(s, 0)
            header += f"  {s+f'(n={n})':>{col}}"
        header += f"  {'µs/vox':>8}"
        print(header)
        print(f"  {'-'*58}")

        for label, strata_mae in self._results.items():
            row = f"  {label:<18}"
            for s in active:
                mae = strata_mae.get(s, float('nan'))
                mark = ' ✓' if s == hard and mae <= self.target_mae else '  '
                row += f"  {mae:>{col-2}.4f}{mark}"
            us = self._timing.get(label, 0) * 1e6
            row += f"  {us:>8.1f}"
            print(row)

        print(f"\n  → Recommended: {self.recommended}")
        print(f"{'='*62}\n")

    def plot(self):
        """MAE vs iteration count curve, hardest stratum only."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — install it to use report.plot()")
            return

        hard   = self.hardest_stratum
        labels = list(self._results.keys())
        iters  = [_label_to_kwargs(l).get('maxiter', 0) for l in labels]
        maes   = [self._results[l].get(hard, float('nan')) for l in labels]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iters, maes, 'o-', color='steelblue', linewidth=2)
        ax.axhline(self.target_mae, color='tomato', linestyle='--',
                   label=f'target MAE = {self.target_mae}')
        rec = self.recommended.get('maxiter')
        if rec and rec in iters:
            rec_mae = maes[iters.index(rec)]
            ax.scatter([rec], [rec_mae], color='tomato', zorder=5, s=80,
                       label=f'recommended: maxiter={rec}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel(f'Normalised MAE ({hard} stratum)')
        ax.set_title('Convergence analysis')
        ax.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_to_kwargs(label):
    """'maxiter=10' → {'maxiter': 10}."""
    if label.startswith('maxiter='):
        return {'maxiter': int(label.split('=')[1])}
    return {}


def _build_x0(optimizer, sample_data, dtype):
    """Initial parameter vector for the sample batch."""
    lower = np.array(optimizer._lower)
    upper = np.array(optimizer._upper)
    mid   = jnp.array((lower + upper) * 0.5, dtype=dtype)
    x0    = jnp.broadcast_to(mid[None], (sample_data.shape[0], len(mid)))

    if getattr(optimizer, '_warm_start_mu', False):
        # Reuse the DTI warm-start from the user's optimizer
        x0 = optimizer.make_x0(sample_data, dtype=dtype)

    return x0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_convergence(
        optimizer,
        data_flat,
        scheme=None,
        sampler='auto',
        n_voxels=1000,
        target_mae=0.01,
        reference_iter=200,
        test_iters=(3, 5, 10, 20, 50, 100),
        dtype=None):
    """Data-driven convergence analysis for a JAX dMRI optimizer.

    Fits a reference solution then sweeps ``test_iters``, measuring
    normalised MAE per signal-complexity stratum.  Returns the fastest
    config that achieves MAE ≤ ``target_mae`` in the hardest stratum.

    Parameters
    ----------
    optimizer : JaxOptimizer
        Already-constructed optimizer (model + scheme + bounds).
    data_flat : (N_vox, N_meas) float32
        Normalised signal (S / S0), mask-flattened.
    scheme : acquisition scheme or None
        Defaults to ``optimizer.acquisition_scheme``.
    sampler : 'auto' | 'brain' | 'sdm' | sampler instance
        Voxel selection strategy:

        'auto'   → 'brain' for WM models (_mu param), else 'sdm'
        'brain'  → ThreeTissueSampler: WM/GM labels, JAX DTI FA, no dipy
        'sdm'    → SDMStratifiedSampler: generic, works on any organ/tissue
        instance → SDMStratifiedSampler, ThreeTissueSampler, or MaskSampler

    n_voxels : int
        Target sample size.  Default 1000.
    target_mae : float
        Normalised MAE threshold (per-parameter, scaled by bounds width).
        Default 0.01.  Roughly: 1 % of the parameter range.
    reference_iter : int
        Iteration count for the ground-truth reference fit.  Default 200.
    test_iters : sequence of int
        L-BFGS-B iteration counts to evaluate.
    dtype : jnp dtype or None
        Computation dtype.  None → float32.

    Returns
    -------
    report : ConvergenceReport
        Has ``.summary()``, ``.plot()``, and ``.recommended`` (dict of
        kwargs ready to pass to JaxOptimizer).

    Notes
    -----
    Each entry in ``test_iters`` plus the reference requires one JIT
    compilation pass (~10–30 s each on first run).  Subsequent calls with
    the same model/scheme reuse the compiled functions.
    """
    from .vmap_fit import build_vmap_fitter
    from .optimizers_jax import JaxOptimizer

    if scheme is None:
        scheme = optimizer.acquisition_scheme
    if dtype is None:
        dtype = jnp.float32

    data_flat = np.asarray(data_flat, dtype=np.float32)
    affinity  = _detect_model_affinity(optimizer.model)

    # --- Resolve sampler ---
    if isinstance(sampler, str):
        if sampler == 'auto':
            sampler = 'brain' if affinity == 'wm' else 'sdm'
        if sampler == 'brain':
            sampler = ThreeTissueSampler(n_voxels=n_voxels)
        elif sampler == 'sdm':
            sampler = SDMStratifiedSampler(n_voxels=n_voxels)
        else:
            raise ValueError(
                f"Unknown sampler {sampler!r}. "
                f"Use 'auto', 'brain', 'sdm', or a sampler instance.")

    print("Sampling representative voxels...", flush=True)
    indices, labels, _ = sampler.sample(data_flat, scheme)
    sample_data = jnp.array(data_flat[indices], dtype=dtype)
    n_sample    = len(indices)

    stratum_names = sampler.stratum_names
    n_per_stratum = {
        s: int((labels == i).sum())
        for i, s in enumerate(stratum_names)
    }
    print(f"  Sample: {n_sample} voxels — "
          + ", ".join(f"{s}={n_per_stratum[s]}" for s in stratum_names
                      if n_per_stratum[s] > 0))

    # Normalisation: MAE relative to parameter range
    lower = np.array(optimizer._lower)
    upper = np.array(optimizer._upper)
    scale = np.maximum(upper - lower, 1e-12)

    # x0 computed once and reused across all configs (fair comparison)
    x0 = _build_x0(optimizer, sample_data, dtype)

    def _run_config(maxiter):
        opt = JaxOptimizer(optimizer.model, scheme, maxiter=maxiter)
        fn  = build_vmap_fitter(opt, dtype=dtype)
        t0  = time.perf_counter()
        res = np.array(fn(x0, sample_data))
        elapsed = time.perf_counter() - t0
        return res, elapsed / n_sample

    # --- Reference fit ---
    print(f"Fitting reference (maxiter={reference_iter}, n={n_sample})...",
          flush=True)
    params_ref, _ = _run_config(reference_iter)

    # --- Sweep ---
    results, timing = {}, {}
    for n_iter in test_iters:
        label = f'maxiter={n_iter}'
        print(f"  Testing {label}...", flush=True, end='  ')
        params_test, sec_per_vox = _run_config(n_iter)

        norm_err = np.abs(params_test - params_ref) / scale[None]
        strata_mae = {}
        for i, s in enumerate(stratum_names):
            mask_s = labels == i
            if mask_s.sum() > 0:
                strata_mae[s] = float(norm_err[mask_s].mean())

        hard_mae = strata_mae.get(stratum_names[0], float('nan'))
        print(f"MAE({stratum_names[0]})={hard_mae:.4f}  "
              f"{sec_per_vox * 1e6:.1f} µs/vox")

        results[label] = strata_mae
        timing[label]  = sec_per_vox

    return ConvergenceReport(
        results=results,
        timing=timing,
        target_mae=target_mae,
        stratum_names=stratum_names,
        n_per_stratum=n_per_stratum,
        reference_iter=reference_iter,
        model_affinity=affinity,
    )
