"""JAX DTI fitter — GPU-accelerated diffusion tensor estimation.

Replaces the dipy TensorModel dependency for orientation warm-start.

The linear DTI model:
    log(S_dw / S0) ≈ -b * g^T D g

where D is the 3×3 symmetric diffusion tensor, g is the unit gradient
direction, b is the b-value.  Solved per-voxel via the precomputed
pseudoinverse of the design matrix A (N_dw × 6), which is constant
across all voxels for a fixed acquisition scheme.

Returns per-voxel:
    mu_sph : (theta, phi) spherical coordinates of the principal eigenvector
    fa     : fractional anisotropy [0, 1]

Design matrix column order: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_mu_indices(model):
    """Find nested optimizer vector indices occupied by _mu parameters.

    Iterates model.parameter_cardinality in order, counting only
    optimized (flag=True) parameters.

    Parameters
    ----------
    model : MultiCompartmentModel

    Returns
    -------
    mu_list : list of (param_name, start_idx, end_idx) tuples
        One entry per optimized _mu parameter.  For NODDI this is a
        single tuple with end_idx - start_idx == 2 (theta, phi).
        For two-fascicle models there are two entries.
        For orientation-free models (NEXI, KARGER, …) the list is empty.
    """
    mu_list = []
    idx = 0
    for pname, card in model.parameter_cardinality.items():
        if model.parameter_optimization_flags.get(pname, True):
            if pname.lower().endswith('_mu'):
                mu_list.append((pname, idx, idx + card))
            idx += card
    return mu_list


@functools.lru_cache(maxsize=16)
def build_dti_fitter(scheme, b_max=None, dtype=None):
    """Build a JIT+vmap DTI fitter from an acquisition scheme.

    Cached by (scheme, b_max, dtype) — repeated calls with the same arguments
    return the already-compiled fitter at no cost.

    Parameters
    ----------
    scheme : DmipyAcquisitionScheme
        Provides .bvalues (s/m²) and .gradient_directions (N_meas, 3).
    b_max : float or None
        Maximum b-value to include in s/m² (e.g. 1.5e9 for 1500 s/mm²).
        None → use all DW measurements.  For multi-shell acquisitions with
        b > 2000 s/mm², setting b_max=1.5e9 substantially improves FA and
        orientation accuracy because the DTI model only holds at low b.
    dtype : jnp.dtype or None
        Computation dtype.  None → float32.

    Returns
    -------
    fit_dti : callable
        fit_dti(data_batch) -> (mu_sph, fa)

        data_batch : jnp.array (N_vox, N_meas) — normalised signal S/S0
        mu_sph     : jnp.array (N_vox, 2)      — (theta, phi) radians
        fa         : jnp.array (N_vox,)         — FA [0, 1]
    """
    if dtype is None:
        dtype = jnp.float32

    bvals = np.array(scheme.bvalues)             # (N_meas,)  s/m²
    bvecs = np.array(scheme.gradient_directions)  # (N_meas, 3)

    # Use only DW measurements in the valid DTI range.
    # b0 images contribute log(~1)≈0, uninformative.
    # High-b images violate the monoexponential assumption → restrict via b_max.
    dw_mask = bvals > 1e6   # b > 1 s/mm²
    if b_max is not None:
        dw_mask = dw_mask & (bvals <= b_max)
    # Convert b to s/mm² for the design matrix so that tensor elements d are
    # in mm²/s (~1e-3), not SI units m²/s (~1e-9).  This keeps d² ~ 1e-6,
    # well above float32 precision thresholds, and matches dipy's convention.
    b_dw = bvals[dw_mask] * 1e-6   # s/m² → s/mm²
    g_dw = bvecs[dw_mask]
    gx, gy, gz = g_dw[:, 0], g_dw[:, 1], g_dw[:, 2]

    # Build design matrix A: (N_dw, 6)
    # Column order: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz  (units: mm²/s)
    A = np.stack([
        b_dw * gx ** 2,
        b_dw * 2 * gx * gy,
        b_dw * 2 * gx * gz,
        b_dw * gy ** 2,
        b_dw * 2 * gy * gz,
        b_dw * gz ** 2,
    ], axis=1)  # (N_dw, 6)

    # Pseudoinverse computed once in float64 for accuracy, then cast.
    # Shape (6, N_dw) — applied as A_pinv @ y per voxel.
    A_pinv_jax = jnp.array(np.linalg.pinv(A), dtype=dtype)
    dw_idx     = jnp.array(np.where(dw_mask)[0], dtype=jnp.int32)

    # Fixed starting vector for power iteration: [1,1,1]/√3 avoids
    # degeneracy with any single coordinate axis.
    _v0 = jnp.array([1.0, 1.0, 1.0], dtype=dtype) / jnp.sqrt(jnp.array(3.0, dtype=dtype))

    # Split A_pinv into 6 row vectors for GEMV-based regression.
    # A single (N_vox, N_dw) @ (N_dw, 6) GEMM fails when N_vox is small
    # (< 512) because XLA requires M and N to be divisible by GPU tile sizes
    # (~8–16 each), and neither 16 nor 6 satisfies this simultaneously.
    # Six individual (N_vox, N_dw) @ (N_dw,) GEMVs have no such constraint
    # since the contracting dim is N_dw (~350) which is large.
    A_pinv_rows = [A_pinv_jax[k] for k in range(6)]  # list of 6 × (N_dw,)

    def _matvec_D(d, v):
        """Symmetric D (encoded as [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz]) times vector v.

        Fully unrolled scalar arithmetic — avoids any (3,3)@(3,) inside
        vmap, which triggers XLA's minimum-tile-size error.
        """
        return jnp.stack([
            d[0] * v[0] + d[1] * v[1] + d[2] * v[2],
            d[1] * v[0] + d[3] * v[1] + d[4] * v[2],
            d[2] * v[0] + d[4] * v[1] + d[5] * v[2],
        ])

    def _fit_from_d(d):
        """Per-voxel: principal eigenvector + FA from tensor coefficients d.

        d : (6,) — [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] from linear regression.

        All operations are pure scalar arithmetic — no matmul inside vmap.
        """
        # Power iteration: 10 steps converge to < 0.1° for typical DTI.
        v = _v0
        for _ in range(10):
            v = _matvec_D(d, v)
            norm = jnp.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            v = v / jnp.maximum(norm, 1e-10)

        # Enforce upper-hemisphere convention (positive z)
        v = jnp.where(v[2] < 0, -v, v)

        # Spherical coordinates
        theta = jnp.arccos(jnp.clip(v[2], -1.0, 1.0))
        phi   = jnp.arctan2(v[1], v[0])

        # FA from tensor invariants — no eigenvalue decomposition needed.
        # trace(D) = λ1+λ2+λ3
        trD  = d[0] + d[3] + d[5]
        # trace(D²) = Dxx²+Dyy²+Dzz² + 2(Dxy²+Dxz²+Dyz²)
        trD2 = (d[0] ** 2 + d[3] ** 2 + d[5] ** 2
                + 2.0 * (d[1] ** 2 + d[2] ** 2 + d[4] ** 2))
        # FA² = (3/2)(trD2 − trD²/3) / trD2
        fa = jnp.where(
            trD2 > 1e-12,
            jnp.sqrt(jnp.clip(1.5 * (trD2 - trD ** 2 / 3.0) / trD2, 0.0, 1.0)),
            0.0)

        return jnp.stack([theta, phi]), fa

    @jax.jit
    def fit_dti(data_batch):
        """Fit DTI on a batch of voxels.

        Parameters
        ----------
        data_batch : jnp.array (N_vox, N_meas) — normalised signal S/S0

        Returns
        -------
        mu_sph : jnp.array (N_vox, 2) — (theta, phi) radians
        fa     : jnp.array (N_vox,)   — FA [0, 1]
        """
        # Step 1: log-signal and linear regression via 6 GEMVs.
        # Each (N_vox, N_dw) @ (N_dw,) → (N_vox,) has no tile-size constraint.
        data_dw = data_batch[:, dw_idx]
        y_batch = -jnp.log(jnp.clip(data_dw, 1e-6, 1.0))  # (N_vox, N_dw)
        d_batch = jnp.stack(
            [y_batch @ A_pinv_rows[k] for k in range(6)], axis=1
        )  # (N_vox, 6)

        # Step 2: per-voxel power iteration + FA — pure scalar ops, safe in vmap.
        return jax.vmap(_fit_from_d)(d_batch)

    return fit_dti
