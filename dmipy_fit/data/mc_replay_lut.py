"""Lean, interpolation-fast *fit form* of a replay-pack family (Tier 3).

For **single diffusion encoding** (one gradient direction per measurement, e.g. standard PGSE) at fixed
pulse timing (delta, Delta), a restricted pore's signal factorises: it depends only on the per-gradient
``(b, angle-to-pore-axis)`` and the pore ``diameter`` — not on the full waveform. We therefore precompute
a small kernel

    E_kernel[diameter, b, cos_theta]                    (cylinder: axial symmetry)
    E_kernel[diameter, b]                               (sphere: isotropic)

by replaying the family once (via the Tier-1 compiled engine) over a grid of single-direction PGSE. The
full walker ensemble (tens of MB per pack) is thereby *lowered* to a KB-scale array for that acquisition
family. Fitting then evaluates each measurement by 2-D/1-D interpolation of the kernel at its
``(b_m, cos_theta_m)`` — microseconds per voxel, differentiable in ``(diameter, mu)`` — instead of a
per-voxel matmul over 20 000 walkers.

Accuracy is **interpolation-grade** (this is a fit accelerator, not the reference — Tier 1/2 are exact):
at the grid densities used in practice the error vs the exact replay is ~1e-3 (sphere), ~1e-2 (cylinder,
dominated by the angular ``cos`` interpolation — halves at 2x ``cos_grid`` density), and the ``rho`` axis
is exact on-grid but coarser off-grid (surface enters as ``exp((rho/D) s)``, nonlinear in ``rho``, so use
a fine ``rho_grid`` or a fixed ``rho``). The signed signal is interpolated (magnitude only at the end) so
diffraction zeros do not inflate.

Scope: the PGSE family at the built ``(delta, Delta)``. Arbitrary waveforms (OGSE, b-tensor, multi-timing)
are NOT covered by a single kernel — use the full replay engine (Tier 1/2) there. An optional ``rho`` axis
carries surface relaxivity.
"""
import numpy as np

from ..core.constants import CONSTANTS
from ..signal_models._replay_fit import compile_scheme, replay_complex

_GAMMA = CONSTANTS["water_gyromagnetic_ratio"]

__all__ = ["build_pgse_kernel", "ReplayKernel"]


def _pgse_dirs(cos_grid):
    """Unit gradient directions at angle theta (cos_grid) to the z-axis, in the x–z plane."""
    c = np.asarray(cos_grid, float)
    s = np.sqrt(np.clip(1 - c ** 2, 0, None))
    return np.stack([s, np.zeros_like(c), c], axis=1)          # (n_cos, 3)


def build_pgse_kernel(family, delta, Delta, b_grid, cos_grid=None, rho_grid=(0.0,), n_t_scheme=1024):
    """Precompute the PGSE kernel LUT for a :class:`ReplayFamily` at fixed (delta, Delta).

    Returns a :class:`ReplayKernel`. ``b_grid`` [s/m^2]; ``cos_grid`` in [0,1] (cylinder only; ignored for
    sphere); ``rho_grid`` surface relaxivities [m/s] (default just 0). Built with the exact Tier-1 engine."""
    from ..data.mc_replay import resample_waveform_to_grid
    from ..core.acquisition_scheme import AcquisitionScheme
    shape = family.shape
    b_grid = np.asarray(b_grid, float)
    cos_grid = np.array([1.0]) if shape == "sphere" else np.asarray(cos_grid, float)
    rho_grid = np.asarray(rho_grid, float)
    dirs = _pgse_dirs(cos_grid)                                # (n_cos, 3)

    # Build the grid of single-direction PGSE waveforms via the SAME from_pgse path the models use, so
    # the kernel's b/timing physics is identical to a fit's scheme (avoids a b-axis miscalibration).
    bvals, gds = [], []
    for d in dirs:
        for b in b_grid:
            bvals.append(b); gds.append(d)
    sch = AcquisitionScheme.from_pgse(np.asarray(bvals), np.asarray(gds), delta, Delta, n_t=n_t_scheme)
    Gp = resample_waveform_to_grid(sch._G, float(sch._dt), family.n_t, family.dt)
    W = compile_scheme(Gp, family.dt, family.K, _GAMMA)         # (3K, n_cos*n_b) compiled once

    # E_kernel[diameter, rho, cos, b]. Store the SIGNED (real) signal, not the magnitude: PGSE signal is
    # real and can cross zero (diffraction lobes); interpolating the magnitude across diameter would
    # over-estimate near a zero, whereas the full replay interpolates the complex signal then takes abs.
    E = np.zeros((len(family.diameters), len(rho_grid), len(cos_grid), len(b_grid)))
    for di in range(len(family.diameters)):
        C, w, K, blt = family._pk[di]
        for ri, rho in enumerate(rho_grid):
            rod = (rho / family.diffusivity) if rho else 0.0
            S = replay_complex(C, w, W, blt_dct=blt, rho_over_D=rod, n_t=family.n_t).real
            E[di, ri] = S.reshape(len(cos_grid), len(b_grid))
    return ReplayKernel(shape, family.diffusivity, np.asarray(family.diameters, float),
                        rho_grid, cos_grid, b_grid, float(delta), float(Delta), E)


class ReplayKernel:
    """KB-scale interpolable fit form: ``E(diameter, b, cos_theta[, rho])`` for a PGSE (delta, Delta)
    family. ``signal(scheme, diameter, mu, rho)`` evaluates a full scheme by per-measurement interpolation."""

    def __init__(self, shape, diffusivity, diameters, rho_grid, cos_grid, b_grid, delta, Delta, E):
        self.shape = shape; self.diffusivity = float(diffusivity)
        self.diameters = diameters; self.rho_grid = rho_grid
        self.cos_grid = cos_grid; self.b_grid = b_grid
        self.delta = delta; self.Delta = Delta
        self.E = E                                             # (n_d, n_rho, n_cos, n_b)

    @property
    def nbytes(self):
        return self.E.nbytes

    def _interp1(self, x, xs, ys):
        return np.interp(np.clip(x, xs[0], xs[-1]), xs, ys)

    def signal(self, bvalues, gradient_directions, diameter, mu=None, rho=0.0):
        """Interpolate E for a PGSE scheme: per measurement look up (b_m, cos_theta_m); interpolate over
        diameter (and rho). ``mu`` is the pore axis (cylinder); ignored for sphere."""
        b = np.asarray(bvalues, float)
        n = np.asarray(gradient_directions, float)
        # diameter bracket
        d = float(np.clip(diameter, self.diameters[0], self.diameters[-1]))
        j = int(np.clip(np.searchsorted(self.diameters, d), 1, len(self.diameters) - 1))
        wlo = (self.diameters[j] - d) / (self.diameters[j] - self.diameters[j - 1])
        # rho slice (nearest/interp on the rho axis)
        Er = self.E if len(self.rho_grid) == 1 else None
        def slab(di):
            if len(self.rho_grid) == 1:
                return self.E[di, 0]
            ri = np.interp(rho, self.rho_grid, np.arange(len(self.rho_grid)))
            r0 = int(np.floor(ri)); r1 = min(r0 + 1, len(self.rho_grid) - 1); f = ri - r0
            return (1 - f) * self.E[di, r0] + f * self.E[di, r1]      # (n_cos, n_b)
        K_lo, K_hi = slab(j - 1), slab(j)
        if self.shape == "sphere":
            klo = np.interp(b, self.b_grid, K_lo[0]); khi = np.interp(b, self.b_grid, K_hi[0])
        else:
            mu_c = mu if mu is not None and np.ndim(mu) == 1 and len(mu) == 3 else _mu_cart(mu)
            cos = np.abs(n @ mu_c)                              # angle to pore axis per measurement
            klo = _interp2(self.cos_grid, self.b_grid, K_lo, cos, b)
            khi = _interp2(self.cos_grid, self.b_grid, K_hi, cos, b)
        # interpolate the SIGNED signal across diameter, then take magnitude (matches full replay's
        # complex-interpolate-then-abs, so diffraction zeros don't inflate)
        return np.abs(wlo * klo + (1 - wlo) * khi)


def _mu_cart(mu):
    from ..utils import utils
    return utils.unitsphere2cart_1d(np.asarray(mu, float))


def _interp2(cos_grid, b_grid, K, cos, b):
    """Bilinear interp of K[cos, b] at per-measurement (cos_m, b_m)."""
    ci = np.interp(cos, cos_grid, np.arange(len(cos_grid)))
    bi = np.interp(b, b_grid, np.arange(len(b_grid)))
    c0 = np.clip(np.floor(ci).astype(int), 0, len(cos_grid) - 2); fc = ci - c0
    b0 = np.clip(np.floor(bi).astype(int), 0, len(b_grid) - 2); fb = bi - b0
    out = ((1 - fc) * (1 - fb) * K[c0, b0] + (1 - fc) * fb * K[c0, b0 + 1]
           + fc * (1 - fb) * K[c0 + 1, b0] + fc * fb * K[c0 + 1, b0 + 1])
    return out
