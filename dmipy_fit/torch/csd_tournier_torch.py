"""The Tournier 2007 constrained spherical deconvolution, batched over voxels in PyTorch (``solver='csd_tournier07_torch'``,
dmipy-fit#37): the iteration of :class:`~dmipy_fit.optimizers_fod.csd_tournier_batch.CsdTournierBatch` as one
batched loop with a per-voxel ``done`` mask -- a converged voxel's coefficients are frozen while the others iterate,
the loop ends when every voxel is done or at ``max_iter`` -- and ``torch.linalg.solve`` on the batch of systems.
The active-set path and the iteration count of every voxel are the JAX solver's.

TF32 is off inside the solve (``torch.backends.cuda.matmul.allow_tf32`` and ``cudnn.allow_tf32`` False for the call,
restored after): a float32 matmul on a CUDA device is TF32 by default, and the JAX solver needed full precision to
match the float64 reference (2e-6 there, 5e-4 and 18 % of voxels on another active-set path with TF32). Float32 on
the device by default; float64 takes no flag. The device is the current CUDA device when one exists, else the CPU,
or ``device=``.
"""
import contextlib

import numpy as np

from ..optimizers_fod.csd_tournier_batch import CsdTournierBatch, SPHERE_JACOBIAN

__all__ = ['CsdTournierTorchOptimizer']


@contextlib.contextmanager
def full_precision():
    """Every float32 product at full precision for the block: TF32 off on matmul and cudnn, restored after."""
    import torch
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m; torch.backends.cudnn.allow_tf32 = c


def solve_batch(signals, A, ATA_reg, L, P_init, lambda_pos, tau, *, unity_constraint, max_iter):
    """``(f (b, n_coef), iterations (b,))`` for ``signals (b, n_meas)``, every argument a torch tensor of one dtype on
    one device: the Tournier iteration for the whole batch, each voxel stopping when its active set is stable."""
    import torch
    b_, n_coef = signals.shape[0], ATA_reg.shape[0]; n_init = P_init.shape[0]
    jac = torch.tensor(SPHERE_JACOBIAN, dtype=signals.dtype, device=signals.device)
    with full_precision(), torch.no_grad():
        rhs = signals @ A                                             # (b, n_coef): A' s per voxel
        f = torch.zeros((b_, n_coef), dtype=signals.dtype, device=signals.device)
        f[:, :n_init] = signals @ P_init.T
        if unity_constraint:
            f[:, 0] = jac
        thr = tau * L[0, 0] * f[:, 0]                                  # (b,)
        neg = (f @ L.T) < thr[:, None]                                 # (b, n_pos)
        done = torch.zeros(b_, dtype=torch.bool, device=signals.device)
        it = torch.zeros(b_, dtype=torch.int32, device=signals.device)
        for _ in range(max_iter):
            idx = torch.nonzero(~done, as_tuple=True)[0]
            if idx.numel() == 0:
                break
            neg_a = neg[idx].to(signals.dtype)                         # (a, n_pos)
            Q = ATA_reg[None] + lambda_pos * ((L.T[None] * neg_a[:, None, :]) @ L)   # A'A + lambda_lb R + lambda_pos L' diag(neg) L
            f_new = torch.linalg.solve(Q, rhs[idx][:, :, None])[:, :, 0]
            if unity_constraint:
                f_new[:, 0] = jac
            neg_new = (f_new @ L.T) < thr[idx][:, None]
            f[idx] = f_new; it[idx] += 1
            done[idx] = (neg_new == neg[idx]).all(1)
            neg[idx] = neg_new
    return f, it


class CsdTournierTorchOptimizer(CsdTournierBatch):
    """The batched Tournier 2007 CSD in PyTorch (``solver='csd_tournier07_torch'``); the constructor and
    ``fit_batch`` are :class:`~dmipy_fit.optimizers_fod.csd_tournier_batch.CsdTournierBatch`'s, ``device`` the
    torch device (the current CUDA device when one exists, else the CPU)."""

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8,
                 lambda_pos=1., lambda_lb=5e-4, tau=0.1, max_iter=50,
                 unity_constraint=True, init_sh_order=4, dtype=np.float32, device=None):
        import torch
        super().__init__(acquisition_scheme, model, x0_vector, sh_order, lambda_pos, lambda_lb, tau, max_iter,
                         unity_constraint, init_sh_order, dtype)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        td = {np.dtype(np.float32): torch.float32, np.dtype(np.float64): torch.float64}[self.dtype]
        self._tdtype = td
        put = lambda a: torch.as_tensor(np.asarray(a, self.dtype), dtype=td, device=self.device)
        self._args = tuple(put(a) for a in (self.A, self.ATA_reg, self.L_positivity, self.P_init))
        self._scalars = (put(self.lambda_pos), put(self.tau))

    def _solve_chunk(self, signals):
        import torch
        s = torch.as_tensor(np.asarray(signals, self.dtype), dtype=self._tdtype, device=self.device)
        f, it = solve_batch(s, *self._args, *self._scalars, unity_constraint=self.unity_constraint, max_iter=self.max_iter)
        return f.cpu().numpy(), it.cpu().numpy()
