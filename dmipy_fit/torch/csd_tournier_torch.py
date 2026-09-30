"""The Tournier 2007 constrained spherical deconvolution, batched over voxels in PyTorch (``solver='csd_tournier07_torch'``,
dmipy-fit#37): the iteration of :class:`~dmipy_fit.optimizers_fod.csd_tournier_batch.CsdTournierBatch` as one
batched loop with a per-voxel ``done`` mask -- a converged voxel's coefficients are frozen while the others iterate,
the loop ends when every voxel is done or at ``max_iter`` -- and one batched Cholesky solve (:func:`spd_solve`) of
the systems per iteration.
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

__all__ = ['CsdTournierTorchOptimizer', 'cholesky_solve', 'spd_solve']


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


def cholesky_solve(Lc, rhs):
    """``M^{-1} rhs`` from the lower Cholesky factor ``Lc (b, n, n)`` of ``M`` and ``rhs (b, n, k)``: two triangular
    solves in float32, ``torch.cholesky_solve`` in float64, the faster of the two for each dtype on the L40S (at
    90,000 x 47 x 47: 2.5 against 6.3 ms in float32, 12.6 against 20.4 ms in float64)."""
    import torch
    if Lc.dtype == torch.float32:
        return torch.linalg.solve_triangular(
            Lc.transpose(1, 2), torch.linalg.solve_triangular(Lc, rhs, upper=False), upper=True)
    return torch.cholesky_solve(rhs, Lc)


def spd_solve(M, rhs):
    """``M^{-1} rhs`` for a batch of symmetric positive definite ``M (b, n, n)`` given by its lower triangle:
    :func:`cholesky_solve` of the Cholesky factorisation; a system whose factorisation fails is solved by LU."""
    import torch
    Lc, info = torch.linalg.cholesky_ex(M)
    x = cholesky_solve(Lc, rhs)
    bad = info != 0
    if bool(bad.any()):
        Mb = torch.tril(M[bad])
        x[bad] = torch.linalg.solve(Mb + torch.tril(Mb, -1).transpose(1, 2), rhs[bad])
    return x


def solve_batch(signals, A, ATA_reg, L, P_init, lambda_pos, tau, *, unity_constraint, max_iter):
    """``(f (b, n_coef), iterations (b,))`` for ``signals (b, n_meas)``, every argument a torch tensor of one dtype on
    one device: the Tournier iteration for the whole batch, each voxel stopping when its active set is stable."""
    import torch
    b_, n_coef = signals.shape[0], ATA_reg.shape[0]; n_init = P_init.shape[0]
    jac = torch.tensor(SPHERE_JACOBIAN, dtype=signals.dtype, device=signals.device)
    il = torch.tril_indices(n_coef, n_coef, device=signals.device)
    low = il[0] * n_coef + il[1]                                      # the lower triangle, flat
    with full_precision(), torch.no_grad():
        T = L[:, il[0]] * L[:, il[1]]                                 # (n_pos, n_low): the rows' outer products
        Q_low = ATA_reg[il[0], il[1]]
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
            Q = torch.zeros((len(idx), n_coef * n_coef), dtype=signals.dtype, device=signals.device)
            # the lower triangle of A'A + lambda_lb R + lambda_pos L' diag(neg) L
            Q[:, low] = lambda_pos * (neg[idx].to(signals.dtype) @ T) + Q_low
            f_new = spd_solve(Q.view(-1, n_coef, n_coef), rhs[idx][:, :, None])[:, :, 0]
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
    torch device (the current CUDA device when one exists, else the CPU). Voxels go to the device in chunks of
    16384 (``DEFAULT_BATCH``; 4096 took 1.45x as long for 32,000 order-8 voxels on the L40S)."""
    DEFAULT_BATCH = 16384

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
