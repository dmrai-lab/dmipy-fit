"""Multi-shell multi-tissue constrained spherical deconvolution (Jeurissen et al. 2014), batched over voxels in
PyTorch (``solver='csd_msmt_torch'``): one anisotropic kernel (the white-matter FOD, SH order ``sh_order``) and any
number of isotropic kernels (one coefficient each), the problem of
:class:`~dmipy_fit.optimizers_fod.csd_cvxpy.CsdCvxpyOptimizer` solved for every voxel of an image at once::

    minimise   || A x - s ||^2 + lambda_lb x' R x
    subject to L x_wm >= 0                  (the FOD on the 362 positivity directions)
               x_k >= 0                     (the fraction coefficient of every kernel)
               sum_k x_k = 1 / (2 sqrt(pi)) (with ``unity_constraint``: the fractions sum to one)

``A`` is the model's convolution kernel with the tissues' ``S0_responses`` folded in (``fit()`` divides the data by
the largest S0 response), so the fractions ``x_k 2 sqrt(pi)`` are geometric. The output is the cvxpy solver's: the
FOD coefficients with the first held at ``1 / (2 sqrt(pi))``, and the ``partial_volume_*``.

The solver is a primal-dual interior-point method, the same fixed-size step for every voxel: the barrier weights
``w = z / s`` of all constraints enter one ``n x n`` system per voxel, ``(Q + G' diag(w) G) dx = r``
(``Q = A'A + lambda_lb R``, ``G`` the stacked constraints), formed for the batch by one matrix product with the table
of the constraints' outer products and factorised by one batched Cholesky per iteration; the factor serves
Mehrotra's predictor and corrector and one Gondzio centrality corrector (three back-substitutions). The start is
Mehrotra's. A voxel stops when its KKT residuals (relative) and its complementarity are below ``tol``; converged
voxels are frozen while the others iterate (the per-voxel ``done`` mask of
:func:`~dmipy_fit.torch.csd_tournier_torch.solve_batch`), up to ``max_iter`` iterations.

An active-set iteration (Tournier's, generalised to the isotropic coefficients, the constraints as a penalty of
weight 1e2 to 1e8) does not settle on this problem: on 100 synthetic voxels it hit the 50-iteration cap (every voxel
from weight 1e4) and ended 0.06 to 0.43 from the constrained optimum, its active set cycling on the degenerate
positivity set of an MSMT FOD; an augmented-Lagrangian form of it, exact at convergence, took 80 to 140 iterations
on average. The interior-point path has no active set to identify. The problem is flat (an objective 5e-9 above the
optimum, where CLARABEL stopped on 21 of 800 BATMAN voxels, left coefficients 5e-5 away), so the stopping tolerance
is tight.

The first ``warm_iter`` iterations run in float32 (the barrier is still far from the boundary and the systems are
well conditioned), the rest in ``dtype`` (float64 by default): near the solution the barrier weights span many
orders of magnitude and a float32 interior-point run stalls 1e-3 from the optimum. TF32 is off throughout
(:func:`~dmipy_fit.torch.csd_tournier_torch.full_precision`). The device tensors of a problem are cached by value:
the scheme's b-values, directions, delta, Delta and TE, the order, the kernel matrix (the kernels and the S0
responses), the constraints, the dtype and the device.

The defaults ``max_iter=22``, ``warm_iter=14``, ``tol=1e-13`` come from the accuracy curve against cvxpy (CLARABEL,
tolerances 1e-14) on 500 voxels of the BATMAN synthetic brain (dmipy-fit#39): the largest coefficient difference
where cvxpy reached the optimum is 9.2e-4 at 16 iterations, 2.0e-4 at 18, 1.4e-5 at 20 and 6.0e-6 at 22, 25 and 30.
"""
import functools
import hashlib
import os

import numpy as np

from ..utils.sh_basis import positivity_basis, sh_degrees
from .csd_tournier_torch import cholesky_solve, full_precision

__all__ = ['CsdMsmtTorchOptimizer', 'solve_batch']

SPHERE_JACOBIAN = 2 * np.sqrt(np.pi)
STEP_TO_BOUNDARY = 0.99
COMPACT_BELOW = 0.5


def _scheme_key(scheme):
    h = hashlib.sha1()
    for name in ('bvalues', 'gradient_directions', 'delta', 'Delta', 'TE'):
        v = getattr(scheme, name, None)
        h.update(name.encode())
        if v is not None:
            h.update(np.ascontiguousarray(v, float).tobytes())
    return h.hexdigest()


@functools.lru_cache(maxsize=16)
def _device_problem(key, A_bytes, n_meas, n_coef, sh_start, vf_indices, lambda_lb, unity, dtype_name, device_name):
    """The device tensors of one problem, in ``dtype_name`` and in float32: ``A``, ``Q``, ``G``, the outer-product
    table ``T`` of the rows of ``G`` and ``Q``, both on the lower triangle (flat indices ``low``), and the unity row
    ``e``."""
    import torch
    A = np.frombuffer(A_bytes, float).reshape(n_meas, -1)
    n = A.shape[1]
    L = positivity_basis(_order_of(n_coef))
    R = np.zeros(n)
    R[sh_start:sh_start + n_coef] = (lambda l: l ** 2 * (l + 1) ** 2)(sh_degrees(_order_of(n_coef)))
    Q = A.T @ A + lambda_lb * np.diag(R)
    G = np.zeros((L.shape[0] + len(vf_indices), n))
    G[:L.shape[0], sh_start:sh_start + n_coef] = L
    G[np.arange(L.shape[0], G.shape[0]), list(vf_indices)] = 1.0
    il = np.tril_indices(n)
    T = G[:, il[0]] * G[:, il[1]]                                                  # (m, n (n + 1) / 2)
    e = np.zeros(n)
    e[list(vf_indices)] = 1.0
    dev = torch.device(device_name)
    out = {}
    for dt in {dtype_name, 'float32'}:
        tdt = getattr(torch, dt)
        put = lambda a: torch.as_tensor(np.asarray(a), dtype=tdt, device=dev)
        out[dt] = dict(A=put(A), Q=put(Q), G=put(G), T=put(T), e=put(e), Q_low=put(Q[il]))
    out['low'] = torch.as_tensor(il[0] * n + il[1], device=dev)
    return out


def _order_of(n_coef):
    order = int(round((-3 + np.sqrt(1 + 8 * n_coef)) / 2))
    if (order + 1) * (order + 2) // 2 != n_coef:
        raise ValueError("{} is not the coefficient count of an even SH order".format(n_coef))
    return order


def _normal_matrix(w, c, low):
    """The lower triangle of ``Q + G' diag(w) G`` for every voxel, ``(b, n, n)``, from the barrier weights
    ``w (b, m)`` (the Cholesky factorisation reads the lower triangle only)."""
    import torch
    n = c['Q'].shape[0]
    M = torch.zeros((w.shape[0], n * n), dtype=w.dtype, device=w.device)
    M[:, low] = w @ c['T'] + c['Q_low']
    return M.view(-1, n, n)


def _max_step(v_inv, dv):
    """The largest ``a <= 1`` with ``v + a dv >= 0``, per voxel, from ``1 / v`` (``v > 0``)."""
    import torch
    return 1.0 / torch.clamp((-dv * v_inv).amax(1), min=1.0)


def solve_batch(signals, problem, *, unity, target, max_iter, warm_iter, tol, dtype):
    """``(x (b, n), iterations (b,), converged (b,))`` for ``signals (b, n_meas)`` (in ``dtype``) on the problem's
    device: the interior-point iteration for the whole batch, float32 for ``warm_iter`` iterations and ``dtype``
    after, each voxel frozen once its KKT residuals and complementarity are below ``tol``.

    The start is Mehrotra's (the least-squares ``x``, slacks shifted positive, then both shifted to balance ``s' z``);
    each iteration takes Mehrotra's predictor-corrector direction and one Gondzio centrality corrector, kept for the
    voxels whose step it lengthens. The iteration runs on a working set; a done voxel stays in it with a zero step
    until fewer than :data:`COMPACT_BELOW` of the working set are still iterating, when the set is written back and
    shrunk to them."""
    import torch
    low = problem['low']
    b_, dev = signals.shape[0], signals.device
    with full_precision(), torch.no_grad():
        c = problem[str(dtype).split('.')[-1]]
        rhs_full = signals.to(dtype) @ c['A']                                      # (b, n): A' s
        cQ = torch.linalg.cholesky(c['Q'])
        xw = torch.cholesky_solve(rhs_full.T, cQ).T
        m, n = c['G'].shape
        if unity:
            v0 = torch.cholesky_solve(c['e'][:, None], cQ)[:, 0]
            xw = xw - ((xw @ c['e'] - target) / (c['e'] @ v0))[:, None] * v0[None]
        Gx = xw @ c['G'].T
        sw = Gx + torch.clamp(-1.5 * Gx.amin(1, keepdim=True), min=0.0)
        sw = sw + 1e-12                                     # a zero signal gives Gx = 0 everywhere
        zw = torch.ones_like(sw)
        sz = (sw * zw).sum(1, keepdim=True)
        sw, zw = sw + 0.5 * sz / zw.sum(1, keepdim=True), zw + 0.5 * sz / sw.sum(1, keepdim=True)
        ew = torch.zeros(b_, dtype=dtype, device=dev)
        x = torch.empty_like(xw)                                                    # the result
        done = torch.zeros(b_, dtype=torch.bool, device=dev)
        it = torch.zeros(b_, dtype=torch.int32, device=dev)

        W = torch.arange(b_, device=dev)                                            # the working set
        bw, dw = rhs_full, done.clone()
        for k in range(max_iter):
            dtk = torch.float32 if k < warm_iter else dtype
            if dtk != xw.dtype or k == 0:
                c = problem[str(dtk).split('.')[-1]]
                xw, sw, zw, ew = (t.to(dtk) for t in (xw, sw, zw, ew))
                bw = rhs_full[W].to(dtk)
            G, e = c['G'], c['e']
            Qx, Gz, Gx = xw @ c['Q'], zw @ G, xw @ G.T
            rd = Qx - bw - Gz
            if unity:
                rd = rd + ew[:, None] * e[None]
                re = xw @ e - target
            rp = Gx - sw
            mu = torch.linalg.vecdot(sw, zw) / m
            inf = float('inf')
            nrm = lambda t: torch.linalg.vector_norm(t, ord=inf, dim=1)
            res = torch.maximum(nrm(rd) / (1 + torch.maximum(torch.maximum(nrm(Qx), nrm(bw)), nrm(Gz))),
                                nrm(rp) / (1 + torch.maximum(nrm(Gx), sw.amax(1))))
            res = torch.maximum(res, mu)
            if unity:
                res = torch.maximum(res, re.abs())
            if k >= warm_iter:                     # only a converged voxel of the final dtype is done
                dw = dw | (res < tol)
            n_active = int((~dw).sum())
            if n_active == 0:
                break
            if n_active < COMPACT_BELOW * len(W):  # write the working set back, keep the voxels still iterating
                x[W], done[W] = xw.to(dtype), dw
                keep = torch.nonzero(~dw, as_tuple=True)[0]
                W = W[keep]
                xw, sw, zw, ew, bw, dw = xw[keep], sw[keep], zw[keep], ew[keep], bw[keep], dw[keep]
                Gx, rd, rp, mu = Gx[keep], rd[keep], rp[keep], mu[keep]
                if unity:
                    re = re[keep]
            inv_s, inv_z = 1.0 / sw, 1.0 / zw
            w = zw * inv_s
            Lc, info = torch.linalg.cholesky_ex(_normal_matrix(w, c, low))
            if unity:
                v = cholesky_solve(Lc, e[None, :, None].expand(len(W), n, 1).contiguous())[:, :, 0]
                ev = v @ e
            zGx = zw * Gx
            ok = (info == 0) & ~dw

            def direction(rc):
                """The Newton step ``(dx, ds, dz, d_eta)`` for the complementarity target ``rc``; ``(z ds + s dz
                = rc - s z``, the dual and primal residuals ``rd``, ``rp`` removed); zero for a voxel whose
                factorisation failed."""
                r = (rc - zGx) * inv_s
                u = cholesky_solve(Lc, (r @ G - rd)[:, :, None])[:, :, 0]
                if unity:
                    de = (u @ e + re) / ev
                    u = u - de[:, None] * v
                else:
                    de = None
                dx = torch.where(ok[:, None] & torch.isfinite(u), u, torch.zeros_like(u))
                Gdx = dx @ G.T
                return dx, Gdx + rp, torch.addcmul(r, w, Gdx, value=-1.0), de

            def step(ds, dz):
                return torch.minimum(_max_step(inv_s, ds), _max_step(inv_z, dz))

            dxa, dsa, dza, _ = direction(torch.zeros_like(sw))
            aa = step(dsa, dza)
            P = dsa * dza
            mua = mu * (1 - aa) + aa ** 2 * P.sum(1) / m                 # (s + a ds)'(z + a dz) / m
            smu = ((mua / mu) ** 3 * mu)[:, None]
            rc = smu - P
            dx, ds, dz, de = direction(rc)
            # one Gondzio corrector: push the trial point's products s z into [0.1, 10] sigma mu
            ap = step(ds, dz)
            at = torch.clamp(1.5 * ap + 0.1, max=1.0)[:, None]
            prod = torch.addcmul(sw, at, ds) * torch.addcmul(zw, at, dz)
            corr = torch.clamp(0.1 * smu - prod, min=0.0) - torch.clamp(prod - 10.0 * smu, min=0.0)
            dx2, ds2, dz2, de2 = direction(rc + torch.clamp(corr, min=-10.0 * smu))
            a2 = step(ds2, dz2)
            better = (a2 >= 1.01 * ap)[:, None]
            dx, ds, dz = torch.where(better, dx2, dx), torch.where(better, ds2, ds), torch.where(better, dz2, dz)
            if unity:
                de = torch.where(better[:, 0], de2, de)
            a = torch.where(ok, STEP_TO_BOUNDARY * torch.where(better[:, 0], a2, ap), torch.zeros_like(ap))[:, None]
            xw = torch.addcmul(xw, a, dx)
            sw = torch.addcmul(sw, a, ds)
            zw = torch.addcmul(zw, a, dz)
            if unity:
                ew = ew + a[:, 0] * torch.where(ok, de, torch.zeros_like(de))
            it[W] += (~dw).to(it.dtype)
        x[W], done[W] = xw.to(dtype), dw
    return x, it, done


class CsdMsmtTorchOptimizer:
    """The batched multi-tissue CSD in PyTorch (``solver='csd_msmt_torch'``).

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme
    model : MultiCompartmentSphericalHarmonicsModel
        One anisotropic and any number of isotropic kernels, the kernels fixed, every volume fraction estimated;
        ``S0_responses`` set (``fit()`` sets it).
    x0_vector : array
        The parameter vector the kernel is built from (all-NaN for a fixed kernel).
    sh_order : int
        The FOD's SH order (default 8, 45 coefficients).
    unity_constraint : bool
        The fractions sum to one.
    lambda_lb : float
        Laplace-Beltrami smoothness weight on the FOD (``fit()`` passes its own).
    max_iter : int
        Interior-point iterations per voxel at most (default 22, from the accuracy curve in the module docstring).
    warm_iter : int
        The first iterations run in float32 (default 14).
    tol : float
        A voxel is done when its relative KKT residuals and its complementarity are below ``tol`` (default 1e-13,
        the data normalised to the largest S0 response).
    dtype : numpy dtype
        The dtype after the warm phase (default float64).
    device : torch device or None
        The current CUDA device when one exists, else the CPU.
    """
    _citations = {
        'definition': [
            {'key': 'jeurissen2014', 'authors': 'Jeurissen B, Tournier J-D, Dhollander T, Connelly A, Sijbers J',
             'title': 'Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell '
                      'diffusion MRI data',
             'journal': 'NeuroImage', 'year': 2014, 'doi': '10.1016/j.neuroimage.2014.07.061'},
            {'key': 'mehrotra1992', 'authors': 'Mehrotra S',
             'title': 'On the implementation of a primal-dual interior point method',
             'journal': 'SIAM Journal on Optimization', 'year': 1992, 'doi': '10.1137/0802028'},
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'SH_convergence', 'name': 'SH convergence',
         'condition_human': 'max_order must be sufficient for the kernel bandwidth',
         'severity': 'info', 'source_key': 'jeurissen2014'},
    ]

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8, unity_constraint=False,
                 lambda_lb=1e-5, max_iter=22, warm_iter=14, tol=1e-13, dtype=np.float64, device=None):
        import torch
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = int(sh_order)
        self.Ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.unity_constraint = bool(unity_constraint)
        self.lambda_lb = float(lambda_lb)
        self.max_iter = int(max_iter)
        self.warm_iter = int(warm_iter)
        self.tol = float(tol)
        self.dtype = np.dtype(dtype)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tdtype = {np.dtype(np.float32): torch.float32, np.dtype(np.float64): torch.float64}[self.dtype]

        if not hasattr(model, 'volume_fractions_fixed'):
            model._check_if_kernel_parameters_are_fixed()
        if model.volume_fractions_fixed:
            raise ValueError("solver='csd_msmt_torch' estimates the volume fractions; with them fixed (one "
                             "composite kernel) use solver='csd_tournier07_torch'")
        fixed = [n for n in model.partial_volume_names if not model.parameter_optimization_flags.get(n, False)]
        if fixed:
            raise ValueError("solver='csd_msmt_torch' estimates every volume fraction; {} fixed".format(fixed))
        aniso = [m for m in model.models if 'orientation' in m.parameter_types.values()]
        if len(aniso) != 1:
            raise ValueError("solver='csd_msmt_torch' takes one anisotropic kernel, got {}".format(len(aniso)))
        if not hasattr(model, 'S0_responses'):
            raise AttributeError("model.S0_responses must be set before constructing the MSMT optimizer; fit() "
                                 "sets it, or set it manually.")
        x0_single = np.reshape(x0_vector, (-1, np.shape(x0_vector)[-1]))[0]
        if not np.all(np.isnan(x0_single)):
            raise ValueError("the kernel must be fixed (an all-NaN x0 for its parameters); a voxel-varying kernel "
                             "has one system per voxel and is not what this optimizer batches")
        self._x0_single = x0_single

        # the unknowns: the anisotropic model's Ncoef FOD coefficients, one coefficient per isotropic model
        vf, start = [], 0
        for m in model.models:
            if 'orientation' in m.parameter_types.values():
                self.sh_start = start
                vf.append(start)
                start += self.Ncoef
            else:
                vf.append(start)
                start += 1
        self.vf_indices = tuple(vf)
        self.A = np.asarray(model._construct_convolution_kernel(**model.parameter_vector_to_parameters(x0_single)),
                            float)
        if self.A.shape[1] != start:
            raise RuntimeError("the kernel has {} columns, the unknowns are {}".format(self.A.shape[1], start))
        self._sh_slice, self._pv_positions = self._locate_outputs()
        dt_name = self.dtype.name
        self._problem = _device_problem(
            _scheme_key(acquisition_scheme), np.ascontiguousarray(self.A).tobytes(), self.A.shape[0], self.Ncoef,
            self.sh_start, self.vf_indices, self.lambda_lb, self.unity_constraint, dt_name, str(self.device))

    def _locate_outputs(self):
        """The parameter-vector slice of ``sh_coeff`` and the position of every ``partial_volume_*`` (in the
        order of ``model.partial_volume_names``), checked against the model's own assembly."""
        params = self.model.parameter_vector_to_parameters(self._x0_single)
        params['sh_coeff'] = np.arange(1, self.Ncoef + 1, dtype=float)
        names = list(self.model.partial_volume_names)
        for i, name in enumerate(names):
            params[name] = 1000. + i
        vec = np.asarray(self.model.parameters_to_parameter_vector(**params), float)
        start = int(np.flatnonzero(vec == 1.)[0])
        sl = slice(start, start + self.Ncoef)
        if not np.array_equal(vec[sl], params['sh_coeff']):
            raise RuntimeError("the model does not lay sh_coeff out contiguously in its parameter vector")
        return sl, [int(np.flatnonzero(vec == 1000. + i)[0]) for i in range(len(names))]

    def fit_batch(self, data_all, x0_all, eta=None, *, diagnostics=False):
        """Fit every voxel: ``(N_voxels, N_parameters)``, or with ``diagnostics`` also ``{'iter_num': (N_voxels,),
        'converged': (N_voxels,)}``.

        ``data_all`` is ``(N_voxels, N_meas)`` signal normalised as ``fit()`` normalises it (sent to the device in
        its own dtype, float32 data at half the bytes, and cast there); ``x0_all``
        ``(N_voxels, N_parameters)`` carries the other parameters into the result. ``eta`` applies the Rician bias
        correction ``sqrt(max(s^2 - eta^2, 0))`` first. Voxels are solved in chunks of ``DMIPY_CSD_MSMT_BATCH``
        (default 131072).
        """
        import torch
        data_all = np.asarray(data_all)
        if not np.issubdtype(data_all.dtype, np.floating):
            data_all = data_all.astype(float)
        x0_all = np.asarray(x0_all, float)
        if eta is not None and eta > 0:
            data_all = np.sqrt(np.maximum(data_all ** 2 - eta ** 2, 0.0))
        n = data_all.shape[0]
        batch = max(1, int(os.environ.get("DMIPY_CSD_MSMT_BATCH", "131072")))
        X = np.zeros((n, self.A.shape[1]))
        iters = np.zeros(n, int)
        conv = np.zeros(n, bool)
        for s in range(0, n, batch):
            sig = torch.as_tensor(data_all[s:s + batch], device=self.device).to(self._tdtype)   # sent as given
            x, it, done = solve_batch(sig, self._problem, unity=self.unity_constraint,
                                      target=1.0 / SPHERE_JACOBIAN, max_iter=self.max_iter,
                                      warm_iter=self.warm_iter, tol=self.tol, dtype=self._tdtype)
            X[s:s + batch] = x.double().cpu().numpy()
            iters[s:s + batch] = it.cpu().numpy()
            conv[s:s + batch] = done.cpu().numpy()
        out = x0_all.copy()
        sh = X[:, self.sh_start:self.sh_start + self.Ncoef].copy()
        sh[:, 0] = 1.0 / SPHERE_JACOBIAN
        out[:, self._sh_slice] = sh
        for pos, k in zip(self._pv_positions, self.vf_indices):
            out[:, pos] = X[:, k] * SPHERE_JACOBIAN
        if diagnostics:
            return out, {'iter_num': iters, 'converged': conv}
        return out

    def __call__(self, data, x0_vector):
        """One voxel: the fitted parameter vector, as :class:`CsdCvxpyOptimizer` returns it."""
        return self.fit_batch(np.asarray(data)[None], np.asarray(x0_vector)[None])[0]
