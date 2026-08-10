"""Exact matrix / multiple-correlation-function (MCF) restricted-diffusion signal for a plane, cylinder,
and sphere under an *arbitrary* gradient waveform ``G(t)``.

The Bloch-Torrey equation ``dm/dt = D grad^2 m - i gamma (g(t).r) m`` with reflecting (Neumann) walls is
solved in the Laplacian eigenbasis: ``c_{k+1} = exp(-dt (D Lambda + i gamma g_k B)) c_k``, where ``Lambda``
are the eigenvalues and ``B_{mn} = <u_m| x |u_n>`` the (restricted-axis) position operator. The signal is the
ground-state amplitude of the ordered propagator product. This is *exact* up to the number of eigenmodes
kept; the Gaussian-phase models (Van Gelderen, Murday-Cotts) are its low-``b`` limit.

References: Callaghan, J. Magn. Reson. 129, 74 (1997), "generalized gradient waveforms"; Barzykin, J. Magn.
Reson. 139, 342 (1999); Grebenkov, Rev. Mod. Phys. 79, 1077 (2007).

Implementation notes
--------------------
* The eigenvalue roots and the *dimensionless* position matrix are computed once on the unit geometry and
  cached; for a pore of size ``a`` they scale as ``lambda = lambda_unit / a**2`` and ``B = a * B_unit``, so a
  fit that sweeps the diameter never re-quadratures.
* Propagation uses a Strang split with a one-time eigendecomposition of the position matrix (O(n_t N^2) per
  waveform); the Trotter error is O(dt^2) and far below the Monte-Carlo/experimental floor for usual grids.
"""
from functools import lru_cache

import numpy as np
from scipy import special
from scipy.optimize import brentq

__all__ = ["matrix_restricted_signal", "project_fixed_direction", "free_axis_bvalue"]


def project_fixed_direction(G_m, dt):
    """Fixed gradient direction ``d`` (the unit direction at peak |G|) and the signed 1-D magnitude
    schedule ``g(t) = G_m . d`` for a single measurement's waveform ``G_m`` of shape (n_t, 3).
    Exact for fixed-direction waveforms (PGSE, standard OGSE); rotating/b-tensor waveforms are projected
    onto their dominant direction (an approximation for those, flagged in the model docstrings)."""
    G_m = np.asarray(G_m, dtype=np.float64)
    mag = np.linalg.norm(G_m, axis=1)
    k = int(np.argmax(mag))
    if mag[k] == 0.0:
        return np.array([1.0, 0.0, 0.0]), np.zeros(len(G_m))
    d = G_m[k] / mag[k]
    return d, G_m @ d


def pgse_waveform(gradient_strength, delta, Delta, direction, n_t=1024):
    """Reconstruct a rectangular bipolar PGSE waveform (n_t, 3) + dt from scalar timing, for schemes that
    carry only (gradient_strength, delta, Delta) and no stored ``_G``."""
    dt = (Delta + delta) / n_t
    nd = max(1, int(round(delta / dt)))
    ng = int(round(Delta / dt))
    G = np.zeros((n_t, 3))
    u = np.asarray(direction, dtype=np.float64)
    G[:nd] = gradient_strength * u
    G[ng:ng + nd] = -gradient_strength * u
    return G, dt


def free_axis_bvalue(g_schedule, dt, gyromagnetic_ratio):
    """b-value of a 1-D gradient schedule for free (Gaussian) diffusion: ``b = int q(t)^2 dt`` with
    ``q(t) = gamma int_0^t g dt'``. Used for the unrestricted axes of the cylinder/plane."""
    g = np.asarray(g_schedule, dtype=np.float64)
    q = gyromagnetic_ratio * np.cumsum(g) * dt
    return float(np.sum(q * q) * dt)


# ----------------------------- Neumann eigenvalue roots (unit geometry) -----------------------------
def _cyl_roots(m, n_roots):
    """First ``n_roots`` non-negative roots of ``J_m'(x) = 0``; ``m = 0`` includes the x = 0 constant mode."""
    if m == 0:
        return np.concatenate([[0.0], special.jnp_zeros(0, n_roots - 1)])
    return special.jnp_zeros(m, n_roots)


def _sph_roots(l, n_roots):
    """First ``n_roots`` non-negative roots of the spherical-Bessel derivative ``j_l'(x) = 0``."""
    xs = np.linspace(1e-6, 4 * (n_roots + l + 2) + 20, 40000)
    d = special.spherical_jn(l, xs, derivative=True)
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    roots, want = [], (n_roots - 1 if l == 0 else n_roots)
    for i in idx:
        try:
            r = brentq(lambda x: special.spherical_jn(l, x, derivative=True), xs[i], xs[i + 1])
            if not roots or r - roots[-1] > 1e-3:
                roots.append(r)
        except ValueError:
            pass
        if len(roots) >= want:
            break
    roots = np.array(roots)
    return np.concatenate([[0.0], roots]) if l == 0 else roots


# ----------------------------- unit-geometry modes + position matrix -----------------------------
def _plane_modes(n_max, nr):
    x = np.linspace(0.0, 1.0, nr)
    U = [np.cos(n * np.pi * x) for n in range(n_max + 1)]
    lam = np.array([(n * np.pi) ** 2 for n in range(n_max + 1)])
    norm = np.array([np.trapezoid(u * u, x) for u in U])
    N = len(U)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            B[i, j] = np.trapezoid(U[i] * x * U[j], x) / np.sqrt(norm[i] * norm[j])
    return lam, B


def _cyl_modes(m_max, n_max, nr):
    r = np.linspace(0.0, 1.0, nr)
    states, lam, R, Rn = [], [], [], []
    for m in range(m_max + 1):
        for al in _cyl_roots(m, n_max):
            states.append(m); lam.append(al ** 2)
            Rj = special.jv(m, al * r); R.append(Rj); Rn.append(np.trapezoid(Rj * Rj * r, r))
    N = len(states); lam = np.array(lam)
    ang_n = lambda m: (2 * np.pi if m == 0 else np.pi)
    t = np.linspace(0, 2 * np.pi, 2048)
    ang = lambda mp, m: np.trapezoid(np.cos(mp * t) * np.cos(t) * np.cos(m * t), t)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if abs(states[i] - states[j]) != 1:
                continue
            rad = np.trapezoid(R[i] * R[j] * r * r, r)
            B[i, j] = rad * ang(states[i], states[j]) / np.sqrt(Rn[i] * ang_n(states[i]) * Rn[j] * ang_n(states[j]))
    return lam, B


def _sph_modes(l_max, n_max, nr):
    r = np.linspace(0.0, 1.0, nr)
    states, lam, R, Rn = [], [], [], []
    for l in range(l_max + 1):
        for be in _sph_roots(l, n_max):
            states.append(l); lam.append(be ** 2)
            Rj = special.spherical_jn(l, be * r); R.append(Rj); Rn.append(np.trapezoid(Rj * Rj * r * r, r))
    N = len(states); lam = np.array(lam)
    th = np.linspace(0, np.pi, 2048); ct = np.cos(th); st = np.sin(th)
    P = {l: special.eval_legendre(l, ct) for l in range(l_max + 2)}
    ang_n = lambda l: 2 * np.pi * np.trapezoid(P[l] * P[l] * st, th)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if abs(states[i] - states[j]) != 1:
                continue
            rad = np.trapezoid(R[i] * R[j] * r * r * r, r)
            ang = 2 * np.pi * np.trapezoid(P[states[i]] * ct * P[states[j]] * st, th)
            B[i, j] = rad * ang / np.sqrt(Rn[i] * ang_n(states[i]) * Rn[j] * ang_n(states[j]))
    return lam, B


@lru_cache(maxsize=32)
def _unit_modes(shape, n_modes, nr=4000):
    """(lambda_unit, beta, U) for the unit geometry: eigenvalues, and the eigendecomposition of the unit
    position matrix (B = size * U diag(beta) U^T). Cached — computed once per (shape, n_modes)."""
    if shape == "plane":
        lam, B = _plane_modes(n_modes, nr)
    elif shape == "cylinder":
        lam, B = _cyl_modes(n_modes, max(4, n_modes // 2 + 2), nr)
    elif shape == "sphere":
        lam, B = _sph_modes(n_modes, max(4, n_modes // 2 + 2), nr)
    else:
        raise ValueError("shape must be 'plane', 'cylinder' or 'sphere'")
    beta, evecs = np.linalg.eigh(B)
    return lam, beta, evecs


def matrix_restricted_batch(shape, g_axes, dt, diffusivity, size, gyromagnetic_ratio,
                            n_modes=16, use_jax=False):
    """Signal for a stack of projected 1-D schedules ``g_axes`` (n_meas, n_t), all on a common ``dt``.
    ``use_jax=True`` evaluates the differentiable GPU twin (needs the ``[jax]`` extra, uniform ``dt``);
    otherwise a NumPy loop. Returns a length-``n_meas`` array."""
    if use_jax:
        import jax.numpy as jnp
        from ..jax.signal_models_jax import matrix_restricted_signal_jax_batch
        lam, beta, U = _unit_modes(shape, int(n_modes))
        out = matrix_restricted_signal_jax_batch(
            jnp.asarray(g_axes), float(dt), float(diffusivity), float(size),
            float(gyromagnetic_ratio), jnp.asarray(lam), jnp.asarray(beta), jnp.asarray(U))
        return np.asarray(out, dtype=float)
    return np.array([matrix_restricted_signal(shape, g, dt, diffusivity, size, gyromagnetic_ratio, n_modes)
                     for g in g_axes])


def matrix_restricted_signal(shape, g_axis, dt, diffusivity, size, gyromagnetic_ratio, n_modes=16):
    """Exact restricted signal for a projected 1-D gradient magnitude schedule.

    Parameters
    ----------
    shape : {'plane', 'cylinder', 'sphere'}
    g_axis : (n_t,) array
        Gradient magnitude along the restricted axis at each time step [T/m] (already projected).
    dt : float
        Time step [s] (uniform).
    diffusivity : float
        Free diffusivity [m^2/s].
    size : float
        Radius (cylinder/sphere) or half-thickness->thickness (plane) that defines the restriction [m].
        For plane, ``size`` is the slab thickness L; for cylinder/sphere it is the radius R.
    gyromagnetic_ratio : float
        [rad/s/T].
    n_modes : int
        Angular/axial mode count (accuracy knob; the default is converged well below 1e-5 for usual b).

    Returns
    -------
    float : signal attenuation E in [0, 1].
    """
    lam_unit, beta, U = _unit_modes(shape, int(n_modes))
    lam = lam_unit / size ** 2
    g = np.ascontiguousarray(g_axis, dtype=np.float64)
    half = np.exp(-0.5 * dt * diffusivity * lam)          # half diffusion step (diagonal)
    rot = gyromagnetic_ratio * dt * size * beta           # gradient rotation phase per unit g
    c = np.zeros(len(lam), dtype=np.complex128); c[0] = 1.0
    Ut = U.T
    for gk in g:
        c *= half
        if gk != 0.0:
            c = U @ (np.exp(-1j * gk * rot) * (Ut @ c))
        c *= half
    return float(abs(c[0]))
