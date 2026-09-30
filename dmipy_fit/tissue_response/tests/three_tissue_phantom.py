"""A synthetic three-tissue volume for the response and multi-tissue CSD tests: a multi-shell scheme (b = 1000 /
2000 / 3000 s/mm^2, 96 directions each, 15 b = 0), and voxels that mix a white-matter zeppelin population (one or two
fibre directions), a grey-matter ball and free water, each tissue with its own S0; Gaussian noise. Deterministic."""
import numpy as np

from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues

S0 = {'wm': 0.7, 'gm': 0.85, 'csf': 1.0}
WM = dict(lambda_par=1.7e-9, lambda_perp=0.4e-9)
D_GM, D_CSF = 0.8e-9, 3.0e-9


def multishell_scheme(shells=(1000, 2000, 3000), n_dir=96, n_b0=15, seed=0):
    rng = np.random.default_rng(seed)
    b, g = [np.zeros(n_b0)], [np.tile([0., 0., 1.], (n_b0, 1))]
    for s in shells:
        d = rng.normal(size=(n_dir, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        b.append(np.full(n_dir, s * 1e6))
        g.append(d)
    return acquisition_scheme_from_bvalues(np.concatenate(b), np.concatenate(g), delta=0.01, Delta=0.03)


def zeppelin(scheme, mu):
    """``(N, N_meas)`` for unit directions ``mu (N, 3)``."""
    c2 = (np.asarray(mu) @ scheme.gradient_directions.T) ** 2
    return np.exp(-scheme.bvalues * (WM['lambda_perp'] + (WM['lambda_par'] - WM['lambda_perp']) * c2))


def ball(scheme, D):
    return np.exp(-scheme.bvalues * D)


def _unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def mixture(scheme, fractions, rng, crossing=0.5):
    """Signals ``(N, N_meas)`` of voxels with tissue ``fractions (N, 3)`` (WM, GM, CSF), the WM of each voxel one
    fibre or (with probability ``crossing``) two with random weights, directions uniform."""
    n = len(fractions)
    w2 = np.where(rng.uniform(size=n) < crossing, rng.uniform(0.2, 0.5, n), 0.0)
    wm = (1 - w2)[:, None] * zeppelin(scheme, _unit(rng, n)) + w2[:, None] * zeppelin(scheme, _unit(rng, n))
    return (fractions[:, :1] * S0['wm'] * wm + fractions[:, 1:2] * S0['gm'] * ball(scheme, D_GM)[None]
            + fractions[:, 2:] * S0['csf'] * ball(scheme, D_CSF)[None])


def voxels(scheme, n=500, seed=1, sigma=0.02):
    """``(data (n, N_meas), fractions (n, 3))``: Dirichlet(2, 1, 1) tissue fractions, noise ``sigma``."""
    rng = np.random.default_rng(seed)
    fr = rng.dirichlet([2., 1., 1.], size=n)
    data = mixture(scheme, fr, rng)
    return data + rng.normal(0, sigma, data.shape), fr


def volume(scheme, shape=(24, 24, 12), seed=2, sigma=0.01):
    """``(data shape + (N_meas,), mask shape, fractions shape + (3,))``: a brain of mostly-pure tissue blocks
    (x < 1/2 white matter with half the voxels crossing, 1/2 <= x < 5/6 grey matter, the rest free water; every
    voxel 85-100 % its tissue, the rest split between the others) inside a zero background ring of 2 voxels."""
    rng = np.random.default_rng(seed)
    nx = shape[0]
    x = np.arange(nx)[:, None, None] * np.ones(shape)
    tissue = np.where(x < nx / 2, 0, np.where(x < nx * 5 / 6, 1, 2)).astype(int).ravel()
    n = tissue.size
    main = rng.uniform(0.85, 1.0, n)
    rest = rng.dirichlet([1., 1.], size=n) * (1 - main)[:, None]
    fr = np.zeros((n, 3))
    fr[np.arange(n), tissue] = main
    others = np.array([[1, 2], [0, 2], [0, 1]])[tissue]
    fr[np.arange(n)[:, None], others] = rest
    data = mixture(scheme, fr, rng) + rng.normal(0, sigma, (n, scheme.number_of_measurements))
    mask = np.zeros(shape, bool)
    mask[2:-2, 2:-2, 2:-2] = True
    data = np.abs(data).reshape(shape + (-1,)) * mask[..., None]
    return data, mask, fr.reshape(shape + (3,))
