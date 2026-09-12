"""A walked substrate as a compartment: any replay pack (``.rpk``), evaluated at a pose.

The canonical families (``C6`` / ``S6`` / ``P6``) interpolate a *dataset* over diameter; this model cites ONE
pack -- a CACTUS bundle, a mesh, a canonical pore at a fixed diameter -- and gives its replay as a compartment
signal, so "fit a pack" is one line: the pack's own axis (its declared substrate frame, RPK.md 4.2) is the
orientation parameter ``mu``, the acquisition is compiled through dmipy-sim's exact replay kernel, and surface
relaxivity is the pack's boundary-local-time replay knob.
"""
import os

import numpy as np

from ..core.constants import CONSTANTS
from ..core.model_properties import ModelProperties
from ..core.signal_model_properties import AnisotropicSignalModelProperties
from ..utils import utils

_GAMMA = CONSTANTS["water_gyromagnetic_ratio"]

__all__ = ["MonteCarloReplay"]


class MonteCarloReplay(ModelProperties, AnisotropicSignalModelProperties):
    """One replay pack as a compartment model, posed by ``mu``.

    ``mu`` is the lab direction of the pack's own structural axis (column 3 of its ``substrate_frame``: the fibre
    bundle, the cylinder axis); the acquisition is rotated into the pack's stored coordinates by
    ``ReplayPack.pose_rotation``, exact by pose covariance, and compiled through
    :func:`dmipy_sim.replay.compile_scheme`. An isotropic pack (a sphere, a free walk) is the same at every
    ``mu``. ``surface_relaxivity`` (m/s) is the pack's exact C2 replay knob, with the pack's walk diffusivity;
    relaxation is a compartment ``eta`` model's business, as for every other compartment.

    Parameters
    ----------
    pack : ReplayPack or path        the walked substrate (``dmipy_sim.replay.read_rpk``).
    mu : array, shape (2)            angles [theta, phi] of the pack's axis on the sphere.
    """
    _citations = {
        'definition': [
            {'key': 'fick2026replay', 'authors': 'Fick RHJ', 'title': 'One walk, every acquisition: replay packs',
             'journal': 'dmrai-lab', 'year': 2026}
        ],
        'default_parameters': {},
    }
    _validity_constraints = [
        {'id': 'fixed_substrate', 'name': 'Fixed substrate',
         'condition_human': 'The pack is walked once: its geometry and diffusivity are fixed at walk time, not fitted.',
         'severity': 'info'},
        {'id': 'acquisition_inside_walk', 'name': 'Acquisition inside the walk',
         'condition_human': "The acquisition must end within the pack's stored duration and stay within its certified envelope.",
         'severity': 'warning'},
    ]
    _required_acquisition_parameters = ['gradient_directions']
    _supports_waveform_scheme = True
    _parameter_ranges = {'mu': ([0, np.pi], [-np.pi, np.pi])}
    _parameter_scales = {'mu': np.r_[1., 1.]}
    _parameter_types = {'mu': 'orientation'}
    _model_type = 'CompartmentModel'

    def __init__(self, pack, *, mu=None):
        from dmipy_sim.replay import ReplayPack, read_rpk
        if isinstance(pack, (str, os.PathLike)):
            pack = read_rpk(os.fspath(pack))
        if not isinstance(pack, ReplayPack):
            raise TypeError(f"pack must be a dmipy_sim.replay.ReplayPack or a path to one, got {type(pack).__name__}")
        self.pack = pack
        self.mu = mu
        self._coeffs = None

    @property
    def diffusivity(self):
        """The pack's walk diffusivity (m^2/s): fixed at walk time."""
        return self.pack.diffusivity

    def _arrays(self):
        if self._coeffs is None:
            from dmipy_sim.replay.compression import read_position_coeffs
            C = read_position_coeffs(self.pack.arrays, dtype=np.float64)
            w = np.asarray(self.pack.arrays.get("spin_weights", np.ones(C.shape[0])), np.float64)
            self._coeffs = (C, w)
        return self._coeffs

    def __call__(self, acquisition_scheme, **kwargs):
        from dmipy_sim.replay import compile_scheme, replay_coefficients, surface_logweight
        from dmipy_sim.replay._replay_kernel import bin_gate
        mu = kwargs.get('mu', self.mu)
        if mu is None:
            raise ValueError("MonteCarloReplay needs mu, the lab direction of the pack's axis")
        rho = float(kwargs.get('surface_relaxivity', 0.0) or 0.0)
        chi = kwargs.get('coherence_gate', None)
        G = getattr(acquisition_scheme, '_G', None)
        if G is None:
            raise ValueError("MonteCarloReplay needs a waveform-first AcquisitionScheme "
                             "(build with AcquisitionScheme.from_pgse/from_waveform) -- no ._G on this scheme.")
        dt = float(acquisition_scheme._dt)
        pk = self.pack
        R = pk.pose_rotation(utils.unitsphere2cart_1d(np.asarray(mu, float)))     # stored -> lab
        G_stored = np.asarray(G, np.float64) @ R                                    # the lab waveform in stored coordinates
        W = compile_scheme(G_stored, dt, pk.K, _GAMMA, n_t=pk.n_t, dt_pack=pk.dt)
        C, w = self._arrays()
        slw = None
        if rho:
            D = self.diffusivity
            if D is None:
                raise ValueError("surface relaxivity needs the walk's diffusivity, which this pack did not record")
            cm = (pk.meta.get("compression", {}).get("channels", {}) or {}).get("boundary_local_time")
            gate = chi if chi is not None else bin_gate(np.ones(G_stored.shape[1]), dt, pk.n_t, pk.dt)[0]
            slw = surface_logweight(pk.arrays, rho / float(D), cm, gate)
        return replay_coefficients(C, w, W, surface_logw=slw)
