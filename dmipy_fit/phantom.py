"""A dmipy-fit compartment model as an **analytic substrate** of a replay phantom (dmipy-sim RPH.md 3.1).

A voxel of a phantom may hold a closed form next to a walked pack: ``ModelSubstrate(C1Stick(), lambda_par=1.7e-9,
m0=0.7)`` is a stick evaluated at the phantom's pose on the phantom's ``ScannerSequence``, composed through the
same SO(3) expansion a pack gets. Orientation belongs to the phantom, never to the model: a model that carries its
own dispersion (``SD1WatsonDistributed``, ``SD2BinghamDistributed``, any ``DistributedModel``) is refused, since a
phantom field on it would disperse twice. The file names the model in this package's namespace
(``"dmipy_fit:C1Stick"``); a reader without dmipy-fit refuses it by name.
"""
import numpy as np

__all__ = ["ModelSubstrate", "analytic_substrate"]


class ModelSubstrate:
    """One compartment model, with its parameters fixed, as a substrate a phantom composes.

    Parameters
    ----------
    model : a dmipy-fit compartment model instance (``C1Stick()``, ``C4CylinderGaussianPhaseApproximation()``,
        ``G1Ball()``, ...). A ``DistributedModel`` is refused.
    m0 : float               proton density of the substrate, required as on a pack.
    T2_s, T1_s : float       optional scalar relaxation weights ``exp(-TE / T2)`` and ``exp(-TM / T1)`` of a
                             sequence's echo and mixing times: what fit's ``eta`` models carry.
    name : str               the substrate id in the file (default ``"model/<Name>"``).
    **params                 every parameter of the model except its orientation ``mu``, by the model's own
                             names and units (``lambda_par=1.7e-9``), validated against ``parameter_ranges``.
    """
    kind = "analytic"

    def __init__(self, model, *, m0, T2_s=None, T1_s=None, name=None, **params):
        from .distributions.distribute_models import DistributedModel
        if isinstance(model, DistributedModel):
            raise ValueError(f"{type(model).__name__} carries its own orientation dispersion and is not citable as a "
                             f"substrate: orientation belongs to the phantom, and a phantom field on a dispersed "
                             f"model would disperse it twice. Cite the undispersed model and give the phantom the "
                             f"Watson / Bingham field.")
        if getattr(model, "_model_type", None) != "CompartmentModel":
            raise TypeError(f"{type(model).__name__} is not a compartment model")
        self.model = model
        self.m0 = float(m0)
        self.T2_s = None if T2_s is None else float(T2_s)
        self.T1_s = None if T1_s is None else float(T1_s)
        self.name = name or f"model/{type(model).__name__}"
        names = list(model.parameter_names)
        self.oriented = "mu" in names
        wanted = [n for n in names if n != "mu"]
        missing = [n for n in wanted if n not in params]
        extra = [n for n in params if n not in wanted]
        if missing or extra:
            raise ValueError(f"{type(model).__name__} takes {wanted}; " + (f"missing {missing}" if missing else "") +
                             (f"; unknown {extra}" if extra else ""))
        for n in wanted:
            lo, hi = model.parameter_ranges[n]
            scale = model.parameter_scales[n]
            v = np.asarray(params[n], float)
            if np.any(v / scale < np.asarray(lo)) or np.any(v / scale > np.asarray(hi)):
                raise ValueError(f"{n} = {params[n]} is outside {type(model).__name__}'s range "
                                 f"[{np.asarray(lo) * scale}, {np.asarray(hi) * scale}] (its own units)")
        self.params = {n: (float(params[n]) if np.ndim(params[n]) == 0 else np.asarray(params[n], float).tolist())
                       for n in wanted}
        self._schemes = {}

    @property
    def model_name(self):
        return f"dmipy_fit:{type(self.model).__name__}"

    def _scheme(self, seq):
        from .core.acquisition_scheme import AcquisitionScheme
        key = id(seq)
        if key not in self._schemes:
            self._schemes = {key: (seq, AcquisitionScheme(seq, require_b0=False))}   # one cached scheme: the current sequence
        return self._schemes[key][1]

    def response(self, seq, pose=None):
        """The model's signal at ``pose`` on ``seq``: complex ``(n_meas,)``. The model's axis is the third column
        of the pose (identity: along z); an isotropic model ignores it. Relaxation weights apply when declared."""
        from .utils.utils import cart2sphere
        scheme = self._scheme(seq)
        kw = dict(self.params)
        if self.oriented:
            axis = (np.eye(3) if pose is None else np.asarray(pose, float).reshape(3, 3))[:, 2]
            kw["mu"] = cart2sphere(axis)[1:]
        E = np.asarray(self.model(scheme, **kw), np.float64).reshape(-1).astype(np.complex128)
        if self.T2_s is not None:
            TE = _echo_time(seq)
            E = E * np.exp(-TE / self.T2_s)
        if self.T1_s is not None:
            TM = float(getattr(seq, "TM", 0.0) or 0.0)
            E = E * np.exp(-TM / self.T1_s)
        return E

    def to_meta(self):
        m = {"id": self.name, "kind": "analytic", "m0": self.m0, "model": self.model_name, "params": dict(self.params)}
        if self.oriented:
            m["oriented"] = True
        if self.T2_s is not None:
            m["T2_s"] = self.T2_s
        if self.T1_s is not None:
            m["T1_s"] = self.T1_s
        return m

    @classmethod
    def from_meta(cls, meta):
        name = str(meta["model"]).split(":", 1)[1]
        model = _model_class(name)()
        return cls(model, m0=meta["m0"], T2_s=meta.get("T2_s"), T1_s=meta.get("T1_s"), name=meta.get("id"),
                   **dict(meta.get("params") or {}))

    def __repr__(self):
        p = ", ".join(f"{k}={v:g}" if np.ndim(v) == 0 else f"{k}={v}" for k, v in self.params.items())
        return f"ModelSubstrate({type(self.model).__name__}(), m0={self.m0:g}{', ' + p if p else ''})"


def analytic_substrate(meta):
    """The reader dmipy-sim calls for a ``"dmipy_fit:<Name>"`` model entry (RPH.md 3.1)."""
    return ModelSubstrate.from_meta(meta)


def _model_class(name):
    import importlib
    from . import signal_models
    for mod_name in signal_models.__all__:
        mod = importlib.import_module(f"dmipy_fit.signal_models.{mod_name}")
        cls = getattr(mod, name, None)
        if cls is not None and getattr(cls, "_model_type", None) == "CompartmentModel":
            return cls
    raise ValueError(f"dmipy_fit defines no compartment model named {name!r}")


def _echo_time(seq):
    enc = getattr(seq, "encoding", None)
    TE = getattr(enc, "TE", None) if enc is not None else None
    if TE is None:
        TE = getattr(seq, "T", None)
    return float(np.max(np.atleast_1d(TE)))
