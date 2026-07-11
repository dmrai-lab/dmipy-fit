"""White-matter analytics (public).

- ``composition`` — the **decoupled, diffusion-only** canonical white-matter model
  (:func:`~dmipy_fit.white_matter.composition.build_white_matter_model`): a standard
  ``MultiCompartmentModel`` of stick / Zeppelin / dot (+ optional ball) compartments, each
  wrapped in an ``OccupancyGatedModel`` carrying T2 and **surface-relaxivity** factors.  The
  intra-pore and exterior surface factors differ per compartment, so surface relaxivity
  introduces a b-independent **signal weighting between intra and extra** (true spin fractions
  ≠ apparent fractions) — a physical effect that ordinary stick+Zeppelin models omit.
- ``surface`` — Brownstein--Tarr / Novikov--Burcaw analytical surface-relaxivity rate models.
- ``mwf`` — standard regularised NNLS T2-spectrum myelin-water fraction
  (:func:`~dmipy_fit.white_matter.mwf.t2_spectrum_mwf`).
"""
from . import surface
from . import mwf
from . import composition
from .mwf import t2_spectrum_mwf
from .composition import (
    build_white_matter_model, white_matter_compartments, canonical_parameters)

__all__ = [
    "surface", "mwf", "composition", "t2_spectrum_mwf",
    "build_white_matter_model", "white_matter_compartments", "canonical_parameters",
]
