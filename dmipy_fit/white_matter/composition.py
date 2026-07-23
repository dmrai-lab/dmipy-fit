"""Diffusion-only analytical white-matter model as a composition of standard pieces.

The canonical white-matter substrate is not a special model class — it is an ordinary
:class:`~dmipy_fit.core.modeling_framework.MultiCompartmentModel` built from standard
diffusion compartments (``C1Stick`` intra-axonal, ``G2Zeppelin`` extra-axonal, ``S1Dot``
stuck-myelin, optional ``G1Ball`` CSF), each wrapped in an
:class:`~dmipy_fit.signal_models.attenuation.OccupancyGatedModel` carrying the *general*,
opt-in occupancy-gated factors: transverse relaxation (``T2``), longitudinal relaxation
(``T1``, active only during a stimulated-echo mixing time ``TM``) and surface relaxivity
(intra-pore + exterior).  Being a real ``MultiCompartmentModel`` it forward-simulates and
fits through the standard machinery, exactly like NODDI.

This is the **decoupled, diffusion-only** analytical model: it takes plain physical
parameters (no ``dmipy_sim.Substrate`` inheritance) and carries no susceptibility or
cross-term physics.  Per-compartment ``T2`` are the *bulk* (intrinsic) values; the surface
factors supply the apparent-T2 shortening explicitly (no double count).  The per-compartment
``T1`` gate transverse-free longitudinal storage: for a plain spin echo (no ``TM``) they are
inert, and on a PGSTE acquisition they apply $\\exp(-\\mathrm{TM}/T_1)$ while ``T2`` and
surface relaxivity see only the encoding lobes.

Defaults are canonical healthy white matter at 3 T; override any of them via keyword.
"""
from __future__ import annotations

from ..core.modeling_framework import MultiCompartmentModel
from ..signal_models.cylinder_models import C1Stick
from ..signal_models.gaussian_models import G2Zeppelin, G1Ball
from ..signal_models.sphere_models import S1Dot
from ..signal_models.attenuation import (
    OccupancyGatedModel, TransverseRelaxation, LongitudinalRelaxation,
    IntraPoreSurfaceRelaxivity, ExteriorSurfaceRelaxivity)
from .surface import exterior_surface_to_volume

# fibre axis +z in dmipy (theta, phi) spherical convention
_MU_Z = [0.0, 0.0]

# --- canonical healthy-WM-at-3T defaults ------------------------------------------------
# Read DIRECTLY from the dmipy_sim biophysical-constants catalogue (the single source of
# truth shared with the Monte-Carlo substrate), so the analytical fit CANNOT drift from the
# forward model.  Every physical value here is catalogue- or Substrate-sourced; only M0
# (a signal scale) and f_axon (the operating-point fibre volume fraction, taken from the
# Substrate default) are modelling choices.  Override any of them via build_white_matter_model.
def _catalogue_defaults():
    import dataclasses
    from dmipy_sim.substrate.biophysical_constants import canonical_white_matter
    from dmipy_sim.substrate.substrate import Substrate
    c = canonical_white_matter(3.0)                     # field-matched, cited values
    sub = {f.name: f.default for f in dataclasses.fields(Substrate)}
    g = c['g_ratio']
    f_axon = sub['f_axon']                              # fibre (OUTER) volume fraction
    w_myelin = sub['myelin_water_proton_density']              # myelin water CONTENT per volume (~0.40)
    # spin (proton) fractions from geometry -- identical construction to Substrate.spin_fractions
    s_i = g ** 2 * f_axon                               # intra-axonal lumen
    s_m = (1.0 - g ** 2) * f_axon * w_myelin            # myelin sheath water (reduced)
    s_e = 1.0 - f_axon                                  # extra-axonal
    tot = s_i + s_m + s_e
    return dict(
        # diffusivities (m^2/s)
        D_intra=c['D_intra'], D_extra=c['D_extra'],
        # Extra-axonal perpendicular (tortuosity) diffusivity. The obstacle to
        # perpendicular diffusion is the whole FIBRE (axon lumen + myelin sheath),
        # so the hindrance scales with the fibre volume fraction f_axon -- NOT the
        # lumen fraction v_ic (the NODDI lambda_par*(1-v_ic) undercounts the myelin
        # as an obstacle). D*(1-f_axon) matches the step-resolved MC extra-axonal
        # D_perp plateau to ~2%. Time-dependence (Burcaw A*ln) is negligible in this
        # regime: the packing correlation length ~1um gives t_c~0.5ms << acquisition
        # Delta, so D_perp is on its tortuosity plateau (measured).
        lambda_perp_extra=c['D_extra'] * (1.0 - f_axon),
        # geometry: ONE Gamma OUTER/fibre-diameter distribution (histology convention)
        g_ratio=g,
        gamma_shape=c['gamma_shape_diameter'],
        gamma_scale_outer_diameter=c['gamma_scale_diameter'],  # catalogue Gamma IS the outer diameter
        f_axon=f_axon, S_ext_over_V=None,               # None -> derive S_ext/V from gamma+f_axon
        # surface relaxivity (m/s), shared by intra + extra walls
        rho2=c['rho2'],
        # bulk transverse relaxation (s)
        T2_intra=c['T2_intra'], T2_extra=c['T2_extra'], T2_myelin=c['T2_myelin'],
        # longitudinal relaxation (s). The public biophysical-constants catalogue does
        # not (yet) resolve compartment-specific T1, so these are literature values for
        # healthy WM at 3T -- a modelling default like M0/f_axon, override via keyword.
        # They are inert unless the acquisition carries a mixing time TM (PGSTE).
        T1_intra=1.2, T1_extra=1.0, T1_myelin=0.44,
        # CSF (only used when include_csf=True)
        D_csf=c['D_csf'], T2_csf=c['T2_csf'], T1_csf=4.0,
        # spin fractions -> partial volumes
        f_intra=s_i / tot, f_extra=s_e / tot, f_myelin=s_m / tot, f_csf=0.0,
        # global signal scale (not a physical constant)
        M0=1.0,
    )


DEFAULTS = _catalogue_defaults()


def white_matter_compartments(include_csf: bool = False, *,
                              gamma_shape=DEFAULTS['gamma_shape'],
                              gamma_scale_outer_diameter=DEFAULTS['gamma_scale_outer_diameter'],
                              f_axon=DEFAULTS['f_axon'],
                              S_ext_over_V=DEFAULTS['S_ext_over_V']):
    """Build the occupancy-gated compartments for the canonical WM substrate.

    Returns a list ``[intra, extra, myelin(, csf)]`` of ``OccupancyGatedModel``
    compartments (the order fixes the ``OccupancyGatedModel_<n>`` parameter names and the
    ``partial_volume_<i>`` ordering).  The surface-factor geometry is fixed at build time;
    the fittable parameters (D, T2, ``surface_relaxivity``, ``g_ratio``, orientation) are
    set/fit afterwards.

    A single Gamma outer (fibre) diameter distribution (``gamma_shape``,
    ``gamma_scale_outer_diameter``) drives BOTH surface factors: the intra-pore ⟨4/d⟩
    average (the intra factor relaxes against the inner wall, ``d_inner = g·d_outer``,
    internally) and — when ``S_ext_over_V`` is None — the exterior surface/volume ratio
    (:func:`dmipy_fit.white_matter.surface.exterior_surface_to_volume`, the same closed form
    the Monte-Carlo substrate uses). Pass ``S_ext_over_V`` explicitly to override.
    """
    if S_ext_over_V is None:
        S_ext_over_V = exterior_surface_to_volume(
            f_axon, gamma_shape, gamma_scale_outer_diameter,
            geometry='cylinder')
    intra = OccupancyGatedModel(C1Stick(), [
        IntraPoreSurfaceRelaxivity(
            gamma_shape=gamma_shape,
            gamma_scale_outer_diameter=gamma_scale_outer_diameter,
            volume_weighted=True),
        TransverseRelaxation(),
        LongitudinalRelaxation(),
    ])
    extra = OccupancyGatedModel(G2Zeppelin(), [
        ExteriorSurfaceRelaxivity(S_ext_over_V=S_ext_over_V),
        TransverseRelaxation(),
        LongitudinalRelaxation(),
    ])
    # myelin water is ~stuck (radial D ~ 0): a stationary Dot, short-T2 only
    myelin = OccupancyGatedModel(
        S1Dot(), [TransverseRelaxation(), LongitudinalRelaxation()])
    compartments = [intra, extra, myelin]
    if include_csf:
        compartments.append(OccupancyGatedModel(
            G1Ball(), [TransverseRelaxation(), LongitudinalRelaxation()]))
    return compartments


def canonical_parameters(include_csf: bool = False, **overrides) -> dict:
    """Canonical forward-parameter dict for :func:`build_white_matter_model`.

    Keyed by the ``MultiCompartmentModel`` parameter names.  Any physical value in
    :data:`DEFAULTS` can be overridden by keyword (e.g. ``rho2=20e-6``, ``g_ratio=0.65``).
    """
    p = dict(DEFAULTS)
    p.update(overrides)
    d = {
        # intra-axonal stick
        'OccupancyGatedModel_1_mu': _MU_Z,
        'OccupancyGatedModel_1_lambda_par': p['D_intra'],
        'OccupancyGatedModel_1_surface_relaxivity': p['rho2'],
        'OccupancyGatedModel_1_g_ratio': p['g_ratio'],
        'OccupancyGatedModel_1_T2': p['T2_intra'],
        'OccupancyGatedModel_1_T1': p['T1_intra'],
        # extra-axonal zeppelin
        'OccupancyGatedModel_2_mu': _MU_Z,
        'OccupancyGatedModel_2_lambda_par': p['D_extra'],
        'OccupancyGatedModel_2_lambda_perp': p['lambda_perp_extra'],
        'OccupancyGatedModel_2_surface_relaxivity': p['rho2'],
        'OccupancyGatedModel_2_T2': p['T2_extra'],
        'OccupancyGatedModel_2_T1': p['T1_extra'],
        # stuck myelin
        'OccupancyGatedModel_3_T2': p['T2_myelin'],
        'OccupancyGatedModel_3_T1': p['T1_myelin'],
        # global scale + partial (spin) volumes
        'S0_global': p['M0'],
        'partial_volume_0': p['f_intra'],
        'partial_volume_1': p['f_extra'],
        'partial_volume_2': p['f_myelin'],
    }
    if include_csf:
        d['OccupancyGatedModel_4_lambda_iso'] = p['D_csf']
        d['OccupancyGatedModel_4_T2'] = p['T2_csf']
        d['OccupancyGatedModel_4_T1'] = p['T1_csf']
        d['partial_volume_3'] = p['f_csf']
    return d


def build_white_matter_model(include_csf: bool = False,
                             tortuosity_constraint: bool = True, **overrides):
    """Build the canonical WM substrate as a fittable ``MultiCompartmentModel``.

    Returns ``(model, parameters)``: ``model`` is a standard ``MultiCompartmentModel``
    (``S0_global`` on, one fibre orientation linked across the intra/extra compartments)
    and ``parameters`` is the canonical forward-parameter dict.  Forward-simulate with
    ``model(scheme, **parameters)``; fit with ``model.fit(scheme, data)`` after fixing the
    parameters you do not want free.  Geometry (``gamma_*``, ``S_ext_over_V``) and every
    physical value in :data:`DEFAULTS` are overridable by keyword.

    ``tortuosity_constraint`` (default True) links the extra-axonal perpendicular
    diffusivity to the parallel diffusivity and the *fibre* volume fraction
    (``lambda_perp = lambda_par * (1 - f_fibre)``), so it is a dependent parameter, not
    a free one.  Two things distinguish this from the stock NODDI tortuosity: (1) the
    obstacle is the whole FIBRE (axon lumen + myelin sheath), not the lumen alone -- the
    myelin hinders extra-axonal diffusion too; (2) the fibre fraction is computed in
    VOLUME terms, converting each compartment's spin (proton) fraction back to volume via
    the myelin water proton density (myelin's spin fraction under-represents its volume).
    The DC tortuosity is valid because the packing correlation time t_c ~ 1um^2/D ~ 0.5ms
    is far below acquisition Delta (see examples/validation/extra_axonal_tortuosity_scale).
    """
    geom = {k: overrides[k] for k in
            ('gamma_shape', 'gamma_scale_outer_diameter', 'f_axon', 'S_ext_over_V')
            if k in overrides}
    compartments = white_matter_compartments(include_csf=include_csf, **geom)
    model = MultiCompartmentModel(models=compartments, S0_global=True)
    # one coherent fibre: the extra-axonal orientation follows the intra-axonal
    model.set_equal_parameter('OccupancyGatedModel_1_mu',
                              'OccupancyGatedModel_2_mu')
    parameters = canonical_parameters(include_csf=include_csf, **overrides)
    parameters.pop('OccupancyGatedModel_2_mu', None)   # linked to _1_mu

    if tortuosity_constraint:
        # lambda_perp = lambda_par * (1 - f_fibre); f_fibre in VOLUME fractions, from the
        # (water-weighted) spin fractions via w_m = myelin_water_proton_density.
        from dmipy_sim.substrate.biophysical_constants import get_default_value
        w_m = float(get_default_value('myelin_water_proton_density'))

        def _fibre_tortuosity(lambda_par, s_intra, s_myelin, s_extra, _w=w_m):
            v_i, v_m, v_e = s_intra, s_myelin / _w, s_extra   # spin -> volume
            return lambda_par * v_e / (v_i + v_m + v_e)

        mdl, nm = model._parameter_map['OccupancyGatedModel_2_lambda_perp']
        model.parameter_links.append([mdl, nm, _fibre_tortuosity, [
            model._parameter_map['OccupancyGatedModel_2_lambda_par'],
            model._parameter_map['partial_volume_0'],    # intra
            model._parameter_map['partial_volume_2'],    # myelin
            model._parameter_map['partial_volume_1']]])  # extra
        for _d in (model.parameter_ranges, model.parameter_cardinality,
                   model.parameter_scales, model.parameter_types,
                   model.parameter_optimization_flags):
            _d.pop('OccupancyGatedModel_2_lambda_perp', None)
        parameters.pop('OccupancyGatedModel_2_lambda_perp', None)   # now linked
    return model, parameters
