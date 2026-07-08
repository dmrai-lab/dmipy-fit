"""JAX implementations of Watson- and Bingham-distributed forward models.

Two strategies are available and selected automatically at factory time:

SH convolution (fast, default when acquisition_scheme is supplied)
------------------------------------------------------------------
Mirrors the original dmipy CPU path:

    1. Evaluate Watson/Bingham PDF on the 362-vertex hemisphere to get a
       spherical function (SF).
    2. Project SF → SH via the precomputed pseudo-inverse (static matmul).
    3. Evaluate each inner model at 10 z-axis-aligned angles (the RH
       sampling scheme) and project those to rotational harmonics (RH).
    4. SH convolution: element-wise multiply dist_sh by rh_inner broadcast
       to each coefficient, scaled by sqrt(4π/(2l+1)).
    5. Reconstruct per-shell: shell_sh_mat @ convolved_sh.

All matrices in steps 2–5 are precomputed at factory time (static).
Runtime cost: O(362) exp + a few small matmuls — no vmap over 362
full forward evals.

Numerical integration (fallback for unsupported inner models)
-------------------------------------------------------------
Original implementation: jax.vmap evaluates the inner model at 362
hemisphere directions, weights by the ODF, and sums.  Used when the inner
model has no JAX RH function (e.g. C2/C3/C4 cylinders).

Normalization (SH path):
    unnorm_sf = exp(kappa * (n·mu)^2) on 362 vertices
    dist_sh   = inv_sh_mat @ unnorm_sf                  # (N_sh,)
    Normalise: dist_sh /= dist_sh[0] * sqrt(4π)
    (l=0 coeff of a unit-integral ODF equals 1/sqrt(4π))
"""

import numpy as np
import jax
import jax.numpy as jnp

from .jax_compat import unitsphere2cart_1d_jax

# ---------------------------------------------------------------------------
# Hemisphere quadrature (shared by both strategies)
# ---------------------------------------------------------------------------

def _hemisphere_vertices_jax():
    """Return dipy symmetric724 hemisphere vertices as a jnp array (362, 3)."""
    from dipy.data import get_sphere, HemiSphere
    sphere = get_sphere(name='symmetric724')
    hemi = HemiSphere(phi=sphere.phi, theta=sphere.theta)
    return jnp.array(hemi.vertices)


_HEMI_VERTS_JAX = None


def _get_hemi_verts():
    global _HEMI_VERTS_JAX
    if _HEMI_VERTS_JAX is None:
        _HEMI_VERTS_JAX = _hemisphere_vertices_jax()
    return _HEMI_VERTS_JAX


# ---------------------------------------------------------------------------
# Rotation matrix helpers (Bingham second axis) — unchanged
# ---------------------------------------------------------------------------

def _rotation_matrix_100_to_xyz_jax(x, y, z):
    y2 = y ** 2
    z2 = z ** 2
    yz = y * z
    denom = y2 + z2
    safe_denom = jnp.where(denom > 1e-30, denom, jnp.ones_like(denom))
    return jnp.array([
        [x,  -y,  -z],
        [y,   (x * y2 + z2) / safe_denom,  ((x - 1.) * yz) / safe_denom],
        [z,   ((x - 1.) * yz) / safe_denom,  (y2 + x * z2) / safe_denom],
    ])


def _rotation_matrix_100_to_theta_phi_jax(theta, phi):
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    sin_p, cos_p = jnp.sin(phi),   jnp.cos(phi)
    return _rotation_matrix_100_to_xyz_jax(
        sin_t * cos_p, sin_t * sin_p, cos_t)


def _rotation_matrix_around_100_jax(psi):
    cos_p, sin_p = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([
        [1.,  0.,    0.   ],
        [0.,  cos_p, -sin_p],
        [0.,  sin_p,  cos_p],
    ])


def _rotation_matrix_100_to_theta_phi_psi_jax(theta, phi, psi):
    R1 = _rotation_matrix_100_to_theta_phi_jax(theta, phi)
    R2 = _rotation_matrix_around_100_jax(psi)
    return jnp.dot(R1, R2)


# ---------------------------------------------------------------------------
# SH convolution helpers (factory-time, numpy)
# ---------------------------------------------------------------------------

def _build_sh_conv_arrays(max_sh_order):
    """Build per-SH-coefficient order index and convolution scale vectors.

    For SH order n, all (2n+1) coefficients share the same RH index (n//2)
    and the same convolution scale sqrt(4π/(2n+1)).

    Returns
    -------
    order_rh_idx : np.ndarray, shape (N_sh,), int
        For each SH coefficient, the index into the RH array (0 = l=0, 1 = l=2, …)
    conv_scales : np.ndarray, shape (N_sh,), float
        sqrt(4π/(2n+1)) broadcast to every coefficient in order n.
    """
    order_rh_idx = []
    conv_scales  = []
    rh_idx = 0
    for n in range(0, max_sh_order + 1, 2):
        n_coef = 2 * n + 1
        order_rh_idx.extend([rh_idx] * n_coef)
        conv_scales.extend([np.sqrt(4.0 * np.pi / (2 * n + 1))] * n_coef)
        rh_idx += 1
    return np.array(order_rh_idx, dtype=np.int32), np.array(conv_scales)


# ---------------------------------------------------------------------------
# JAX RH functions for supported inner models
# (evaluate model at z-axis mu=[0,0] on the 10-point RH scheme)
# ---------------------------------------------------------------------------

def _c1stick_rh_fn(b_shell, cos_sq_rh, _sin_sq_rh, params):
    """C1Stick at z-axis: exp(-b * lambda_par * cos²θ)."""
    return jnp.exp(-b_shell * params['lambda_par'] * cos_sq_rh)


def _g2zeppelin_rh_fn(b_shell, cos_sq_rh, sin_sq_rh, params):
    """G2Zeppelin at z-axis: exp(-b*(lambda_par*cos²θ + lambda_perp*sin²θ))."""
    return jnp.exp(-b_shell * (params['lambda_par'] * cos_sq_rh +
                                params['lambda_perp'] * sin_sq_rh))


def _g1ball_rh_fn(b_shell, cos_sq_rh, _sin_sq_rh, params):
    """G1Ball: isotropic → constant across angles."""
    return jnp.exp(-b_shell * params['lambda_iso']) * jnp.ones_like(cos_sq_rh)


# Map from model class to RH function
from ..signal_models.gaussian_models import G1Ball, G2Zeppelin
from ..signal_models.cylinder_models import C1Stick

_JAX_RH_FNS = {
    G1Ball:      _g1ball_rh_fn,
    C1Stick:     _c1stick_rh_fn,
    G2Zeppelin:  _g2zeppelin_rh_fn,
}


def _all_inner_models_have_rh(model_obj):
    """Return True iff every inner model has a JAX RH function."""
    return all(type(m) in _JAX_RH_FNS for m in model_obj.models)


# ---------------------------------------------------------------------------
# SH-based Watson distributed factory
# ---------------------------------------------------------------------------

def build_watson_sh_jax_fn(model_obj, acquisition_scheme):
    """Build an SH-convolution JAX forward function for SD1WatsonDistributed.

    Uses the precomputed shell_sh_matrices and inverse_rh_matrix from
    acquisition_scheme to evaluate the convolution via static matmuls,
    avoiding 362 full inner-model evaluations.

    Parameters
    ----------
    model_obj : SD1WatsonDistributed
    acquisition_scheme : DmipyAcquisitionScheme

    Returns
    -------
    fn : callable (scheme_jax, params_dist) -> jnp.array (N_meas,)
    """
    from ..distributions.distributions import inverse_sh_matrix_kernel

    hemi_verts = _get_hemi_verts()                       # (362, 3), static

    # Maximum SH order across all DWI shells
    max_sh_order = max(
        int(acquisition_scheme.shell_sh_orders[idx])
        for idx in acquisition_scheme.unique_dwi_indices
    )

    # ODF → SH projection matrix  (N_sh_max × 362)
    inv_sh_mat_jax = jnp.array(inverse_sh_matrix_kernel[max_sh_order])

    # Per-coefficient RH index and convolution scale
    order_rh_idx_np, conv_scales_np = _build_sh_conv_arrays(max_sh_order)
    order_rh_idx_jax = jnp.array(order_rh_idx_np)   # (N_sh_max,)
    conv_scales_jax  = jnp.array(conv_scales_np)     # (N_sh_max,)

    # RH sampling: cos²θ and sin²θ at 10 z-axis angles
    rh_scheme  = acquisition_scheme.rotational_harmonics_scheme
    rh_thetas  = np.linspace(0.0, np.pi / 2.0, rh_scheme.Nsamples)
    cos_sq_rh_jax = jnp.array(np.cos(rh_thetas) ** 2)   # (10,)
    sin_sq_rh_jax = 1.0 - cos_sq_rh_jax                  # (10,)

    # Per-DWI-shell static data
    shell_entries = []
    for shell_idx in acquisition_scheme.unique_dwi_indices:
        sh_order = int(acquisition_scheme.shell_sh_orders[shell_idx])
        N_sh     = int((sh_order + 2) * (sh_order + 1) // 2)
        N_rh     = sh_order // 2 + 1
        b_shell  = float(acquisition_scheme.shell_bvalues[shell_idx])
        indices  = np.where(acquisition_scheme.shell_indices == shell_idx)[0]
        shell_entries.append({
            'b':           b_shell,
            'N_sh':        N_sh,
            'N_rh':        N_rh,
            'indices':     jnp.array(indices, dtype=jnp.int32),
            'shell_sh_mat':jnp.array(
                acquisition_scheme.shell_sh_matrices[shell_idx]),   # (N_meas_shell, N_sh)
            'inv_rh_mat':  jnp.array(
                rh_scheme.inverse_rh_matrix[sh_order]),              # (N_rh, 10)
        })

    # Watson distribution parameter keys
    watson_mu_key  = model_obj._inverted_parameter_map[(model_obj.distribution, 'mu')]
    watson_odi_key = model_obj._inverted_parameter_map[(model_obj.distribution, 'odi')]

    # Inner model data
    inner_models     = model_obj.models
    inner_rh_fns     = [_JAX_RH_FNS[type(m)] for m in inner_models]
    inner_param_maps = []
    for m in inner_models:
        pm = [(lp, model_obj._inverted_parameter_map[(m, lp)])
              for lp in m.parameter_ranges
              if m.parameter_types[lp] != 'orientation']
        inner_param_maps.append(pm)

    has_pv   = len(inner_models) > 1
    pv_names = list(model_obj.partial_volume_names) if has_pv else []

    N_meas = acquisition_scheme.number_of_measurements

    def watson_sh_fn(scheme_jax, params_dist):
        params_dist = model_obj.add_linked_parameters_to_parameters(params_dist)

        # Step 1: Watson ODF → SH coefficients
        mu      = params_dist[watson_mu_key]
        odi     = params_dist[watson_odi_key]
        mu_cart = unitsphere2cart_1d_jax(mu)
        kappa   = 1.0 / jnp.tan(odi * jnp.pi / 2.0)
        cos_sq  = jnp.dot(hemi_verts, mu_cart) ** 2     # (362,)
        sf_unnorm = jnp.exp(kappa * cos_sq)              # (362,)
        dist_sh   = jnp.dot(inv_sh_mat_jax, sf_unnorm)  # (N_sh_max,)
        # Normalise: the l=0,m=0 coeff of a unit-sphere-integral ODF is 1/√(4π)
        dist_sh   = dist_sh / (dist_sh[0] * jnp.sqrt(4.0 * jnp.pi))

        # Steps 2–5: per-shell, per-inner-model convolution with T2
        E = jnp.ones(N_meas)
        TE_arr = scheme_jax.get('TE')

        for entry in shell_entries:
            b      = entry['b']
            N_sh   = entry['N_sh']
            N_rh   = entry['N_rh']

            remaining_vf = jnp.array(1.0)
            E_shell = jnp.zeros(len(entry['indices']))
            for i, (inner_model, rh_fn, pm) in enumerate(
                    zip(inner_models, inner_rh_fns, inner_param_maps)):
                inner_p = {lp: params_dist[dp] for lp, dp in pm}
                if has_pv and i < len(inner_models) - 1:
                    pv           = params_dist[pv_names[i]]
                    vf           = remaining_vf * pv
                    remaining_vf = remaining_vf - vf
                else:
                    vf = remaining_vf

                E_at_rh  = rh_fn(b, cos_sq_rh_jax, sin_sq_rh_jax, inner_p)   # (10,)
                rh_inner = jnp.dot(entry['inv_rh_mat'], E_at_rh)[:N_rh]       # (N_rh,)

                # Per-inner-model SH convolution
                rh_per_coef  = rh_inner[order_rh_idx_jax[:N_sh]]
                convolved_sh = dist_sh[:N_sh] * rh_per_coef * conv_scales_jax[:N_sh]
                E_inner_shell = jnp.dot(entry['shell_sh_mat'], convolved_sh)

                # Apply T2 weighting: exp(-TE/T2) factored out of spherical integral
                T2_val = inner_p.get('T2')
                if T2_val is not None and TE_arr is not None:
                    TE_shell = TE_arr[entry['indices']]
                    t2_factor = jnp.where(
                        jnp.isfinite(T2_val), jnp.exp(-TE_shell / T2_val), 1.0)
                    E_inner_shell = E_inner_shell * t2_factor

                E_shell = E_shell + vf * E_inner_shell

            E = E.at[entry['indices']].set(E_shell)

        return E

    return watson_sh_fn


# ---------------------------------------------------------------------------
# SH-based Bingham distributed factory
# ---------------------------------------------------------------------------

def build_bingham_sh_jax_fn(model_obj, acquisition_scheme):
    """Build an SH-convolution JAX forward function for SD2BinghamDistributed.

    Same structure as the Watson SH factory; only the ODF weights differ.
    """
    from ..distributions.distributions import inverse_sh_matrix_kernel

    hemi_verts = _get_hemi_verts()

    max_sh_order = max(
        int(acquisition_scheme.shell_sh_orders[idx])
        for idx in acquisition_scheme.unique_dwi_indices
    )
    inv_sh_mat_jax = jnp.array(inverse_sh_matrix_kernel[max_sh_order])

    order_rh_idx_np, conv_scales_np = _build_sh_conv_arrays(max_sh_order)
    order_rh_idx_jax = jnp.array(order_rh_idx_np)
    conv_scales_jax  = jnp.array(conv_scales_np)

    rh_scheme     = acquisition_scheme.rotational_harmonics_scheme
    rh_thetas     = np.linspace(0.0, np.pi / 2.0, rh_scheme.Nsamples)
    cos_sq_rh_jax = jnp.array(np.cos(rh_thetas) ** 2)
    sin_sq_rh_jax = 1.0 - cos_sq_rh_jax

    shell_entries = []
    for shell_idx in acquisition_scheme.unique_dwi_indices:
        sh_order = int(acquisition_scheme.shell_sh_orders[shell_idx])
        N_sh     = int((sh_order + 2) * (sh_order + 1) // 2)
        N_rh     = sh_order // 2 + 1
        b_shell  = float(acquisition_scheme.shell_bvalues[shell_idx])
        indices  = np.where(acquisition_scheme.shell_indices == shell_idx)[0]
        shell_entries.append({
            'b':           b_shell,
            'N_sh':        N_sh,
            'N_rh':        N_rh,
            'indices':     jnp.array(indices, dtype=jnp.int32),
            'shell_sh_mat':jnp.array(
                acquisition_scheme.shell_sh_matrices[shell_idx]),
            'inv_rh_mat':  jnp.array(
                rh_scheme.inverse_rh_matrix[sh_order]),
        })

    bingham_mu_key           = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'mu')]
    bingham_psi_key          = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'psi')]
    bingham_odi_key          = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'odi')]
    bingham_beta_fraction_key = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'beta_fraction')]

    inner_models     = model_obj.models
    inner_rh_fns     = [_JAX_RH_FNS[type(m)] for m in inner_models]
    inner_param_maps = []
    for m in inner_models:
        pm = [(lp, model_obj._inverted_parameter_map[(m, lp)])
              for lp in m.parameter_ranges
              if m.parameter_types[lp] != 'orientation']
        inner_param_maps.append(pm)

    has_pv   = len(inner_models) > 1
    pv_names = list(model_obj.partial_volume_names) if has_pv else []

    N_meas = acquisition_scheme.number_of_measurements

    def bingham_sh_fn(scheme_jax, params_dist):
        params_dist = model_obj.add_linked_parameters_to_parameters(params_dist)

        # Step 1: Bingham ODF → SH coefficients
        mu            = params_dist[bingham_mu_key]
        psi           = params_dist[bingham_psi_key]
        odi           = params_dist[bingham_odi_key]
        beta_fraction = params_dist[bingham_beta_fraction_key]

        mu_cart = unitsphere2cart_1d_jax(mu)
        kappa   = 1.0 / jnp.tan(odi * jnp.pi / 2.0)
        beta    = beta_fraction * kappa

        R       = _rotation_matrix_100_to_theta_phi_psi_jax(mu[0], mu[1], psi)
        mu_beta = jnp.dot(R, jnp.array([0.0, 1.0, 0.0]))

        cos_sq1   = jnp.dot(hemi_verts, mu_cart)  ** 2   # (362,)
        cos_sq2   = jnp.dot(hemi_verts, mu_beta)  ** 2   # (362,)
        sf_unnorm = jnp.exp(kappa * cos_sq1 + beta * cos_sq2)
        dist_sh   = jnp.dot(inv_sh_mat_jax, sf_unnorm)
        dist_sh   = dist_sh / (dist_sh[0] * jnp.sqrt(4.0 * jnp.pi))

        # Steps 2–5: per-shell, per-inner-model convolution with T2
        E = jnp.ones(N_meas)
        TE_arr = scheme_jax.get('TE')

        for entry in shell_entries:
            b    = entry['b']
            N_sh = entry['N_sh']
            N_rh = entry['N_rh']

            remaining_vf = jnp.array(1.0)
            E_shell = jnp.zeros(len(entry['indices']))
            for i, (inner_model, rh_fn, pm) in enumerate(
                    zip(inner_models, inner_rh_fns, inner_param_maps)):
                inner_p = {lp: params_dist[dp] for lp, dp in pm}
                if has_pv and i < len(inner_models) - 1:
                    pv           = params_dist[pv_names[i]]
                    vf           = remaining_vf * pv
                    remaining_vf = remaining_vf - vf
                else:
                    vf = remaining_vf

                E_at_rh  = rh_fn(b, cos_sq_rh_jax, sin_sq_rh_jax, inner_p)
                rh_inner = jnp.dot(entry['inv_rh_mat'], E_at_rh)[:N_rh]

                # Per-inner-model SH convolution
                rh_per_coef  = rh_inner[order_rh_idx_jax[:N_sh]]
                convolved_sh = dist_sh[:N_sh] * rh_per_coef * conv_scales_jax[:N_sh]
                E_inner_shell = jnp.dot(entry['shell_sh_mat'], convolved_sh)

                # Apply T2 weighting: exp(-TE/T2) factored out of spherical integral
                T2_val = inner_p.get('T2')
                if T2_val is not None and TE_arr is not None:
                    TE_shell = TE_arr[entry['indices']]
                    t2_factor = jnp.where(
                        jnp.isfinite(T2_val), jnp.exp(-TE_shell / T2_val), 1.0)
                    E_inner_shell = E_inner_shell * t2_factor

                E_shell = E_shell + vf * E_inner_shell

            E = E.at[entry['indices']].set(E_shell)

        return E

    return bingham_sh_fn


# ---------------------------------------------------------------------------
# Numerical integration fallback (unchanged from original implementation)
# ---------------------------------------------------------------------------

def build_watson_distributed_jax_fn(model_obj, inner_jax_fns_dict):
    """Numerical-integration Watson forward function (fallback).

    Used when inner models lack JAX RH functions (e.g. C2/C3/C4 cylinders).
    """
    hemi_verts_jax = _get_hemi_verts()

    watson_mu_key  = model_obj._inverted_parameter_map[(model_obj.distribution, 'mu')]
    watson_odi_key = model_obj._inverted_parameter_map[(model_obj.distribution, 'odi')]

    inner_models     = model_obj.models
    inner_jax_fns    = [inner_jax_fns_dict[m] for m in inner_models]
    inner_param_maps = []
    for inner_model in inner_models:
        pm = []
        for lp in inner_model.parameter_ranges:
            if inner_model.parameter_types[lp] != 'orientation':
                full = model_obj._inverted_parameter_map[(inner_model, lp)]
                pm.append((lp, full))
        inner_param_maps.append(pm)

    has_pv    = len(inner_models) > 1
    pv_names  = list(model_obj.partial_volume_names) if has_pv else []

    def watson_distributed_fn(scheme_jax, params_dist):
        params_dist = model_obj.add_linked_parameters_to_parameters(params_dist)

        mu      = params_dist[watson_mu_key]
        odi     = params_dist[watson_odi_key]
        mu_cart = unitsphere2cart_1d_jax(mu)
        kappa   = 1.0 / jnp.tan(odi * jnp.pi / 2.0)
        cos_sq  = jnp.dot(hemi_verts_jax, mu_cart) ** 2
        w_unnorm = jnp.exp(kappa * cos_sq)
        weights  = w_unnorm / jnp.sum(w_unnorm)

        N_meas = scheme_jax['bvalues'].shape[0]

        def eval_direction(n_k):
            theta_k = jnp.arccos(jnp.clip(n_k[2], -1.0, 1.0))
            phi_k   = jnp.arctan2(n_k[1], n_k[0])
            mu_k    = jnp.stack([theta_k, phi_k])

            E = jnp.zeros(N_meas)
            remaining_vf = jnp.array(1.0)
            for i, (inner_model, inner_fn, pm) in enumerate(
                    zip(inner_models, inner_jax_fns, inner_param_maps)):
                inner_p = {'mu': mu_k}
                for (local_name, dist_name) in pm:
                    inner_p[local_name] = params_dist[dist_name]

                if has_pv and i < len(inner_models) - 1:
                    pv      = params_dist[pv_names[i]]
                    vf      = remaining_vf * pv
                    remaining_vf = remaining_vf - vf
                else:
                    vf = remaining_vf

                E = E + vf * inner_fn(scheme_jax, inner_p)
            return E

        all_signals = jax.vmap(eval_direction)(hemi_verts_jax)
        return jnp.dot(weights, all_signals)

    return watson_distributed_fn


def build_bingham_distributed_jax_fn(model_obj, inner_jax_fns_dict):
    """Numerical-integration Bingham forward function (fallback)."""
    hemi_verts_jax = _get_hemi_verts()

    bingham_mu_key           = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'mu')]
    bingham_psi_key          = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'psi')]
    bingham_odi_key          = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'odi')]
    bingham_beta_fraction_key = model_obj._inverted_parameter_map[
        (model_obj.distribution, 'beta_fraction')]

    inner_models     = model_obj.models
    inner_jax_fns    = [inner_jax_fns_dict[m] for m in inner_models]
    inner_param_maps = []
    for inner_model in inner_models:
        pm = []
        for lp in inner_model.parameter_ranges:
            if inner_model.parameter_types[lp] != 'orientation':
                full = model_obj._inverted_parameter_map[(inner_model, lp)]
                pm.append((lp, full))
        inner_param_maps.append(pm)

    has_pv   = len(inner_models) > 1
    pv_names = list(model_obj.partial_volume_names) if has_pv else []

    def bingham_distributed_fn(scheme_jax, params_dist):
        params_dist = model_obj.add_linked_parameters_to_parameters(params_dist)

        mu            = params_dist[bingham_mu_key]
        psi           = params_dist[bingham_psi_key]
        odi           = params_dist[bingham_odi_key]
        beta_fraction = params_dist[bingham_beta_fraction_key]

        mu_cart = unitsphere2cart_1d_jax(mu)
        kappa   = 1.0 / jnp.tan(odi * jnp.pi / 2.0)
        beta    = beta_fraction * kappa

        R        = _rotation_matrix_100_to_theta_phi_psi_jax(mu[0], mu[1], psi)
        mu_beta  = jnp.dot(R, jnp.array([0.0, 1.0, 0.0]))

        cos_sq1  = jnp.dot(hemi_verts_jax, mu_cart)  ** 2
        cos_sq2  = jnp.dot(hemi_verts_jax, mu_beta)  ** 2
        w_unnorm = jnp.exp(kappa * cos_sq1 + beta * cos_sq2)
        weights  = w_unnorm / jnp.sum(w_unnorm)

        N_meas = scheme_jax['bvalues'].shape[0]

        def eval_direction(n_k):
            theta_k = jnp.arccos(jnp.clip(n_k[2], -1.0, 1.0))
            phi_k   = jnp.arctan2(n_k[1], n_k[0])
            mu_k    = jnp.stack([theta_k, phi_k])

            E = jnp.zeros(N_meas)
            remaining_vf = jnp.array(1.0)
            for i, (inner_model, inner_fn, pm) in enumerate(
                    zip(inner_models, inner_jax_fns, inner_param_maps)):
                inner_p = {'mu': mu_k}
                for (local_name, dist_name) in pm:
                    inner_p[local_name] = params_dist[dist_name]

                if has_pv and i < len(inner_models) - 1:
                    pv           = params_dist[pv_names[i]]
                    vf           = remaining_vf * pv
                    remaining_vf = remaining_vf - vf
                else:
                    vf = remaining_vf

                E = E + vf * inner_fn(scheme_jax, inner_p)
            return E

        all_signals = jax.vmap(eval_direction)(hemi_verts_jax)
        return jnp.dot(weights, all_signals)

    return bingham_distributed_fn
