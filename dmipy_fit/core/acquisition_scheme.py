import numpy as np
from collections import namedtuple
from .gradient_conversions import (
    g_from_b, q_from_b, b_from_q, g_from_q, b_from_g, q_from_g)
from .constants import CONSTANTS
from ..utils import utils
from ..utils.spherical_convolution import real_sym_rh_basis
from dipy.reconst.shm import real_sh_tournier as real_sym_sh_mrtrix
from scipy.cluster.hierarchy import fcluster, linkage
from dipy.core.gradients import gradient_table, GradientTable
from warnings import warn


__all__ = [
    'get_sh_order_from_bval',
    'AcquisitionScheme',
    'PGSEAcquisitionScheme',
    'RotationalHarmonicsAcquisitionScheme',
    'SphericalMeanAcquisitionScheme',
    'acquisition_scheme_from_bvalues',
    'acquisition_scheme_from_qvalues',
    'acquisition_scheme_from_gradient_strengths',
    'acquisition_scheme_from_schemefile',
    'unify_length_reference_delta_Delta',
    'calculate_shell_bvalues_and_indices',
    'check_acquisition_scheme',
    'gtab_dipy2dmipy',
    'gtab_dmipy2dipy'
]


def get_sh_order_from_bval(bval):
    "Estimates minimum sh_order to represent data of given b-value."
    bvals = np.r_[2.02020202e+08, 7.07070707e+08, 1.21212121e+09,
                  2.52525253e+09, 3.13131313e+09, 5.35353535e+09,
                  np.inf]
    sh_orders = np.arange(2, 15, 2)
    return sh_orders[np.argmax(bvals > bval)]


class PGSEAcquisitionScheme:
    """
    Class that calculates and contains all information needed to simulate and
    fit data using microstructure models.
    """

    def __init__(self, bvalues, gradient_directions, qvalues,
                 gradient_strengths, delta, Delta, TE,
                 min_b_shell_distance, b0_threshold):
        self.min_b_shell_distance = float(min_b_shell_distance)
        self.b0_threshold = float(b0_threshold)
        self.bvalues = bvalues.astype(float)
        self.b0_mask = self.bvalues <= b0_threshold
        self.number_of_b0s = np.sum(self.b0_mask)
        self.number_of_measurements = len(self.bvalues)
        self.gradient_directions = gradient_directions.astype(float)
        self.qvalues = None
        if qvalues is not None:
            self.qvalues = qvalues.astype(float)
        self.gradient_strengths = None
        if gradient_strengths is not None:
            self.gradient_strengths = gradient_strengths.astype(float)
        self.delta = None
        if delta is not None:
            self.delta = delta.astype(float)
        self.Delta = None
        if Delta is not None:
            self.Delta = Delta.astype(float)
        self.TE = None
        self.N_TE = 1  # default if not given
        if TE is not None:
            self.TE = TE.astype(float)
        self.tau = None
        if self.delta is not None and self.Delta is not None:
            self.tau = Delta - delta / 3.
        # Finite-RF / minimum-echo-time bookkeeping.  ``_minimum_te`` is the
        # ideal (hard-pulse) gradient-schedule echo-time floor and ``_te_auto``
        # records that TE defaulted to it; both are set by the from_* constructors
        # after build.  Finite RF lengthens that floor:
        #   * the excitation (90 deg) always adds the full tau_exc: M_xy is born at
        #     the pulse centre and the encoding can only start once the pulse ends,
        #     a tau_exc/2 lead-in, which spin-echo symmetry mirrors into an equal
        #     tau_exc/2 trail-out before the echo (the lobes sit symmetric about the
        #     180), so 2 * tau_exc/2 = tau_exc total;
        #   * the refocusing (180 deg) adds only the part of its duration that does
        #     not fit the gradient-off window straddling it (``_refocus_gap``).  In
        #     PGSE that window is Delta - delta wide, so a typical refocusing pulse
        #     hides inside it; in OGSE the oscillating trains butt the pulse
        #     (_refocus_gap = 0), so the full tau_180 lengthens TE.
        self._tau_exc = 0.0
        self._tau_180 = 0.0
        self._refocus_gap = np.inf
        self._minimum_te = None
        self._te_auto = False
        # Coherence-pathway family, stamped by the from_* constructors; used to
        # dispatch the sequence-diagram renderer.  Default for raw construction.
        self.sequence_type = 'waveform'
        # True when the stored G(t) is the 180-folded EFFECTIVE gradient (lobes
        # after a refocusing pulse stored with flipped sign so q refocuses);
        # the diagram un-folds it to show the physical same-polarity gradient.
        self._effective_gradient = False
        self._compute_shells()

    @property
    def tau_exc(self):
        """Excitation (90 deg) RF pulse duration in seconds (0 = ideal hard pulse).

        Setting a finite duration on a scheme whose echo time defaulted to the
        gradient-schedule minimum extends that minimum (by the full tau_exc for the
        excitation -- tau_exc/2 lead-in mirrored by tau_exc/2 trail-out -- plus any
        refocusing spill-over); if the echo time was set explicitly, a value that
        would push the minimum above it raises instead.
        """
        return self._tau_exc

    @tau_exc.setter
    def tau_exc(self, value):
        self._tau_exc = float(value)
        self._apply_finite_rf()

    @property
    def tau_180(self):
        """Refocusing (180 deg) RF pulse duration in seconds (0 = ideal hard pulse).

        Only the part of the pulse that does not fit the gradient-off window
        straddling it (``_refocus_gap``) lengthens the echo time: in PGSE the
        Delta - delta inter-lobe gap absorbs a typical pulse, whereas in OGSE the
        trains butt the pulse and the full duration is added.
        """
        return self._tau_180

    @tau_180.setter
    def tau_180(self, value):
        self._tau_180 = float(value)
        self._apply_finite_rf()

    def _effective_minimum_te(self):
        """Echo-time floor including any finite-RF lengthening, or None."""
        if self._minimum_te is None:
            return None
        refocus = max(0.0, self._tau_180 - self._refocus_gap)
        return self._minimum_te + self._tau_exc + refocus

    def _apply_finite_rf(self):
        """Re-apply the finite-RF echo-time floor after a tau_* change."""
        te_min = self._effective_minimum_te()
        if te_min is None:
            return  # GRE/CPMG and freeform schemes: TE is defined explicitly
        if self._te_auto:
            self.TE = np.full(self.number_of_measurements, te_min)
            self._compute_shells()
        elif self.TE is not None and np.any(
                np.asarray(self.TE) < te_min - _TE_FLOOR_ATOL):
            raise ValueError(
                "Echo time TE = {:.3f} ms is below the minimum {:.3f} ms once "
                "the finite RF pulses (tau_exc = {:.3f} ms, refocusing "
                "spill-over = {:.3f} ms) are included.".format(
                    float(np.min(self.TE)) * 1e3, te_min * 1e3,
                    self._tau_exc * 1e3,
                    max(0.0, self._tau_180 - self._refocus_gap) * 1e3))

    def _compute_shells(self):
        """Compute (or recompute) shell clustering and all derived properties.

        Groups measurements by (delta, Delta, TE, oscillation_frequency) and
        then clusters b-values within each group.  Including
        oscillation_frequency ensures that PGSE and OGSE shells at the same
        b-value are never merged into one shell.

        Called automatically at the end of __init__ and whenever OGSE fields
        are set after construction (e.g. from_ogse, concatenate).
        """
        bvalues = self.bvalues
        b0_threshold = self.b0_threshold
        min_b_shell_distance = self.min_b_shell_distance

        if self.number_of_measurements > 1:
            # Build grouping columns from whatever timing/encoding fields are set
            # (delta, Delta, TE, oscillation_frequency). Sequences without PGSE
            # timing (OGSE, GRE) have delta/Delta = None, so filter None out — an
            # un-filtered None becomes a 0-d array and breaks the column_stack.
            osc_freq = getattr(self, 'oscillation_frequency', None)
            cols = [c for c in (self.delta, self.Delta, self.TE, osc_freq)
                    if c is not None]
            if cols:
                deltas = np.column_stack(cols)
            else:
                deltas = np.c_[np.zeros(len(bvalues))]

            unique_deltas = np.unique(deltas, axis=0)
            self.shell_indices = np.zeros(len(bvalues), dtype=int)
            self.shell_bvalues = []
            max_index = 0
            for unique_deltas_ in unique_deltas:
                delta_mask = np.all(deltas == unique_deltas_, axis=1)
                masked_bvals = bvalues[delta_mask]
                if len(masked_bvals) > 1:
                    shell_indices_, shell_bvalues_ = (
                        calculate_shell_bvalues_and_indices(
                            masked_bvals, min_b_shell_distance))
                else:
                    shell_indices_, shell_bvalues_ = np.array(0), masked_bvals
                self.shell_indices[delta_mask] = shell_indices_ + max_index
                self.shell_bvalues.append(shell_bvalues_)
                max_index = max(self.shell_indices + 1)
            self.shell_bvalues = np.hstack(self.shell_bvalues)
            self.shell_b0_mask = self.shell_bvalues <= b0_threshold

            first_indices = [
                np.argmax(self.shell_indices == ind)
                for ind in np.arange(self.shell_indices.max() + 1)]
            self.shell_qvalues = None
            if self.qvalues is not None:
                self.shell_qvalues = self.qvalues[first_indices]
            self.shell_gradient_strengths = None
            if self.gradient_strengths is not None:
                self.shell_gradient_strengths = (
                    self.gradient_strengths[first_indices])
            # Timing contract: `delta`/`Delta` are PER-MEASUREMENT (length
            # number_of_measurements) and are what a model's __call__ reads;
            # `shell_delta`/`shell_Delta` are the PER-SHELL representatives
            # (length N_shells), for per-shell code paths (spherical_mean /
            # rotational_harmonics). The SphericalMeanAcquisitionScheme is
            # already per-shell and stores those values under `delta`/`Delta`
            # with no `shell_delta` -- see its docstring.
            self.shell_delta = None
            if self.delta is not None:
                self.shell_delta = self.delta[first_indices]
            self.shell_Delta = None
            if self.Delta is not None:
                self.shell_Delta = self.Delta[first_indices]
            self.shell_TE = None
            if self.TE is not None:
                self.shell_TE = self.TE[first_indices]
                if (len(np.unique(self.TE)) != len(np.unique(
                        self.TE[self.b0_mask]))):
                    msg = "Not every TE shell has b0 measurements.\n"
                    msg += "This is required to properly normalize the signal."
                    msg += " Make sure the TE values for b0-measurements have "
                    msg += "not defaulted to 0 for example."
                    raise ValueError(msg)
                self.N_TE = len(self.shell_TE)
            # Per-shell mixing time TM (stimulated-echo longitudinal storage);
            # None for spin-echo schemes. Carried so the longitudinal-relaxation
            # factor applies in the spherical-mean path just like TE does.
            self.shell_TM = None
            _tm = getattr(self, 'TM', None)
            if _tm is not None:
                self.shell_TM = np.asarray(_tm)[first_indices]
            # Per-shell transverse occupancy time tau_perp (the window over which
            # magnetisation is transverse: the two STE encoding lobes = 2*delta).
            # None when unset -> the transverse gate falls back to TE (spin echo,
            # where the whole echo is transverse). Carried like shell_TM/shell_TE.
            self.shell_tau_perp = None
            _tp = getattr(self, 'tau_perp', None)
            if _tp is not None:
                self.shell_tau_perp = np.asarray(_tp)[first_indices]
        else:
            self.shell_bvalues = bvalues
            self.shell_indices = np.r_[int(0)]
            if self.shell_bvalues > b0_threshold:
                self.shell_b0_mask = np.r_[False]
            else:
                self.shell_b0_mask = np.r_[True]
            self.shell_qvalues = self.qvalues
            self.shell_gradient_strengths = self.gradient_strengths
            self.shell_delta = self.delta
            self.shell_Delta = self.Delta
            self.shell_TE = self.TE
            self.shell_TM = getattr(self, 'TM', None)
            self.shell_tau_perp = getattr(self, 'tau_perp', None)

        self.unique_b0_indices = np.unique(self.shell_indices[self.b0_mask])
        self.unique_dwi_indices = np.unique(self.shell_indices[~self.b0_mask])
        self.unique_shell_indices = np.unique(self.shell_indices)
        self.N_b0_shells = len(self.unique_b0_indices)
        self.N_dwi_shells = len(self.unique_dwi_indices)
        self.N_shells = len(self.unique_shell_indices)
        self.shell_sh_matrices = {}
        self.shell_sh_orders = {}
        for shell_index in self.unique_b0_indices:
            self.shell_sh_orders[shell_index] = 0
        for shell_index in self.unique_dwi_indices:
            shell_mask = self.shell_indices == shell_index
            bvecs_shell = self.gradient_directions[shell_mask]
            _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
            self.shell_sh_orders[shell_index] = get_sh_order_from_bval(
                self.shell_bvalues[shell_index])
            self.shell_sh_matrices[shell_index] = real_sym_sh_mrtrix(
                self.shell_sh_orders[shell_index], theta_, phi_, legacy=False)[0]
        if sum(self.b0_mask) == 0:
            msg = "No b0 measurements were detected. Check if the b0_threshold"
            msg += " option is high enough, or if there is a mistake in the "
            msg += "acquisition design."
            warn(msg)

        self.spherical_mean_scheme = SphericalMeanAcquisitionScheme(
            self.shell_bvalues,
            self.shell_qvalues,
            self.shell_gradient_strengths,
            self.shell_Delta,
            self.shell_delta,
            self.shell_TE,
            self.shell_TM,
            self.shell_tau_perp)
        if len(self.unique_dwi_indices) > 0:
            self.rotational_harmonics_scheme = (
                RotationalHarmonicsAcquisitionScheme(self))

    @property
    def shell_fingerprints(self):
        """Per-shell identifier: list of (b_value_s_m2, oscillation_freq_hz).

        Index i corresponds to shell_indices == i.  For PGSE schemes the
        frequency is always 0.0.  Used by TissueResponseModel to match shells
        across different AcquisitionScheme objects.
        """
        return [(float(self.shell_bvalues[i]), 0.0)
                for i in self.unique_shell_indices]

    @property
    def print_acquisition_info(self):
        """
        prints a small summary of the acquisition scheme. Is useful to check if
        the function correctly separated the shells and if the input parameters
        were given in the right scale.
        """
        print("Acquisition scheme summary\n")
        print("total number of measurements: {}".format(
            self.number_of_measurements))
        print("number of b0 measurements: {}".format(self.number_of_b0s))
        print("number of DWI shells: {}\n".format(
            np.sum(~self.shell_b0_mask)))
        upper_line = "shell_index |# of DWIs |bvalue [s/mm^2] "
        upper_line += "|gradient strength [mT/m] |delta [ms] |Delta[ms]"
        upper_line += " |TE[ms]"
        print(upper_line)
        for ind in np.arange(max(self.shell_indices) + 1):
            if (self.shell_TE is not None and
                self.shell_delta is not None and
                    self.shell_Delta is not None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        int(1e3 * self.shell_gradient_strengths[ind]),
                        self.shell_delta[ind] * 1e3,
                        self.shell_Delta[ind] * 1e3, self.shell_TE[ind] * 1e3))
            elif (self.shell_TE is None and
                  self.shell_delta is not None and
                    self.shell_Delta is not None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        int(1e3 * self.shell_gradient_strengths[ind]),
                        self.shell_delta[ind] * 1e3,
                        self.shell_Delta[ind] * 1e3, 'N/A'))
            elif (self.shell_TE is None and
                  self.shell_delta is None and
                    self.shell_Delta is not None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        'N/A', 'N/A', self.shell_Delta[ind] * 1e3, 'N/A'))
            elif (self.shell_TE is None and
                  self.shell_delta is not None and
                    self.shell_Delta is None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        'N/A', self.shell_delta[ind] * 1e3, 'N/A', 'N/A'))
            elif (self.shell_TE is None and
                  self.shell_delta is None and
                    self.shell_Delta is None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        'N/A', 'N/A', 'N/A', 'N/A'))

    def btensor(self):
        """Reconstruct the b-tensor from PGSE parameters: B[m] = bvalues[m] * n[m]⊗n[m].

        For PGSE the b-tensor is rank-1 by construction.  This method makes
        PGSEAcquisitionScheme API-compatible with AcquisitionScheme.btensor()
        so that Gaussian signal models can use a single B-tensor code path for
        both PGSE and arbitrary rotating waveforms.

        Returns
        -------
        B : ndarray, shape (n_m, 3, 3), float64
            B_ij[m] = bvalues[m] * gradient_directions[m, i]
                                  * gradient_directions[m, j]
        """
        n = self.gradient_directions   # (n_m, 3)
        b = self.bvalues               # (n_m,)
        return b[:, None, None] * np.einsum('mi,mj->mij', n, n)

    def to_gradient_array(self, n_t=1000):
        """Convert PGSE scheme to a freeform gradient array for Monte Carlo simulation.

        Returns
        -------
        G : np.ndarray, shape (n_measurements, n_t, 3), float32, T/m
            Gradient waveforms encoding the PGSE pulse structure.
        dt : float
            Uniform time step in seconds. Total duration = Delta + delta.

        Raises
        ------
        ValueError
            If delta, Delta, or gradient_strengths are None.
        ValueError
            If delta or Delta are not uniform across measurements.
        """
        if self.delta is None or self.Delta is None or self.gradient_strengths is None:
            raise ValueError(
                "to_gradient_array() requires delta, Delta, and gradient_strengths. "
                "Build the scheme with acquisition_scheme_from_gradient_strengths()."
            )
        delta_tol = np.float32(1e-6)
        if (np.max(self.delta) - np.min(self.delta)) > delta_tol:
            raise ValueError(
                "to_gradient_array() requires uniform delta across measurements. "
                f"Got range [{self.delta.min():.6g}, {self.delta.max():.6g}] s."
            )
        if (np.max(self.Delta) - np.min(self.Delta)) > delta_tol:
            raise ValueError(
                "to_gradient_array() requires uniform Delta across measurements. "
                f"Got range [{self.Delta.min():.6g}, {self.Delta.max():.6g}] s."
            )
        delta = float(self.delta[0])
        Delta = float(self.Delta[0])
        T_total = Delta + delta
        dt = T_total / (n_t - 1)
        n_pulse = max(1, round(delta / dt))
        n_Delta = round(Delta / dt)

        n_m = self.number_of_measurements
        G = np.zeros((n_m, n_t, 3), dtype=np.float32)
        for m in range(n_m):
            g_vec = (self.gradient_strengths[m] *
                     self.gradient_directions[m]).astype(np.float32)
            G[m, :n_pulse, :] = g_vec
            G[m, n_Delta:n_Delta + n_pulse, :] = -g_vec
        return G, float(dt)

    def to_schemefile(self, filename):
        """
        Exports acquisition scheme information in schemefile format, which can
        be used by the Camino Monte-Carlo simulator.

        Parameters
        ----------
        filename : string,
            location at which to save the schemefile.
        """
        TE_ = self.TE
        if TE_ is None:
            TE_ = self.Delta + 2 * self.delta + 0.001
        schemefile_data = np.hstack(
            [self.gradient_directions,
             self.gradient_strengths[:, None],
             self.Delta[:, None],
             self.delta[:, None],
             TE_[:, None]])
        header = "#g_x  g_y  g_z  |G| DELTA delta TE\n"
        header += "VERSION: STEJSKALTANNER"
        np.savetxt(filename, schemefile_data,
                   header=header, comments='')

    def visualise_acquisition_G_Delta_rainbow(
            self,
            Delta_start=None, Delta_end=None, G_start=None, G_end=None,
            bval_isolines=np.r_[0, 250, 1000, 2500, 5000, 7500, 10000, 14000],
            alpha_shading=0.6
    ):
        """This function visualizes a q-tau acquisition scheme as a function of
        gradient strength and pulse separation (big_delta). It represents every
        measurements at its G and big_delta position regardless of b-vector,
        with a background of b-value isolines for reference. It assumes there
        is only one unique pulse length (small_delta) in the acquisition
        scheme.

        Parameters
        ----------
        Delta_start : float,
            optional minimum big_delta that is plotted in seconds
        Delta_end : float,
            optional maximum big_delta that is plotted in seconds
        G_start : float,
            optional minimum gradient strength that is plotted in T/m
        G_end : float,
            optional maximum gradient strength taht is plotted in T/m
        bval_isolines : array,
            optional array of bvalue isolines that are plotted in background
            given in s/mm^2
        alpha_shading : float between [0-1]
            optional shading of the bvalue colors in the background
        """
        Delta = self.Delta  # in seconds
        delta = self.delta  # in seconds
        G = self.gradient_strengths  # in SI units T/m

        if len(np.unique(delta)) > 1:
            msg = "This acquisition has multiple small_delta values. "
            msg += "This visualization assumes there is only one small_delta."
            raise ValueError(msg)

        if Delta_start is None:
            Delta_start = 0.005
        if Delta_end is None:
            Delta_end = Delta.max() + 0.004
        if G_start is None:
            G_start = 0.
        if G_end is None:
            G_end = G.max() + .05

        Delta_ = np.linspace(Delta_start, Delta_end, 50)
        G_ = np.linspace(G_start, G_end, 50)
        Delta_grid, G_grid = np.meshgrid(Delta_, G_)
        bvals_ = b_from_g(G_grid.ravel(), delta[0], Delta_grid.ravel()) / 1e6
        bvals_ = bvals_.reshape(G_grid.shape)

        # local import because matplotlib is not in the strict requirements.
        import matplotlib.pyplot as plt
        plt.contourf(Delta_, G_, bvals_,
                     levels=bval_isolines,
                     cmap='rainbow', alpha=alpha_shading)
        cb = plt.colorbar(spacing="proportional")
        cb.ax.tick_params(labelsize=16)
        plt.scatter(Delta, G, c='k', s=25)

        plt.xlim(Delta_start, Delta_end)
        plt.ylim(G_start, G_end)
        cb.set_label('b-value ($s$/$mm^2$)', fontsize=18)
        plt.xlabel(r'Pulse Separation $\Delta$ [sec]', fontsize=18)
        plt.ylabel('Gradient Strength [T/m]', fontsize=18)

    def return_pruned_acquisition_scheme(self, shell_indices, data=None):
        """Returns pruned acquisition scheme and optionally also prunes data.

        Parameters
        ----------
        shell_indices: list of integers,
            the shell indices that correspond with the shells that should be
            returned. For the zeroth and second shell this is e.g. [0, 2]
        data: NDarray,
            DW-data that corresponds with the acquisition scheme. If it is
            given, then the data is pruned the same way as the acquisition
            scheme, meaning the pruned scheme and data can be used and fitted
            together again.

        Returns
        -------
        pruned_scheme: PGSEAcquisitionScheme object,
            the pruned acquisition scheme
        pruned_data: NDarray,
            the pruned data corresponding to the acquisition scheme.
        """
        booleans = []
        for index in shell_indices:
            booleans.append(self.shell_indices == index)
        mask = np.any(booleans, axis=0)

        bvals = self.bvalues[mask]
        gradient_directions = self.gradient_directions[mask]
        delta = self.delta[mask]
        Delta = self.Delta[mask]
        if self.TE is not None:
            TE = self.TE[mask]
        else:
            TE = None

        pruned_scheme = acquisition_scheme_from_bvalues(
            bvals, gradient_directions, delta, Delta, TE,
            self.min_b_shell_distance, self.b0_threshold)
        if data is None:
            return pruned_scheme
        else:
            pruned_data = data[..., mask]
            return pruned_scheme, pruned_data


# Backward compatibility alias
DmipyAcquisitionScheme = PGSEAcquisitionScheme


# ---------------------------------------------------------------------------
# Waveform-first acquisition scheme (ADR-001 option A)
# ---------------------------------------------------------------------------

_WaveformView = namedtuple('_WaveformView', ['G', 'dt', 'echo_idx'])


def _trap_profile(t, start, delta, eps):
    """Unit trapezoid amplitude (0..1) sampled at times ``t`` (seconds).

    Ramps 0->1 over ``eps``, holds, ramps 1->0 over ``eps``, with the two ramp
    MIDPOINTS at ``start`` and ``start + delta`` -- i.e. ``delta`` is the
    half-amplitude (50%) width and the lobe physically spans ``delta + eps``.  A
    symmetric trapezoid placed this way has area ``delta`` (matching a rectangle
    of width delta), so the b-value and diffusion time are referenced to the ramp
    midpoints, the standard slew convention.  ``eps <= 0`` gives a rectangle.
    """
    if eps <= 0:
        return ((t >= start) & (t < start + delta)).astype(np.float64)
    a = np.zeros_like(t, dtype=np.float64)
    up = (t >= start) & (t < start + eps)
    a[up] = (t[up] - start) / eps
    flat = (t >= start + eps) & (t < start + delta)
    a[flat] = 1.0
    dn = (t >= start + delta) & (t < start + delta + eps)
    a[dn] = 1.0 - (t[dn] - (start + delta)) / eps
    return a


def _trap_cosine_profile(t, sigma, f, slew, g_mag):
    """Trapezoidal (flat-top, slew-limited) cosine-OGSE amplitude, T/m.

    A triangle carrier aligned with the cosine (peak +1 at t=0, -1 at the
    half-period) is scaled so its slope equals the slew rate and then *clipped*
    at +/- g_mag: the transitions become straight slew ramps and the extrema are
    genuine flat plateaus -- the trapezoidal "minimum achievable rise time" OGSE
    of Drobnjak 2016 (for N=1 this is one +/- pair, i.e. PGSE).  A leading and
    trailing slew ramp tapers the train to zero at the ends; the carrier is
    zero outside ``[0, sigma]``.
    """
    P = 1.0 / f
    phi = (f * t) % 1.0
    tri = 1.0 - 4.0 * np.minimum(phi, 1.0 - phi)        # +1 at peak, -1 at trough
    trap = np.clip((slew * P / 4.0) * tri, -g_mag, g_mag)
    ramp = g_mag / slew
    env = np.clip(t / ramp, 0.0, 1.0) * np.clip((sigma - t) / ramp, 0.0, 1.0)
    env = np.where((t >= 0) & (t < sigma), env, 0.0)
    return trap * env


def _refocusing_residual(G, dt):
    """Relative net gradient moment ``max|q(TE)| / max|q|`` for one measurement.

    ``q(t) = integral G dt`` (the stored G is the effective, 180-folded gradient).
    A moment-nulled (refocused) waveform has ``q(TE) = 0`` so stationary spins
    rephase at the echo; the residual is ~0.  A non-nulled waveform leaves a net
    ``q(TE)`` -- dmipy-sim then dephases the ensemble to little/no signal at TE,
    while the analytical model still returns a (meaningless) b-tensor value.
    """
    G = np.asarray(G, dtype=np.float64)
    q = np.cumsum(G * dt, axis=0)
    qmax = float(np.max(np.abs(q)))
    if qmax <= 0.0:
        return 0.0
    return float(np.max(np.abs(q[-1]))) / qmax


def _calc_b_from_waveform(G, dt):
    """Compute b-values from gradient waveform using trapezoidal integration.

    Implements b = γ² ∫₀^T |q(t)|² dt, where q(t) = γ ∫₀^t G(t') dt'.
    Matches dmipy_sim.waveforms.calc_b() within discretization error (~0.16%
    for PGSE at n_t=1000; see SC-013 in governance/scientific_ledger.yaml).

    Parameters
    ----------
    G : ndarray, shape (n_m, n_t, 3), T/m
    dt : float, seconds

    Returns
    -------
    b : ndarray, shape (n_m,), s/m²
    """
    gamma = CONSTANTS['water_gyromagnetic_ratio']
    G_f64 = np.asarray(G, dtype=np.float64)
    q = np.cumsum(G_f64 * dt, axis=1) * gamma   # (n_m, n_t, 3) rad/m
    q_sq = np.sum(q ** 2, axis=2)               # (n_m, n_t)
    b = np.trapezoid(q_sq, dx=dt, axis=1)       # (n_m,) s/m²
    return b.astype(np.float64)


class AcquisitionScheme(PGSEAcquisitionScheme):
    """Waveform-first acquisition scheme (ADR-001 option A).

    Extends PGSEAcquisitionScheme by storing the gradient waveform G(t) as
    the canonical representation. This enables one scheme object for both:

    - Analytical signal models: via inherited bvalues, gradient_directions,
      shell_indices, shell_sh_matrices, etc. (PGSEAcquisitionScheme).
    - Monte Carlo simulation: via .waveform → dmipy_sim.simulate().

    Primary state: G (n_m, n_t, 3) float32 T/m, dt (float, seconds).

    Construct with class methods:
        AcquisitionScheme.from_pgse(bvalues, gradient_directions, delta, Delta)
        AcquisitionScheme.from_waveform(G, dt, gradient_directions, ...)

    The legacy factory functions (acquisition_scheme_from_bvalues etc.) still
    return PGSEAcquisitionScheme for full backward compatibility.
    """

    def __init__(self, G, dt, bvalues, gradient_directions, qvalues,
                 gradient_strengths, delta, Delta, TE,
                 min_b_shell_distance, b0_threshold,
                 oscillation_frequency=None,
                 gradient_rise_time=None,
                 n_oscillation_cycles=None):
        super().__init__(bvalues, gradient_directions, qvalues,
                         gradient_strengths, delta, Delta, TE,
                         min_b_shell_distance, b0_threshold)
        self._G = np.asarray(G, dtype=np.float32)
        self._dt = float(dt)
        # OGSE-specific per-measurement fields (None for pure PGSE schemes)
        self.oscillation_frequency = oscillation_frequency
        self.gradient_rise_time = gradient_rise_time
        self.n_oscillation_cycles = n_oscillation_cycles
        # Re-cluster shells now that oscillation_frequency is known, so PGSE
        # and OGSE measurements at the same b-value get separate shell indices.
        if oscillation_frequency is not None and np.any(oscillation_frequency > 0):
            self._compute_shells()

    @property
    def is_ogse(self):
        """Boolean array (n_m,): True where oscillation_frequency > 0."""
        if self.oscillation_frequency is None:
            return np.zeros(self.number_of_measurements, dtype=bool)
        return self.oscillation_frequency > 0

    @property
    def shell_fingerprints(self):
        """Per-shell identifier: list of (b_value_s_m2, oscillation_freq_hz).

        Overrides PGSEAcquisitionScheme to include the mean oscillation
        frequency per shell, so PGSE and OGSE shells at the same b-value
        have distinct fingerprints.
        """
        osc = self.oscillation_frequency
        fps = []
        for idx in self.unique_shell_indices:
            mask = self.shell_indices == idx
            b = float(self.shell_bvalues[idx])
            freq = float(np.mean(osc[mask])) if osc is not None else 0.0
            fps.append((b, freq))
        return fps

    @property
    def waveform(self):
        """Freeform gradient waveform view for Monte Carlo simulation.

        Returns a named tuple with attributes:
          G        : (n_m, n_t, 3) float32 ndarray, T/m
          dt       : float, seconds
          echo_idx : int — last timestep (echo assumed at end of sequence)

        Accepted by dmipy_sim.simulate(n_walkers, D, scheme, geometry) directly.
        """
        return _WaveformView(G=self._G, dt=self._dt,
                             echo_idx=self._G.shape[1] - 1)

    @property
    def refocusing_residual(self):
        """Worst-case relative net gradient moment ``max|q(TE)|/max|q|`` over all
        measurements -- a consistency guard between the engines.

        ~0 means the gradient is moment-nulled (the spin/stimulated echo forms and
        the analytical b / b-tensor is meaningful).  A large value means the
        waveform does not refocus: dmipy-sim will dephase to little or no signal
        at TE, whereas dmipy-fit would still emit a (physically meaningless)
        attenuation.  Use it to assert a loaded free waveform actually refocuses.
        """
        G = np.asarray(self._G)
        return max((_refocusing_residual(G[m], self._dt)
                    for m in range(G.shape[0])), default=0.0)

    # Physical flags a sim Sequence may carry; copied verbatim onto the wrapped
    # analytical scheme (oscillation_* are passed through __init__, not here).
    _SEQ_FLAGS = (
        'sequence_type', '_minimum_te', '_te_auto', '_refocus_gap',
        '_effective_gradient', '_ogse_two_train', '_refocus_duration',
        'TM', 'tau_perp_SE', 'ste_flip_angles', '_ramp_time',
        'cpmg_n_echoes', 'cpmg_TE', 'cpmg_beta_deg', 'n_t_per_echo', 'refocused',
        '_refocus_idx',
    )

    @classmethod
    def _wrap_sequence(cls, seq, min_b_shell_distance, b0_threshold):
        """Build an analytical AcquisitionScheme from a physical dmipy-sim Sequence.

        The forward-truth waveform / encoding (G, b, q, gradient strengths, timing)
        is generated by dmipy-sim; this wraps it with the analytical shell / SH /
        rotational-harmonics layer.  Because the Sequence's arrays are bit-identical
        to the historical inline construction, the resulting scheme is unchanged
        (locked by dmipy-sim/tests/test_sequences_parity.py).
        """
        osc = getattr(seq, 'oscillation_frequency', None)
        scheme = cls(seq.G, seq.dt, seq.bvalues, seq.gradient_directions,
                     seq.qvalues, seq.gradient_strengths, seq.delta, seq.Delta,
                     seq.TE, min_b_shell_distance, b0_threshold,
                     oscillation_frequency=osc,
                     gradient_rise_time=getattr(seq, 'gradient_rise_time', None),
                     n_oscillation_cycles=getattr(seq, 'n_oscillation_cycles', None))
        for attr in cls._SEQ_FLAGS:
            if hasattr(seq, attr):
                setattr(scheme, attr, getattr(seq, attr))
        return scheme

    @classmethod
    def from_pgse(cls, bvalues, gradient_directions, delta, Delta, TE=None,
                  n_t=1000, slew_rate=np.inf, min_b_shell_distance=50e6,
                  b0_threshold=10e6):
        """Build AcquisitionScheme from PGSE parameters.

        ``slew_rate`` (T/m/s), if given, makes the gradient lobes trapezoidal:
        each lobe ramps at this slew rate, with the ramp midpoints at the nominal
        delta/Delta edges (the half-amplitude convention), so the diffusion timing
        is preserved and the lobe physically occupies delta + G/slew_rate.  The
        gradient is then rescaled so the numerically integrated b-value still
        equals the requested target.  Default (None) gives ideal rectangular lobes.

        Supports multi-shell acquisitions with varying delta/Delta per
        measurement. The waveform uses T_total = max(delta + Delta) across
        all measurements so every pulse fits within the time window.

        Parameters
        ----------
        bvalues : array, shape (n_m,), s/m²
        gradient_directions : array, shape (n_m, 3)
        delta : float or array, shape (n_m,), seconds — pulse duration (δ)
        Delta : float or array, shape (n_m,), seconds — pulse separation (Δ)
        TE : float, array, or None — echo time(s) in seconds
        n_t : int — waveform timesteps (default 1000)
        min_b_shell_distance, b0_threshold : float — shell clustering params

        Returns
        -------
        AcquisitionScheme
        """
        # Forward-truth waveform generated by dmipy-sim (the physical sequence
        # owner); this scheme adds the analytical shell/SH layer on top.
        from dmipy_sim.sequences import Sequence as _Sequence
        seq = _Sequence.from_pgse(bvalues, gradient_directions, delta, Delta,
                                  TE=TE, n_t=n_t, slew_rate=slew_rate)
        return cls._wrap_sequence(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_pgste(cls, bvalues, gradient_directions, delta, TM, TE=None,
                   n_t=1000, slew_rate=np.inf, min_b_shell_distance=50e6,
                   b0_threshold=10e6):
        """Build a PGSTE (pulsed-gradient stimulated-echo) AcquisitionScheme.

        The stimulated echo splits the diffusion encoding around a mixing time
        ``TM`` during which the magnetisation is stored longitudinally: a
        dephasing gradient lobe of duration ``delta``, then the ``TM`` storage
        window, then a rephasing lobe.  The effective pulse separation is
        ``Delta = delta + TM`` (the diffusion time spans the storage), so the
        b-value / q-value encoding is that of the equivalent PGSE.  The scheme
        additionally carries ``TM``, which activates the longitudinal
        :class:`~dmipy_fit.signal_models.attenuation.LongitudinalRelaxation`
        factor ($\\exp(-\\mathrm{TM}/T_1)$).

        The magnetisation is transverse only during the two encoding lobes, so the
        transverse occupancy time is ``tau_perp = 2*delta``; the ``TM`` window carries
        no transverse relaxation or surface relaxivity, only $T_1$.  These are two
        distinct physical times and are stored separately: ``tau_perp`` gates the
        transverse factors ($T_2$ / surface relaxivity), while ``TE`` is the true
        echo time.  In the idealised zero-width-pulse limit the echo forms after
        both encoding lobes (transverse, ``2*delta``) plus the storage window
        (longitudinal, ``TM``), so ``TE`` defaults to ``2*delta + TM``; pass ``TE``
        to override.

        Instantaneous (hard) RF pulses only -- no finite-pulse or flip-angle
        parameters.  The stimulated echo's constant amplitude factor is absorbed
        by the global signal scale (``S0_global``) on the fit path and is not
        applied here.

        Parameters
        ----------
        bvalues : array, shape (n_m,), s/m^2
        gradient_directions : array, shape (n_m, 3)
        delta : float, seconds -- encoding pulse duration (delta)
        TM : float, seconds -- mixing (longitudinal storage) time
        TE : float, array, or None -- echo time(s); default ``2*delta + TM``
        n_t : int -- waveform timesteps (default 1000)
        min_b_shell_distance, b0_threshold : float -- shell clustering params

        Returns
        -------
        AcquisitionScheme with ``TM`` set, ``Delta = delta + TM``, the transverse
        occupancy time ``tau_perp = 2*delta`` and the echo time ``TE``
        (``2*delta + TM`` by default).
        """
        bvalues = np.asarray(bvalues, dtype=float)
        n_m = len(bvalues)
        delta_arr = np.full(n_m, float(delta))
        Delta_arr = np.full(n_m, float(delta) + float(TM))
        # Two distinct physical times: the transverse occupancy is the two encoding
        # lobes (2*delta); the echo time is 2*delta transverse + TM longitudinal in
        # the zero-width-pulse limit. Do NOT conflate them onto one TE field.
        tau_perp_val = 2.0 * float(delta)           # STE transverse occupancy time
        te_default = 2.0 * float(delta) + float(TM)  # echo time
        try:
            from dmipy_sim.sequences import Sequence  # noqa: F401  (availability check)
            # Forward-truth encoding waveform from dmipy-sim (via the PGSE lobes at
            # Delta = delta + TM); the STE-specific transverse time and TM are set
            # below so the transverse factors are gated to the encoding only.
            scheme = cls.from_pgse(
                bvalues, gradient_directions, delta_arr, Delta_arr, TE=None,
                n_t=n_t, slew_rate=slew_rate,
                min_b_shell_distance=min_b_shell_distance,
                b0_threshold=b0_threshold)
        except ImportError:
            # dmipy-sim unavailable: build the pure-PGSE (waveform-free) scheme so
            # the analytical factors still work without the simulator dependency.
            scheme = acquisition_scheme_from_bvalues(
                bvalues, gradient_directions, delta_arr, Delta_arr, TE=None,
                min_b_shell_distance=min_b_shell_distance,
                b0_threshold=b0_threshold)
        te_val = te_default if TE is None else TE
        scheme.TE = np.full(n_m, float(te_val)) if np.ndim(te_val) == 0 \
            else np.asarray(te_val, dtype=float)
        scheme._te_auto = False
        scheme.TM = np.full(n_m, float(TM))
        scheme.tau_perp = np.full(n_m, tau_perp_val)
        scheme._compute_shells()
        return scheme

    @classmethod
    def from_cpmg(cls, n_echoes, TE, bvalues=None, gradient_directions=None,
                  beta_deg=180.0, n_t_per_echo=100,
                  min_b_shell_distance=50e6, b0_threshold=10e6):
        """Build a CPMG (multi-echo spin echo) AcquisitionScheme.

        A CPMG train is a spin echo (the static off-resonance field IS
        refocused, ``refocused = True`` → $\\Xi_{\\rm IA}=1$, no frequency shift),
        but with a *train* of $N$ refocusing pulses spaced by ``TE``: the
        measurement axis is the echo index, with per-echo time $(k{+}1)\\,$TE.
        With ideal refocusing the signal is the per-compartment multi-exponential
        $\\sum_c f_c\\exp(-(k{+}1)\\TE/T_{2,c})$ — exactly what the analytic
        :class:`UnifiedWhiteMatterModel` evaluates from the per-measurement ``TE``
        array (so $\\beta=180^\\circ$ CPMG needs no special analytic branch); for
        an imperfect refocusing angle the Monte Carlo replays the EPG coherence
        pathways (``dmipy_sim.waveforms.cpmg`` / ``apply_cpmg_with_relaxation``).

        The bipolar diffusion gradient ($+G$ first half, $-G$ second half of each
        echo period) is built so the primary pathway refocuses at every echo.

        Parameters
        ----------
        n_echoes : int — number of refocusing pulses / echoes (the measurement axis).
        TE : float — echo spacing in seconds.
        bvalues : float, array, or None — diffusion weighting (per echo); ``None``/0
            gives an unweighted relaxometry train.
        gradient_directions : array (n_echoes, 3) or None — diffusion directions.
        beta_deg : float — refocusing flip angle; 180 = ideal CPMG.
        n_t_per_echo : int — waveform steps per echo period (even).

        Returns
        -------
        AcquisitionScheme with per-echo ``TE`` and CPMG markers
        (``cpmg_n_echoes``, ``cpmg_TE``, ``cpmg_beta_deg``, ``n_t_per_echo``);
        ``refocused = True``.
        """
        from dmipy_sim.sequences import Sequence as _Sequence
        seq = _Sequence.from_cpmg(n_echoes, TE, bvalues=bvalues,
                                  gradient_directions=gradient_directions,
                                  beta_deg=beta_deg, n_t_per_echo=n_t_per_echo)
        return cls._wrap_sequence(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_waveform(cls, G, dt, gradient_directions, delta=None, Delta=None,
                      TE=None, allow_unrefocused=False,
                      min_b_shell_distance=50e6, b0_threshold=10e6):
        """Build AcquisitionScheme from arbitrary gradient waveform.

        b-values are computed numerically from the waveform via
        _calc_b_from_waveform(). For PGSE at n_t=1000 this introduces ~0.16%
        systematic error vs the analytic formula (SC-013). Models that require
        qvalues or gradient_strengths (e.g. C3CylinderCallaghanApproximation,
        which needs tau = Delta - delta/3) require delta and Delta to be set.

        Parameters
        ----------
        G : array, shape (n_m, n_t, 3), T/m
        dt : float, seconds — uniform timestep
        gradient_directions : array, shape (n_m, 3) — unit direction vectors
        delta : float, array, or None — δ in seconds (needed for qvalues)
        Delta : float, array, or None — Δ in seconds (needed for qvalues)
        TE : float, array, or None — echo time(s) in seconds

        Returns
        -------
        AcquisitionScheme
        """
        from dmipy_sim.sequences import Sequence as _Sequence
        # dmipy-sim builds the physical sequence and performs the moment-nulling
        # (refocusing) guard; this wraps it with the analytical layer.
        seq = _Sequence.from_waveform(G, dt, gradient_directions, delta=delta,
                                      Delta=Delta, TE=TE,
                                      allow_unrefocused=allow_unrefocused)
        return cls._wrap_sequence(seq, min_b_shell_distance, b0_threshold)


    @classmethod
    def from_ogse(cls, bvalues, gradient_directions, oscillation_frequency,
                  gradient_duration, n_cycles=1, gradient_rise_time=0.,
                  TE=None, n_t=1000, slew_rate=np.inf, refocus_duration=0.0,
                  min_b_shell_distance=50e6, b0_threshold=10e6):
        """Build AcquisitionScheme from cosine OGSE parameters.

        Parameters
        ----------
        bvalues : array (n_m,), s/m²
        gradient_directions : array (n_m, 3)
        oscillation_frequency : float or array (n_m,), Hz
        gradient_duration : float or array (n_m,), seconds — full cosine window σ
        n_cycles : int or array (n_m,) — number of oscillation cycles N
        gradient_rise_time : float or array (n_m,), seconds — t_r (0 = pure cosine)
        TE : float, array, or None
        n_t : int — waveform timesteps
        min_b_shell_distance, b0_threshold : float — shell clustering params

        Returns
        -------
        AcquisitionScheme with OGSE fields set.

        Notes
        -----
        b-value for pure cosine OGSE (Xu 2009):
            b = γ²G²σ / (4π²f²)
        Solving for G: G = sqrt(b * 4π²f² / (γ²σ))

        RF / 180 convention (OGSE = Oscillating Gradient SPIN ECHO — it HAS a 180):
        the stored ``G(t)`` is the EFFECTIVE (180-folded) gradient. There is an implicit
        90 excitation at t=0, and a 180 refocusing pulse:
          * slew-limited two-train (``slew_rate`` set): the 180 sits in the gradient-OFF
            GAP between the two oscillating lobes and pushes them apart by
            ``refocus_duration`` — i.e. the 180 is at ``σ + refocus_duration/2``, NOT at
            TE/2. The post-180 lobe is stored sign-flipped (``_effective_gradient=True``,
            ``_ogse_two_train=True``, ``_refocus_duration``); un-fold it (negate after the
            gap) to recover the physical same-sign cosine.
          * ideal single cosine (``slew_rate=np.inf``, the idealized instantaneous
            limit; fit's default): no gap; the 180 is at the cosine centre σ/2. A
            per-walker Bloch replay must place the 180 / un-fold accordingly
            (dmipy_sim.pulse_sequence._build_ogse).
        """
        from dmipy_sim.sequences import Sequence as _Sequence
        seq = _Sequence.from_ogse(
            bvalues, gradient_directions, oscillation_frequency,
            gradient_duration, n_cycles=n_cycles,
            gradient_rise_time=gradient_rise_time, TE=TE, n_t=n_t,
            slew_rate=slew_rate, refocus_duration=refocus_duration)
        scheme = cls._wrap_sequence(seq, min_b_shell_distance, b0_threshold)
        # gradient_duration is read by OGSE signal models; carried separately
        # (not in _SEQ_FLAGS since __init__ takes the other oscillation fields).
        scheme.gradient_duration = seq.gradient_duration
        return scheme

    @classmethod
    def from_btensor_ste(cls, bvalues, delta, Delta, TE=None, n_t=1000,
                         min_b_shell_distance=50e6, b0_threshold=10e6):
        """Build a Spherical Tensor Encoding (STE) AcquisitionScheme (b_delta=0).

        Uses three sequential bipolar gradient pairs, one per Cartesian axis
        (x, y, z), played back-to-back within T_total = delta + Delta.
        Non-overlapping q(t) support guarantees B_off-diagonal = 0 exactly;
        symmetry gives B = (b/3) I and b_delta = 0.

        This is the canonical STE waveform from dmipy-sim.  It is not
        time-optimal (q-MAS waveforms achieve higher b for the same G and
        duration) but is trivially verifiable from first principles.

        RF / 180 convention: STE is run as a SPIN ECHO — an implicit 90 at t=0 and a
        180 at TE/2. The three bipolar pairs are stored as the encoding gradient; for a
        per-walker Bloch replay the post-180 part is un-folded (negated) about TE/2 so the
        180 refocuses static off-resonance. This leaves the b-tensor ISOTROPIC (b_delta=0,
        verified): the un-fold and the 180 sign flip cancel for the encoding (s*G_unfold =
        G), so B is the same isotropic tensor as the no-180 self-refocusing case. NOTE the
        bipolar pairs are SQUARE here; a realistic replay slew-limits them
        (dmipy_sim.pulse_sequence._build_btensor_ste / pedagogy._slew_limit).

        Parameters
        ----------
        bvalues : float or array, shape (n_m,), s/m²
        delta : float — gradient block duration (s); only delta+Delta matters
        Delta : float — block separation (s); only delta+Delta matters
        TE : float or None — echo time (s)
        n_t : int — waveform timesteps (default 1000)

        Returns
        -------
        AcquisitionScheme
            gradient_directions is set to [0,0,1] per measurement (nominal;
            the encoding is isotropic — use scheme.btensor() for true shape).
        """
        from dmipy_sim.sequences import Sequence as _Sequence
        seq = _Sequence.from_btensor_ste(bvalues, delta, Delta, TE=TE, n_t=n_t)
        return cls._wrap_sequence(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_btensor_pte(cls, bvalues, plane_normal, delta, Delta, TE=None,
                         n_t=1000, min_b_shell_distance=50e6, b0_threshold=10e6):
        """Build a Planar Tensor Encoding (PTE) AcquisitionScheme (b_delta=-0.5).

        Two sequential bipolar gradient pairs, one per in-plane axis (u, v),
        played back-to-back within T_total = delta + Delta.
        Non-overlapping q(t) support gives B_uv = 0 exactly; eigenvalues are
        (b/2, b/2, 0) along (u, v, plane_normal), so b_delta = -0.5.

        This is the canonical PTE waveform from dmipy-sim.

        Parameters
        ----------
        bvalues : float or array, shape (n_m,), s/m²
        plane_normal : array, shape (3,) — unit vector normal to encoding plane
        delta : float — gradient block duration (s); only delta+Delta matters
        Delta : float — block separation (s); only delta+Delta matters
        TE : float or None — echo time (s)
        n_t : int — waveform timesteps (default 1000)

        Returns
        -------
        AcquisitionScheme
            gradient_directions is set to the u-axis per measurement (nominal;
            use scheme.btensor() for the true encoding shape).
        """
        from dmipy_sim.sequences import Sequence as _Sequence
        seq = _Sequence.from_btensor_pte(bvalues, plane_normal, delta, Delta,
                                         TE=TE, n_t=n_t)
        return cls._wrap_sequence(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_btensor_waveform(cls, G, dt, *, echo_idx=None, TE=None,
                              allow_offcenter_180=False,
                              min_b_shell_distance=50e6, b0_threshold=10e6):
        """Wrap a precomputed b-tensor gradient waveform as an AcquisitionScheme.

        For an externally designed b-tensor encoding (e.g. a dmipy-design
        ``design_waveform`` output) instead of the canonical square bipolar
        pairs.  ``G`` is the EFFECTIVE gradient (180 folded in, q(TE)=0).  Pass the
        design's ``echo_idx`` so the Bloch un-fold/180 sit exactly where the design
        placed them (defaults to TE/2).  A non-TE/2 180 is guarded — it raises
        unless ``allow_offcenter_180=True`` (an off-centre 180 refocuses static
        field at 2·t_180, not at TE).  The b-tensor shape is whatever the gradient
        numbers produce (use ``scheme.btensor()``).  See
        ``dmipy_sim.sequences.Sequence.from_btensor_waveform``.
        """
        from dmipy_sim.sequences import Sequence as _Sequence
        seq = _Sequence.from_btensor_waveform(
            G, dt, echo_idx=echo_idx, TE=TE, allow_offcenter_180=allow_offcenter_180)
        return cls._wrap_sequence(seq, min_b_shell_distance, b0_threshold)

    def gamma_lm(self, l_max=4):
        """Compute angular power spectrum Gamma_lm = integral |G(t)|^2 Y_lm(Ghat(t)) dt.

        Returns the real SH expansion of the gradient power spectrum for each
        measurement. Only l=0 and l=2 are returned (6 coefficients total, ordered
        Y00, Y2-2, Y2-1, Y20, Y21, Y22), since higher orders do not contribute to
        the cylinder GPA signal in the fast-eigenmode limit.

        For measurements without a waveform (PGSEAcquisitionScheme only), or
        measurements with all-zero gradients, the coefficients are set to zero.

        The integration is performed via Riemann sum:
            Gamma_lm ~= sum_t |G(t)|^2 * Y_lm(Ghat(t)) * dt

        Parameters
        ----------
        l_max : int, optional
            Maximum SH order to compute. Currently only l=0 and l=2 are
            implemented regardless of l_max. Default 4.

        Returns
        -------
        gamma_lm : ndarray, shape (n_m, 6), float64
            Columns correspond to: Y00, Y2-2, Y2-1, Y20, Y21, Y22.
        """
        if not hasattr(self, '_gamma_lm_cache'):
            self._gamma_lm_cache = {}
        if l_max in self._gamma_lm_cache:
            return self._gamma_lm_cache[l_max]

        n_m = self.number_of_measurements
        # 6 coefficients: Y00, Y2-2, Y2-1, Y20, Y21, Y22
        result = np.zeros((n_m, 6), dtype=np.float64)

        G = np.asarray(self._G, dtype=np.float64)  # (n_m, n_t, 3)
        dt = float(self._dt)
        n_t = G.shape[1]

        for m in range(n_m):
            G_m = G[m]  # (n_t, 3)
            G_mag = np.linalg.norm(G_m, axis=-1)  # (n_t,)
            nonzero = G_mag > 0.0
            if not np.any(nonzero):
                continue  # all-zero waveform — leave gamma_lm = 0

            # Unit direction Ghat(t): only at non-zero timesteps
            # At zero-gradient timesteps the integrand is zero anyway,
            # so we can set an arbitrary direction (e.g. x).
            Ghat = np.zeros_like(G_m)
            Ghat[nonzero] = G_m[nonzero] / G_mag[nonzero, None]
            # Magnitude squared at each timestep (integrand weight)
            G2 = G_mag ** 2  # (n_t,)

            x = Ghat[:, 0]
            y = Ghat[:, 1]
            z = Ghat[:, 2]

            # Real SH basis evaluated at Ghat(t):
            # Y_00 = 1/sqrt(4*pi)
            Y00 = np.full(n_t, 1.0 / np.sqrt(4.0 * np.pi))
            # Y_2,-2 = sqrt(15/(4*pi)) * x*y
            Y2m2 = np.sqrt(15.0 / (4.0 * np.pi)) * x * y
            # Y_2,-1 = sqrt(15/(4*pi)) * y*z
            Y2m1 = np.sqrt(15.0 / (4.0 * np.pi)) * y * z
            # Y_2, 0 = sqrt(5/(16*pi)) * (2*z^2 - x^2 - y^2)
            Y20 = np.sqrt(5.0 / (16.0 * np.pi)) * (2.0 * z**2 - x**2 - y**2)
            # Y_2, 1 = sqrt(15/(4*pi)) * x*z
            Y21 = np.sqrt(15.0 / (4.0 * np.pi)) * x * z
            # Y_2, 2 = sqrt(15/(16*pi)) * (x^2 - y^2)
            Y22 = np.sqrt(15.0 / (16.0 * np.pi)) * (x**2 - y**2)

            result[m, 0] = np.sum(G2 * Y00) * dt
            result[m, 1] = np.sum(G2 * Y2m2) * dt
            result[m, 2] = np.sum(G2 * Y2m1) * dt
            result[m, 3] = np.sum(G2 * Y20) * dt
            result[m, 4] = np.sum(G2 * Y21) * dt
            result[m, 5] = np.sum(G2 * Y22) * dt

        self._gamma_lm_cache[l_max] = result
        return result

    def btensor(self):
        """Compute the b-tensor B_ij = γ² ∫ q_i(t) q_j(t) dt for each measurement.

        Returns
        -------
        B : ndarray, shape (n_m, 3, 3), float64
            B_ij in s/m².  For PGSE along x: B ≈ b × [[1,0,0],[0,0,0],[0,0,0]].
            For STE: B ≈ (b_trace/3) × I₃.
        """
        if hasattr(self, '_btensor_cache'):
            return self._btensor_cache

        gamma = CONSTANTS['water_gyromagnetic_ratio']
        G = np.asarray(self._G, dtype=np.float64)   # (n_m, n_t, 3)
        dt = float(self._dt)
        q = np.cumsum(G * dt, axis=1) * gamma        # (n_m, n_t, 3) rad/m
        # B_ij = ∫ q_i q_j dt ≈ sum_t q_i(t) q_j(t) * dt
        # Use np.einsum for efficiency: (n_m, n_t, 3) outer on last dim
        B = np.einsum('mti,mtj->mij', q, q) * dt    # (n_m, 3, 3)
        self._btensor_cache = B
        return B

    @classmethod
    def concatenate(cls, schemes):
        """Concatenate multiple AcquisitionScheme objects along the measurement axis.

        Parameters
        ----------
        schemes : list of AcquisitionScheme instances

        Returns
        -------
        AcquisitionScheme with all measurements concatenated.

        Notes
        -----
        For fields present only in some schemes (e.g. oscillation_frequency),
        PGSE schemes are filled with zeros.
        """
        if not schemes:
            raise ValueError("concatenate() requires at least one scheme.")

        # Determine shared n_t from the maximum
        max_n_t = max(s._G.shape[1] for s in schemes)
        dt = schemes[0]._dt  # Use first scheme's dt (should be compatible)

        # Concatenate waveforms (pad shorter waveforms with zeros)
        G_parts = []
        for s in schemes:
            n_t_s = s._G.shape[1]
            if n_t_s < max_n_t:
                pad = np.zeros(
                    (s._G.shape[0], max_n_t - n_t_s, 3), dtype=np.float32)
                G_parts.append(np.concatenate([s._G, pad], axis=1))
            else:
                G_parts.append(s._G)
        G_cat = np.concatenate(G_parts, axis=0)

        def _cat_or_none(attr):
            arrays = [getattr(s, attr, None) for s in schemes]
            if all(a is None for a in arrays):
                return None
            # Fill None with zeros
            filled = []
            for s, a in zip(schemes, arrays):
                n_m = s.number_of_measurements
                if a is None:
                    filled.append(np.zeros(n_m, dtype=np.float64))
                else:
                    filled.append(np.asarray(a, dtype=np.float64))
            return np.concatenate(filled)

        bvalues = np.concatenate([s.bvalues for s in schemes])
        gradient_directions = np.concatenate(
            [s.gradient_directions for s in schemes])
        qvalues = _cat_or_none('qvalues')
        gradient_strengths = _cat_or_none('gradient_strengths')
        delta = _cat_or_none('delta')
        Delta = _cat_or_none('Delta')
        TE = _cat_or_none('TE')

        # Recalculate bvalues from concatenated waveform for accuracy
        bvalues_num = _calc_b_from_waveform(G_cat, dt)
        # Use original bvalues for shell clustering (more reliable for PGSE)
        # but keep numerically-derived ones if no bvalues present

        min_b_shell_distance = schemes[0].min_b_shell_distance
        b0_threshold = schemes[0].b0_threshold

        inst = cls.__new__(cls)
        super(AcquisitionScheme, inst).__init__(
            bvalues, gradient_directions, qvalues, gradient_strengths,
            delta if not np.all(delta == 0) else None,
            Delta if not np.all(Delta == 0) else None,
            TE if TE is not None and not np.all(TE == 0) else None,
            min_b_shell_distance, b0_threshold)

        inst._G = G_cat
        inst._dt = dt

        # OGSE fields: concatenate, filling PGSE measurements with 0
        osc_freqs = []
        rise_times = []
        n_cycles_list = []
        for s in schemes:
            n_m = s.number_of_measurements
            if s.oscillation_frequency is not None:
                osc_freqs.append(s.oscillation_frequency)
            else:
                osc_freqs.append(np.zeros(n_m))
            if s.gradient_rise_time is not None:
                rise_times.append(s.gradient_rise_time)
            else:
                rise_times.append(np.zeros(n_m))
            if s.n_oscillation_cycles is not None:
                n_cycles_list.append(s.n_oscillation_cycles)
            else:
                n_cycles_list.append(np.zeros(n_m))

        # Only set OGSE fields if any scheme has them
        has_ogse = any(
            s.oscillation_frequency is not None for s in schemes)
        if has_ogse:
            inst.oscillation_frequency = np.concatenate(osc_freqs)
            inst.gradient_rise_time = np.concatenate(rise_times)
            inst.n_oscillation_cycles = np.concatenate(n_cycles_list)

            # gradient_duration: concatenate (fill PGSE with zeros)
            g_durations = []
            for s in schemes:
                n_m_s = s.number_of_measurements
                gd = getattr(s, 'gradient_duration', None)
                if gd is not None:
                    g_durations.append(np.asarray(gd, dtype=np.float64))
                else:
                    g_durations.append(np.zeros(n_m_s))
            inst.gradient_duration = np.concatenate(g_durations)
        else:
            inst.oscillation_frequency = None
            inst.gradient_rise_time = None
            inst.n_oscillation_cycles = None
            inst.gradient_duration = None

        # ---- per-measurement coherence-gating attributes ----
        # These were silently dropped before, which broke "scheme = scheme + atom" for
        # the gating atoms: a PGSE + PGSTE union lost the mixing time (T1 became
        # insensitive), an STE/finite-pulse union lost its transverse time, and a
        # multi-field / multi-angle union reverted to the single defaults.
        # TM (stimulated-echo storage time): absent -> 0 (no longitudinal storage, so
        # exp(-TM/T1)=1, the correct PGSE limit).
        inst.TM = _cat_or_none('TM')
        # tau_perp_SE (transverse-occupancy time): absent -> that scheme's TE, so the
        # T2 / surface-relaxivity gate uses the correct transverse time (not the full TE).
        if any(getattr(s, 'tau_perp_SE', None) is not None for s in schemes):
            parts = []
            for s in schemes:
                n_m = s.number_of_measurements
                tp = getattr(s, 'tau_perp_SE', None)
                if tp is None:
                    te = getattr(s, 'TE', None)
                    tp = (np.full(n_m, np.nan) if te is None
                          else np.broadcast_to(np.asarray(te, float), (n_m,)).copy())
                else:
                    tp = np.broadcast_to(np.asarray(tp, float), (n_m,)).copy()
                parts.append(tp)
            inst.tau_perp_SE = np.concatenate(parts)
        else:
            inst.tau_perp_SE = None
        # tau_perp (transverse occupancy time; STE encoding = 2*delta): absent ->
        # that scheme's TE (spin echo: the whole echo is transverse), so the T2 /
        # surface-relaxivity gate keeps the correct transverse time across a union.
        if any(getattr(s, 'tau_perp', None) is not None for s in schemes):
            parts = []
            for s in schemes:
                n_m = s.number_of_measurements
                tp = getattr(s, 'tau_perp', None)
                if tp is None:
                    te = getattr(s, 'TE', None)
                    tp = (np.full(n_m, np.nan) if te is None
                          else np.broadcast_to(np.asarray(te, float), (n_m,)).copy())
                else:
                    tp = np.broadcast_to(np.asarray(tp, float), (n_m,)).copy()
                parts.append(tp)
            inst.tau_perp = np.concatenate(parts)
        else:
            inst.tau_perp = None

        # Re-cluster with oscillation_frequency in grouping key when OGSE present
        if has_ogse:
            inst._compute_shells()

        return inst

    def __add__(self, other):
        return AcquisitionScheme.concatenate([self, other])

    def __iadd__(self, other):
        merged = AcquisitionScheme.concatenate([self, other])
        self.__dict__.update(merged.__dict__)
        return self


class RotationalHarmonicsAcquisitionScheme:
    """
    AcquisitionScheme instance that contains the information necessary to
    calculate the rotational harmonics for a model for every acquisition shell.
    It is instantiated using a regular PGSEAcquisitionScheme and
    N_angular_samples determines how many samples are taken between mu=[0., 0.]
    and mu=[np.pi/2, 0.].

    Parameters
    ----------
    dmipy_acquisition_scheme: PGSEAcquisitionScheme instance
        An acquisition scheme that has been instantiated using dMipy.
    N_angular_samples: int
        Integer representing the number of angular samples per shell.
    """

    def __init__(self, dmipy_acquisition_scheme, N_angular_samples=10):
        self.Nsamples = N_angular_samples
        scheme = dmipy_acquisition_scheme

        thetas = np.linspace(0, np.pi / 2, N_angular_samples)
        r = np.ones(N_angular_samples)
        phis = np.zeros(N_angular_samples)
        angles = np.c_[r, thetas, phis]
        angles_cart = utils.sphere2cart(angles)

        b_all_shells = []
        Gdirs_all_shells = []
        delta_all_shells = []
        Delta_all_shells = []
        shell_indices = []
        for shell_index in scheme.unique_dwi_indices:
            b = scheme.shell_bvalues[shell_index]
            b_all_shells.append(np.tile(b, N_angular_samples))
            if scheme.shell_delta is not None:
                delta = scheme.shell_delta[shell_index]
                delta_all_shells.append(np.tile(delta, N_angular_samples))
            if scheme.shell_Delta is not None:
                Delta = scheme.shell_Delta[shell_index]
                Delta_all_shells.append(np.tile(Delta, N_angular_samples))
            Gdirs_all_shells.append(angles_cart)
            shell_indices.append(np.tile(shell_index, N_angular_samples))

        self.shell_indices = np.hstack(shell_indices)
        self.bvalues = np.hstack(b_all_shells)
        self.gradient_directions = np.vstack(Gdirs_all_shells)
        self.delta = None
        if scheme.shell_delta is not None:
            self.delta = np.hstack(delta_all_shells)
        self.Delta = None
        if scheme.shell_Delta is not None:
            self.Delta = np.hstack(Delta_all_shells)
        if self.delta is not None and self.Delta is not None:
            self.gradient_strengths = g_from_b(
                self.bvalues,
                self.delta,
                self.Delta)
            self.qvalues = q_from_g(
                self.gradient_strengths,
                self.delta)
            self.tau = self.Delta - self.delta / 3.0
        else:
            self.gradient_strengths = self.qvalues = self.tau = None
        self.b0_mask = np.tile(False, len(self.bvalues))
        self.TE = None
        self.shell_delta = scheme.shell_delta
        self.shell_Delta = scheme.shell_Delta
        self.unique_b0_indices = scheme.unique_b0_indices
        self.unique_shell_indices = scheme.unique_shell_indices
        self.unique_dwi_indices = scheme.unique_dwi_indices
        self.N_b0_shells = len(self.unique_b0_indices)
        self.N_dwi_shells = len(self.unique_dwi_indices)
        self.N_shells = len(self.unique_shell_indices)
        self.number_of_measurements = len(self.bvalues)

        self.shell_sh_matrices = {}
        self.shell_sh_orders = {}
        for shell_index in scheme.unique_dwi_indices:
            self.shell_sh_orders[shell_index] = int(
                scheme.shell_sh_orders[shell_index])
            self.shell_sh_matrices[shell_index] = real_sym_sh_mrtrix(
                self.shell_sh_orders[shell_index], thetas, phis, legacy=False)[0]

        self.inverse_rh_matrix = {
            rh_order: np.linalg.pinv(real_sym_rh_basis(
                rh_order, thetas, phis
            )) for rh_order in np.arange(0, 15, 2)
        }

    def btensor(self):
        """Reconstruct B-tensor from PGSE parameters: B[m] = bvalues[m] * n[m]⊗n[m]."""
        n = self.gradient_directions   # (n_m, 3)
        b = self.bvalues               # (n_m,)
        return b[:, None, None] * np.einsum('mi,mj->mij', n, n)


class SphericalMeanAcquisitionScheme:
    r"""Acquisition scheme for spherical-mean models -- already reduced to one
    entry per shell.

    Timing-attribute contract (see also ``PGSEAcquisitionScheme``)
    -------------------------------------------------------------
    A model's ``__call__`` runs **per measurement**, so it must read the
    per-measurement timings ``.delta`` / ``.Delta`` (and ``.bvalues`` /
    ``.qvalues`` / ...), which every scheme type exposes. On this scheme each
    "measurement" **is** a shell, so ``.delta`` / ``.Delta`` here already hold the
    per-shell values (constructed from the full scheme's ``shell_delta`` /
    ``shell_Delta``) and ``number_of_measurements == N_shells``.

    This scheme deliberately does **not** define ``shell_delta`` / ``shell_Delta``
    -- those are the *per-shell representatives of a finer per-measurement scheme*
    (the full ``AcquisitionScheme`` and the ``RotationalHarmonicsAcquisitionScheme``
    carry them). A ``__call__`` that reaches for ``acquisition_scheme.shell_delta``
    is a category error: it works on the full/RH scheme but raises here. Use
    ``.delta`` / ``.Delta`` and derive any unique-timing grouping locally
    (``np.unique([delta, Delta], axis=1)``). Per-shell code paths that always
    receive a full/RH scheme -- ``spherical_mean`` / ``rotational_harmonics_
    representation`` overrides -- may use ``shell_delta`` / ``shell_Delta``.
    """

    def __init__(self, bvalues, qvalues,
                 gradient_strengths, Deltas, deltas, TE=None, TM=None,
                 tau_perp=None):
        self.bvalues = bvalues
        self.qvalues = qvalues
        self.gradient_strengths = gradient_strengths
        self.Delta = Deltas
        self.delta = deltas
        # Per-shell TE so occupancy-gated relaxation factors (T2 / surface
        # relaxivity) apply in the spherical-mean path just as they do in the full
        # signal. None when the scheme has no TE; diffusion-only models ignore it.
        self.TE = TE
        # Per-shell mixing time TM so the longitudinal-relaxation factor
        # (exp(-TM/T1)) applies in the spherical-mean path; None for spin echo.
        self.TM = TM
        # Per-shell transverse occupancy time tau_perp so the transverse gate uses
        # the STE encoding window (2*delta) rather than TE in the spherical-mean
        # path; None (spin echo) -> the gate falls back to TE.
        self.tau_perp = tau_perp
        self.number_of_measurements = len(bvalues)


def acquisition_scheme_from_bvalues(
        bvalues, gradient_directions, delta=None, Delta=None, TE=None,
        min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Creates an acquisition scheme object from bvalues, gradient directions,
    pulse duration $\delta$ and pulse separation time $\Delta$.

    Parameters
    ----------
    bvalues: 1D numpy array of shape (Ndata)
        bvalues of the acquisition in s/m^2.
        e.g., a bvalue of 1000 s/mm^2 must be entered as 1000 * 1e6 s/m^2
    gradient_directions: 2D numpy array of shape (Ndata, 3)
        gradient directions array of cartesian unit vectors.
    delta: float or 1D numpy array of shape (Ndata)
        if float, pulse duration of every measurements in seconds.
        if array, potentially varying pulse duration per measurement.
    Delta: float or 1D numpy array of shape (Ndata)
        if float, pulse separation time of every measurements in seconds.
        if array, potentially varying pulse separation time per measurement.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    PGSEAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    # Unit sanity: bvalues must be s/m^2 (1000 s/mm^2 = 1e9 s/m^2). A max nonzero
    # bvalue below ~1e5 s/m^2 (= 0.1 s/mm^2) is physically implausible for DWI and
    # almost always means s/mm^2 was passed raw -- which would be silently
    # misclassified as all-b0 (b0_threshold ~ 1e7). Warn loudly.
    _bv = np.atleast_1d(np.asarray(bvalues, dtype=float))
    _bmax = float(_bv.max()) if _bv.size else 0.0
    if 0.0 < _bmax < 1e5:
        warn("acquisition_scheme_from_bvalues: max b-value is {:.3g} s/m^2, far below "
             "any real DWI shell. bvalues must be in s/m^2 (multiply s/mm^2 by 1e6, e.g. "
             "1000 s/mm^2 -> 1e9). As given they will be treated as b0 and any fit will "
             "be meaningless.".format(_bmax), UserWarning, stacklevel=2)
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(
        bvalues, delta, Delta, TE)
    check_acquisition_scheme(
        bvalues, gradient_directions, delta_, Delta_, TE_)
    if delta is not None and Delta is not None:
        qvalues = q_from_b(bvalues, delta_, Delta_)
        gradient_strengths = g_from_b(bvalues, delta_, Delta_)
    else:
        qvalues = gradient_strengths = None
    return PGSEAcquisitionScheme(bvalues, gradient_directions, qvalues,
                                  gradient_strengths, delta_, Delta_, TE_,
                                  min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_qvalues(
        qvalues, gradient_directions, delta, Delta, TE=None,
        min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Creates an acquisition scheme object from qvalues, gradient directions,
    pulse duration $\delta$ and pulse separation time $\Delta$.

    Parameters
    ----------
    qvalues: 1D numpy array of shape (Ndata)
        diffusion sensitization of the acquisition in 1/m.
        e.g. a qvalue of 10 1/mm must be entered as 10 * 1e3 1/m
    gradient_directions: 2D numpy array of shape (Ndata, 3)
        gradient directions array of cartesian unit vectors.
    delta: float or 1D numpy array of shape (Ndata)
        if float, pulse duration of every measurements in seconds.
        if array, potentially varying pulse duration per measurement.
    Delta: float or 1D numpy array of shape (Ndata)
        if float, pulse separation time of every measurements in seconds.
        if array, potentially varying pulse separation time per measurement.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    PGSEAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(
        qvalues, delta, Delta, TE)
    check_acquisition_scheme(
        qvalues, gradient_directions, delta_, Delta_, TE_)
    bvalues = b_from_q(qvalues, delta, Delta)
    gradient_strengths = g_from_q(qvalues, delta)
    return PGSEAcquisitionScheme(bvalues, gradient_directions, qvalues,
                                  gradient_strengths, delta_, Delta_, TE_,
                                  min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_gradient_strengths(
        gradient_strengths, gradient_directions, delta, Delta, TE=None,
        min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Creates an acquisition scheme object from gradient strengths, gradient
    directions pulse duration $\delta$ and pulse separation time $\Delta$.

    Parameters
    ----------
    gradient_strengths: 1D numpy array of shape (Ndata)
        gradient strength of the acquisition in T/m.
        e.g., a gradient strength of 300 mT/m must be entered as 300 / 1e3 T/m
    gradient_directions: 2D numpy array of shape (Ndata, 3)
        gradient directions array of cartesian unit vectors.
    delta: float or 1D numpy array of shape (Ndata)
        if float, pulse duration of every measurements in seconds.
        if array, potentially varying pulse duration per measurement.
    Delta: float or 1D numpy array of shape (Ndata)
        if float, pulse separation time of every measurements in seconds.
        if array, potentially varying pulse separation time per measurement.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    PGSEAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(
        gradient_strengths, delta, Delta, TE)
    check_acquisition_scheme(gradient_strengths, gradient_directions,
                             delta_, Delta_, TE_)
    bvalues = b_from_g(gradient_strengths, delta, Delta)
    qvalues = q_from_g(gradient_strengths, delta)
    return PGSEAcquisitionScheme(bvalues, gradient_directions, qvalues,
                                  gradient_strengths, delta_, Delta_, TE_,
                                  min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_schemefile(
        file_path, min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Created an acquisition scheme object from a Camino scheme file, containing
    gradient directions, strengths, pulse duration $\delta$ and pulse
    separation time $\Delta$ and TE.

    Parameters
    ----------
    file_path: string
        absolute file path to schemefile location
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    PGSEAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    skiprows = 0
    while True:
        try:
            scheme = np.loadtxt(file_path, skiprows=skiprows)
            break
        except ValueError:
            skiprows += 1

    bvecs = scheme[:, :3]
    bvecs[np.linalg.norm(bvecs, axis=1) == 0.] = np.r_[1., 0., 0.]
    G = scheme[:, 3]
    Delta = scheme[:, 4]
    delta = scheme[:, 5]
    TE = scheme[:, 6]
    return acquisition_scheme_from_gradient_strengths(
        G, bvecs, delta, Delta, TE, min_b_shell_distance, b0_threshold)


# Absolute tolerance (seconds) for echo-time floor comparisons.
_TE_FLOOR_ATOL = 1e-9

# Tight refocusing tolerance: real moment-nulled waveforms sit at <1e-4 relative
# net moment; a non-refocused one is ~1.  from_waveform raises above this unless
# allow_unrefocused=True.
_REFOCUS_ATOL = 1e-3


def _resolve_te(TE, t_total_min, n_m):
    """Resolve the echo time against the minimum echo time of an encoding.

    ``t_total_min`` is the time from excitation to the spin/stimulated echo set
    by the gradient schedule (PGSE: ``Delta + delta``; PGSTE: ``2*delta + TM``;
    b-tensor: ``delta + Delta``).  When ``TE`` is ``None`` the echo time defaults
    to that minimum -- the natural choice, since a longer TE only adds leading and
    trailing dead time and so signal loss.  A supplied TE below the minimum is
    unphysical (the echo cannot form before the encoding completes) and raises;
    a longer TE is accepted unchanged.

    Returns
    -------
    TE : ndarray (n_m,)
    was_auto : bool
        True when TE defaulted to the minimum.  Recorded so that attaching a
        finite excitation pulse afterwards can push the echo out by tau_exc/2.
    """
    if TE is None:
        return np.full(n_m, float(t_total_min)), True
    TE_arr = np.broadcast_to(np.asarray(TE, dtype=np.float64), (n_m,)).copy()
    if np.any(TE_arr < t_total_min - _TE_FLOOR_ATOL):
        raise ValueError(
            "Echo time TE = {:.3f} ms is below the minimum echo time "
            "{:.3f} ms set by the gradient schedule; the echo cannot form "
            "before the encoding completes.".format(
                float(np.min(TE_arr)) * 1e3, float(t_total_min) * 1e3))
    return TE_arr, False


def unify_length_reference_delta_Delta(reference_array, delta, Delta, TE):
    """
    If either delta or Delta are given as float, makes them an array the same
    size as the reference array.

    Parameters
    ----------
    reference_array : array of size (Nsamples)
        typically b-values, q-values or gradient strengths.
    delta : float or array of size (Nsamples)
        pulse duration in seconds.
    Delta : float or array of size (Nsamples)
        pulse separation in seconds.
    TE : None, float or array of size (Nsamples)
        Echo time of the acquisition in seconds.

    Returns
    -------
    delta_ : array of size (Nsamples)
        pulse duration copied to be same size as reference_array
    Delta_ : array of size (Nsamples)
        pulse separation copied to be same size as reference_array
    TE_ : None or array of size (Nsamples)
        Echo time copied to be same size as reference_array
    """
    if delta is None:
        delta_ = delta
    elif isinstance(delta, float) or isinstance(delta, int):
        delta_ = np.tile(delta, len(reference_array))
    else:
        delta_ = delta.copy()
    if Delta is None:
        Delta_ = Delta
    elif isinstance(Delta, float) or isinstance(Delta, int):
        Delta_ = np.tile(Delta, len(reference_array))
    else:
        Delta_ = Delta.copy()
    if TE is None:
        TE_ = TE
    elif isinstance(TE, float) or isinstance(TE, int):
        TE_ = np.tile(TE, len(reference_array))
    else:
        TE_ = TE.copy()
    return delta_, Delta_, TE_


def calculate_shell_bvalues_and_indices(bvalues, max_distance=20e6):
    """
    Calculates which measurements belong to different acquisition shells.
    It uses scipy's linkage clustering algorithm, which uses the max_distance
    input as a limit of including measurements in the same cluster.

    For example, if bvalues were [1, 2, 3, 4, 5] and max_distance was 1, then
    all bvalues would belong to the same cluster.
    However, if bvalues were [1, 2, 4, 5] max max_distance was 1, then this
    would result in 2 clusters.

    Parameters
    ----------
    bvalues: 1D numpy array of shape (Ndata)
        bvalues of the acquisition in s/m^2.
    max_distance: float
        maximum b-value distance for a measurement to be included in the same
        shell.

    Returns
    -------
    shell_indices: 1D numpy array of shape (Ndata)
        array of integers, starting from 0, representing to which shell a
        measurement belongs. The number itself has no meaning other than just
        being different for different shells.
    shell_bvalues: 1D numpy array of shape (Nshells)
        array of the mean bvalues for every acquisition shell.
    """
    linkage_matrix = linkage(np.c_[bvalues])
    clusters = fcluster(linkage_matrix, max_distance, criterion='distance')
    shell_indices = np.empty_like(bvalues, dtype=int)
    cluster_bvalues = np.zeros((np.max(clusters), 2))
    for ind in np.unique(clusters):
        cluster_bvalues[ind - 1] = np.mean(bvalues[clusters == ind]), ind
    shell_bvalues, ordered_cluster_indices = (
        cluster_bvalues[cluster_bvalues[:, 0].argsort()].T)
    for i, ind in enumerate(ordered_cluster_indices):
        shell_indices[clusters == ind] = i
    return shell_indices, shell_bvalues


def check_acquisition_scheme(
        bqg_values, gradient_directions, delta, Delta, TE):
    "function to check the validity of the input parameters."
    if bqg_values.ndim > 1:
        msg = "b/q/G input must be a one-dimensional array. "
        msg += "Currently its dimensions is {}.".format(
            bqg_values.ndim
        )
        raise ValueError(msg)
    if len(bqg_values) != len(gradient_directions):
        msg = "b/q/G input and gradient_directions must have the same length. "
        msg += "Currently their lengths are {} and {}.".format(
            len(bqg_values), len(gradient_directions)
        )
        raise ValueError(msg)
    if delta is not None:
        if len(bqg_values) != len(delta):
            msg = "b/q/G input and delta must have the same length. "
            msg += "Currently their lengths are {} and {}.".format(
                len(bqg_values), len(delta)
            )
            raise ValueError(msg)
        if delta.ndim > 1:
            msg = "delta must be one-dimensional array. "
            msg += "Currently its dimension is {}".format(
                delta.ndim
            )
            raise ValueError(msg)
        if np.min(delta) < 0:
            msg = "delta must be zero or positive. "
            msg += "Currently its minimum value is {}.".format(
                np.min(delta)
            )
            raise ValueError(msg)
    if Delta is not None:
        if len(bqg_values) != len(Delta):
            msg = "b/q/G input and Delta must have the same length. "
            msg += "Currently their lengths are {} and {}.".format(
                len(bqg_values), len(Delta)
            )
            raise ValueError(msg)
        if Delta.ndim > 1:
            msg = "Delta must be one-dimensional array. "
            msg += "Currently its dimension is {}.".format(
                Delta.ndim
            )
            raise ValueError(msg)
        if np.min(Delta) < 0:
            msg = "Delta must be zero or positive. "
            msg += "Currently its minimum value is {}.".format(
                np.min(Delta)
            )
            raise ValueError(msg)

    if gradient_directions.ndim != 2 or gradient_directions.shape[1] != 3:
        msg = "gradient_directions n must be two dimensional array of shape "
        msg += "[N, 3]. Currently its shape is {}.".format(
            gradient_directions.shape)
        raise ValueError(msg)
    if np.min(bqg_values) < 0.:
        msg = "b/q/G input must be zero or positive. "
        msg += "Minimum value found is {}.".format(bqg_values.min())
        raise ValueError(msg)
    gradient_norms = np.linalg.norm(gradient_directions, axis=1)
    zero_norms = gradient_norms == 0.
    if not np.all(abs(gradient_norms[~zero_norms] - 1.) < 0.001):
        msg = "gradient orientations n are not unit vectors. "
        raise ValueError(msg)
    if TE is not None and len(TE) != len(bqg_values):
        msg = "If given, TE must be same length b/q/G input."
        msg += "Currently their lengths are {} and {}.".format(
            len(TE), len(gradient_directions)
        )
    if TE is not None:
        te_min = np.min(TE)
        te_max = np.max(TE)
        if te_min < 0.005:
            warn(
                "TE minimum value {:.4f} s is below 5 ms. "
                "TE must be given in seconds. "
                "Did you accidentally provide TE in milliseconds?".format(te_min),
                UserWarning
            )
        if te_max > 0.500:
            warn(
                "TE maximum value {:.4f} s exceeds 500 ms. "
                "TE must be given in seconds. "
                "Did you accidentally provide TE in milliseconds?".format(te_max),
                UserWarning
            )


def gtab_dipy2dmipy(dipy_gradient_table, min_b_shell_distance=50e6,
                    b0_threshold=10e6):
    """Converts a dipy gradient_table to a dmipy acquisition_scheme.
    If no big_delta or small_delta is defined in the gradient table, then None
    is passed to the PGSEAcquisitionScheme for these fields, and no models
    can be used that need this information.

    Parameters
    ----------
    dipy_gradient_table: dipy GradientTable instance,
        object that contains bvals, bvecs, pulse separation and duration
        information.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    PGSEAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.

    """
    if not isinstance(dipy_gradient_table, GradientTable):
        msg = "Input must be a dipy GradientTable object. "
        raise ValueError(msg)
    bvals = dipy_gradient_table.bvals * 1e6
    bvecs = dipy_gradient_table.bvecs
    delta = dipy_gradient_table.small_delta
    Delta = dipy_gradient_table.big_delta

    if delta is None or Delta is None:
        msg = "pulse_separation (big_delta) or pulse_duration (small_delta) "
        msg += "are not defined in the Dipy gtab. This means the resulting "
        msg += "PGSEAcquisitionScheme cannot be used with CompartmentModels "
        msg += "that need these."
        warn(msg)

    gtab_dmipy = acquisition_scheme_from_bvalues(
        bvalues=bvals, gradient_directions=bvecs, delta=delta, Delta=Delta,
        min_b_shell_distance=min_b_shell_distance, b0_threshold=b0_threshold)
    return gtab_dmipy


def gtab_dmipy2dipy(dmipy_gradient_table):
    """Converts a dmipy acquisition scheme to a dipy gradient_table.

    Parameters
    ----------
    PGSEAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.

    Returns
    -------
    dipy_gradient_table: dipy GradientTable instance,
        object that contains bvals, bvecs, pulse separation and duration
        information.
    """
    if not isinstance(dmipy_gradient_table, PGSEAcquisitionScheme):
        msg = "Input must be a PGSEAcquisitionScheme object. "
        raise ValueError(msg)
    bvals = dmipy_gradient_table.bvalues / 1e6
    bvecs = dmipy_gradient_table.gradient_directions
    delta = dmipy_gradient_table.delta
    Delta = dmipy_gradient_table.Delta

    if delta is None:
        pass  # leave delta undefined in dipy gtab
    elif len(np.unique(delta)) > 1:
        msg = "Cannot create Dipy GradientTable for Acquisition schemes with "
        msg += "multiple delta (pulse duration) values, due to current "
        msg += "limitations of Dipy GradientTables."
        raise ValueError(msg)
    elif len(np.unique(delta)) == 1:
        delta = delta[0]

    if Delta is None:
        pass  # leave Delta undefined in dipy gtab
    elif len(np.unique(Delta)) > 1:
        msg = "Cannot create Dipy GradientTable for Acquisition schemes with "
        msg += "multiple Delta (pulse sepration) values, due to current "
        msg += "limitations of Dipy GradientTables."
        raise ValueError(msg)
    elif len(np.unique(Delta)) == 1:
        Delta = Delta[0]

    dipy_gradient_table = gradient_table(
        bvals=bvals, bvecs=bvecs, small_delta=delta, big_delta=Delta)
    return dipy_gradient_table
