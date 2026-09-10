import numpy as np
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
        # The legacy analytical scheme (no waveform): a PGSE family by construction.
        self.sequence_type = 'pgse'
        self._te_auto = False
        self._compute_shells()

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
# The acquisition scheme over dmipy-sim's acquisition object
# ---------------------------------------------------------------------------

def _as_protocol(sequence):
    """A ``ScannerSequence`` or a ``Protocol`` as a Protocol (one sequence: one entry, its rows ``0 .. n-1``)."""
    from dmipy_sim.acquisition.scanner_sequence import Protocol, ScannerSequence
    if isinstance(sequence, Protocol):
        return sequence
    if isinstance(sequence, ScannerSequence):
        return Protocol([sequence])
    raise TypeError("an AcquisitionScheme wraps a dmipy-sim ScannerSequence or Protocol (the acquisition object), "
                    f"got {type(sequence).__name__}")


def _gather(protocol, values, fill=None):
    """A per-measurement array over ``protocol`` in acquisition order from ``values(seq)`` per sequence -- an
    array of that sequence's ``n_meas`` rows, or ``None``. ``None`` when no sequence has a value; a sequence
    without one takes ``fill`` (a scalar, or a callable of the sequence)."""
    per = [values(seq) for seq in protocol]
    if all(v is None for v in per):
        return None
    out = None
    for seq, rows, v in zip(protocol, protocol.rows, per):
        if v is None:
            f = fill(seq) if callable(fill) else fill
            v = np.full((seq.n_meas,), np.nan if f is None else f, dtype=float)
        v = np.asarray(v, dtype=float)
        if v.ndim == 0 or v.shape[0] != seq.n_meas:
            v = np.broadcast_to(v, (seq.n_meas,) + v.shape[1:] if v.ndim else (seq.n_meas,))
        if out is None:
            out = np.empty((protocol.n_meas,) + v.shape[1:], dtype=float)
        out[rows] = v
    return out


def _enc(seq, field):
    """An ``Encoding`` field of ``seq`` (``None`` when absent)."""
    return None if seq.encoding is None else getattr(seq.encoding, field, None)


def _by_te(TE, n_m, build):
    """One sequence per distinct echo time, each holding the rows of that TE in acquisition order:
    ``build(rows, te)`` returns the ScannerSequence of ``rows`` (``te`` ``None`` = the smallest that fits). A
    single TE (or none) is one sequence; several are a Protocol that remembers the interleaving."""
    from dmipy_sim.acquisition.scanner_sequence import Protocol
    if TE is None or np.ndim(TE) == 0:
        return build(np.arange(n_m), None if TE is None else float(TE))
    TE = np.broadcast_to(np.asarray(TE, dtype=float), (n_m,))
    groups = [np.flatnonzero(TE == te) for te in np.unique(TE)]
    if len(groups) == 1:
        return build(groups[0], float(TE[0]))
    return Protocol([build(rows, float(TE[rows[0]])) for rows in groups], rows=groups)


def _rows_of(x, rows, n_m):
    """The rows ``rows`` of a per-measurement or scalar parameter."""
    if x is None or np.ndim(x) == 0:
        return x
    x = np.asarray(x)
    return x[rows] if x.shape[0] == n_m else x


class AcquisitionScheme(PGSEAcquisitionScheme):
    """The analytical acquisition scheme over dmipy-sim's acquisition object.

    A scheme wraps one :class:`dmipy_sim.acquisition.scanner_sequence.ScannerSequence` -- what the scanner does
    from the excitation to the readout: the physical gradient, the RF schedule, the readout, a timing budget,
    the per-measurement ``Encoding`` -- or a :class:`~dmipy_sim.acquisition.scanner_sequence.Protocol` of them
    (one per echo time; a scheme with several TEs is several sequences, interleaved as the acquisition was).
    Every per-measurement quantity a signal model reads (``bvalues``, ``gradient_directions``, ``delta``,
    ``Delta``, ``TE``, ``TM``, ``tau_perp``, the OGSE fields, ...) is that object's declared ``Encoding``, in
    acquisition order; the shell / SH / rotational-harmonics layer is this class's own. Nothing about the
    gradient is re-derived here: ``btensor()``, ``refocusing_residual``, the effective gradient ``_G`` the
    waveform-integrating models read, all come from the object.

    Build one from a dmipy-sim sequence (``AcquisitionScheme(seq)``) or through the constructors, which are
    dmipy-sim's builders with the shell parameters added:
    ``from_pgse``, ``from_pgste``, ``from_cpmg``, ``from_ogse``, ``from_btensor_ste``, ``from_btensor_pte``,
    ``from_waveform``, ``from_btensor_waveform``. The legacy factories (``acquisition_scheme_from_bvalues`` ...)
    return the waveform-free :class:`PGSEAcquisitionScheme`.

    Monte Carlo: ``dmipy_sim.simulate(n, D, scheme, geometry)`` reads ``scheme.waveform`` -- the object itself --
    so the analytical model and the walk see the identical acquisition.
    """

    def __init__(self, sequence, min_b_shell_distance=50e6, b0_threshold=10e6):
        P = _as_protocol(sequence)
        for seq in P:
            if seq.encoding is None:
                raise TypeError("an AcquisitionScheme reads each sequence's Encoding (b, directions, delta, ...); "
                                "build the sequence with a dmipy-sim builder or sequences.from_waveform")
        self.protocol = P
        n_m = P.n_meas
        bvalues = _gather(P, lambda s: _enc(s, "bvalues"))
        dirs = _gather(P, lambda s: _enc(s, "gradient_directions"))
        qvalues = _gather(P, lambda s: _enc(s, "qvalues"), fill=0.0)
        gs = _gather(P, lambda s: _enc(s, "gradient_strengths"), fill=0.0)
        delta = _gather(P, lambda s: _enc(s, "delta"), fill=0.0)
        Delta = _gather(P, lambda s: _enc(s, "Delta"), fill=0.0)
        TE = _gather(P, lambda s: _enc(s, "TE"), fill=lambda s: s.T)
        # the OGSE fields, before the shells are computed (they enter the shell key)
        self.oscillation_frequency = _gather(P, lambda s: _enc(s, "oscillation_frequency"), fill=0.0)
        self.gradient_rise_time = _gather(P, lambda s: _enc(s, "gradient_rise_time"), fill=0.0)
        self.n_oscillation_cycles = _gather(P, lambda s: _enc(s, "n_oscillation_cycles"), fill=0.0)
        self.gradient_duration = _gather(P, lambda s: _enc(s, "gradient_duration"), fill=0.0)
        # coherence-pathway quantities the relaxation factors gate on: the stimulated echo's storage time (0 for a
        # spin echo: exp(-TM/T1) = 1) and the time transverse (the whole echo for a spin echo)
        self.TM = _gather(P, lambda s: None if s.TM is None else np.full(s.n_meas, s.TM), fill=0.0)
        self.tau_perp = _gather(P, lambda s: _enc(s, "tau_perp_SE"), fill=lambda s: s.T)
        self.tau_perp_SE = self.tau_perp
        self.ste_flip_angles = next((_enc(s, "ste_flip_angles") for s in P if _enc(s, "ste_flip_angles") is not None), None)
        self.refocused = all(_enc(s, "refocused") is not False for s in P)
        for k in ("cpmg_n_echoes", "cpmg_TE", "cpmg_beta_deg", "n_t_per_echo"):
            setattr(self, k, next((_enc(s, k) for s in P if _enc(s, k) is not None), None))
        super().__init__(bvalues, dirs, qvalues, gs, delta, Delta, TE, min_b_shell_distance, b0_threshold)
        self.sequence_type = P[0].family if len(P) == 1 else "protocol"
        self._te_auto = any(bool(_enc(s, "te_auto")) for s in P)
        self._colinear_pgse = all(s.family in ("pgse", "pgste") for s in P)

    # ── the object ─────────────────────────────────────────────────────────────────────────────────────────
    @property
    def sequence(self):
        """The ``ScannerSequence`` when the scheme holds one, else its ``Protocol``."""
        return self.protocol[0] if len(self.protocol) == 1 else self.protocol

    @property
    def waveform(self):
        """What ``dmipy_sim.simulate`` and a pack's ``replay`` read: the acquisition object itself."""
        return self.sequence

    @property
    def timing(self):
        """The timing budget the sequences were built to (``None`` for instantaneous pulses)."""
        return self.protocol[0].timing

    def _rf_duration(self, role):
        """The duration of the first pulse of ``role`` in the schedule (0 for a hard or absent pulse)."""
        from dmipy_sim.acquisition.rf import _ROLE_OF_LABEL
        for e in self.protocol[0].rf:
            if _ROLE_OF_LABEL.get(e.label) == role:
                return float(e.duration_s)
        return 0.0

    @property
    def tau_exc(self):
        """The excitation's duration (s); 0 for an instantaneous pulse. Read from the schedule."""
        return self._rf_duration("excite")

    @property
    def tau_180(self):
        """The refocusing pulse's duration (s); 0 for an instantaneous pulse."""
        return self._rf_duration("refocus")

    @property
    def tau_90(self):
        """A stimulated echo's store / recall duration (s); 0 for an instantaneous pulse."""
        return self._rf_duration("store")

    @property
    def refocusing_residual(self):
        """The worst relative net gradient moment ``|q(TE)| / max|q|`` over the sequences (~0: refocused)."""
        return max(s.refocusing_residual for s in self.protocol)

    @property
    def _common_grid(self):
        n_t, dt = self.protocol[0].n_t, self.protocol[0].dt
        return all(s.n_t == n_t and abs(s.dt - dt) < 1e-12 * dt for s in self.protocol)

    @property
    def _grid(self):
        """The models' shared grid ``(n_t, dt)``: the sequences' own when they agree, otherwise the finest step
        over the longest duration (a sequence's gradient is held per step, zero after its readout)."""
        if self._common_grid:
            return int(self.protocol[0].n_t), float(self.protocol[0].dt)
        dt = min(float(s.dt) for s in self.protocol)
        return int(max(round(float(s.T) / dt) for s in self.protocol)) + 1, dt

    @property
    def _G(self):
        """The EFFECTIVE gradient ``(n_m, n_t, 3)`` in acquisition order on the shared grid :attr:`_grid`, for the
        models that integrate a waveform (built once, on first read); :meth:`waveform_of` reads a measurement on
        its own sequence's grid."""
        cached = self.__dict__.get("_G_cache")
        if cached is not None:
            return cached
        n_t, dt = self._grid
        out = np.zeros((self.number_of_measurements, n_t, 3), np.float32)
        for seq, rows in zip(self.protocol, self.protocol.rows):
            G = np.asarray(seq.G_eff, np.float32)
            if seq.n_t == n_t and abs(seq.dt - dt) < 1e-12 * dt:
                out[rows] = G
            else:                                              # held per step: G[k] plays over [k dt, (k+1) dt)
                k = np.floor(np.arange(n_t) * dt / float(seq.dt) + 1e-9).astype(int)
                ok = k < seq.n_t
                out[np.ix_(rows, np.flatnonzero(ok))] = G[:, k[ok]]
        self.__dict__["_G_cache"] = out
        return out

    @property
    def _dt(self):
        return self._grid[1]

    def waveform_of(self, m):
        """``(G_eff, dt)`` of measurement ``m``: its sequence's effective gradient row and step."""
        for seq, rows in zip(self.protocol, self.protocol.rows):
            hit = np.flatnonzero(rows == m)
            if hit.size:
                return np.asarray(seq.G_eff, np.float64)[int(hit[0])], float(seq.dt)
        raise IndexError(m)

    @property
    def is_ogse(self):
        """Boolean array (n_m,): True where oscillation_frequency > 0."""
        if self.oscillation_frequency is None:
            return np.zeros(self.number_of_measurements, dtype=bool)
        return self.oscillation_frequency > 0

    @property
    def shell_fingerprints(self):
        """Per-shell identifier: list of (b_value_s_m2, oscillation_freq_hz), so PGSE and OGSE shells at the same
        b-value have distinct fingerprints."""
        osc = self.oscillation_frequency
        fps = []
        for idx in self.unique_shell_indices:
            mask = self.shell_indices == idx
            b = float(self.shell_bvalues[idx])
            freq = float(np.mean(osc[mask])) if osc is not None else 0.0
            fps.append((b, freq))
        return fps

    # ── constructors: dmipy-sim's builders, the shell parameters added ─────────────────────────────────────
    @classmethod
    def from_sequence(cls, sequence, min_b_shell_distance=50e6, b0_threshold=10e6):
        """The scheme of a dmipy-sim ``ScannerSequence`` or ``Protocol``."""
        return cls(sequence, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_pgse(cls, bvalues, gradient_directions, delta, Delta, TE=None, n_t=1000, slew_rate=np.inf,
                  timing=None, min_b_shell_distance=50e6, b0_threshold=10e6):
        """PGSE through :func:`dmipy_sim.sequences.pgse`: two same-sign lobes ``Delta`` apart, the 180 midway.

        ``bvalues`` (s/m²), ``gradient_directions`` (n_m, 3); ``delta`` / ``Delta`` (s) per measurement or
        scalar; ``TE`` (s) scalar, per measurement (one sequence per distinct TE, interleaved as given) or None
        (the smallest that fits); ``slew_rate`` (T/m/s, ``np.inf`` the idealised square lobes the analytical
        models assume); ``timing`` a :class:`dmipy_sim.acquisition.timing.SequenceTiming` budget.
        """
        from dmipy_sim.sequences import pgse
        bvalues = np.asarray(bvalues, dtype=float)
        dirs = np.asarray(gradient_directions, dtype=float)
        n_m = len(bvalues)
        check_acquisition_scheme(bvalues, dirs, *unify_length_reference_delta_Delta(bvalues, delta, Delta, None)[:2], None)
        seq = _by_te(TE, n_m, lambda rows, te: pgse(
            dirs[rows], _rows_of(delta, rows, n_m), _rows_of(Delta, rows, n_m), bvalues=bvalues[rows], TE=te,
            n_t=n_t, slew_rate=slew_rate, timing=timing))
        return cls(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_pgste(cls, bvalues, gradient_directions, delta, TM, TE=None, n_t=1000, slew_rate=np.inf,
                   timing=None, ste_flip_angles=(90.0, 90.0, 90.0), min_b_shell_distance=50e6, b0_threshold=10e6):
        """PGSTE through :func:`dmipy_sim.sequences.pgste`: a dephasing lobe, longitudinal storage over ``TM``,
        the same lobe rephasing; ``Delta = delta + TM``, ``TM`` carried for the longitudinal factor and the
        transverse time ``tau_perp`` for the transverse ones."""
        from dmipy_sim.sequences import pgste
        bvalues = np.asarray(bvalues, dtype=float)
        dirs = np.asarray(gradient_directions, dtype=float)
        n_m = len(bvalues)
        seq = _by_te(TE, n_m, lambda rows, te: pgste(
            dirs[rows], _rows_of(delta, rows, n_m), TM, bvalues=bvalues[rows], TE=te, n_t=n_t,
            slew_rate=slew_rate, timing=timing, ste_flip_angles=ste_flip_angles))
        return cls(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_cpmg(cls, n_echoes, TE, bvalues=None, gradient_directions=None, beta_deg=180.0, n_t_per_echo=100,
                  polarity="constant", slew_rate=np.inf, timing=None, min_b_shell_distance=50e6, b0_threshold=10e6):
        """CPMG through :func:`dmipy_sim.sequences.cpmg`: a 90 and ``n_echoes`` refocusing pulses of ``beta_deg``
        at ``(k + 1/2) TE``, an echo read at every ``k TE``; the diffusion gradient at ``polarity`` constant or
        alternating per interval. The scheme's ``TE`` is the train's; the readout is every echo."""
        from dmipy_sim.sequences import cpmg
        seq = cpmg(n_echoes, TE, gradient_directions=gradient_directions, bvalues=bvalues, polarity=polarity,
                   beta_deg=beta_deg, n_t_per_echo=n_t_per_echo, slew_rate=slew_rate, timing=timing)
        return cls(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_waveform(cls, G, dt, gradient_directions, delta=None, Delta=None, TE=None, allow_unrefocused=False,
                      min_b_shell_distance=50e6, b0_threshold=10e6):
        """An arbitrary played gradient through :func:`dmipy_sim.sequences.from_waveform`: b numerically from
        ``G``; no pulses declared, so the gradient must refocus on its own."""
        from dmipy_sim.sequences import from_waveform
        seq = from_waveform(G, dt, gradient_directions, delta=delta, Delta=Delta, TE=TE,
                            allow_unrefocused=allow_unrefocused)
        return cls(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_ogse(cls, bvalues, gradient_directions, oscillation_frequency, gradient_duration, *, shape="cosine",
                  Delta=None, TE=None, n_t=1000, slew_rate=np.inf, timing=None,
                  min_b_shell_distance=50e6, b0_threshold=10e6):
        """OGSE through :func:`dmipy_sim.sequences.ogse`: an oscillating block of ``gradient_duration`` on each side
        of the 180 -- ``shape='cosine'`` (the frequency-selective cosine over whole periods; the analytical OGSE
        models' idealised form, so the default here) or ``'trapezoid'`` (Drobnjak's train of lobes); ``Delta``
        the block separation when stated. The block must hold whole periods (refused, not snapped)."""
        from dmipy_sim.sequences import ogse
        bvalues = np.asarray(bvalues, dtype=float)
        dirs = np.asarray(gradient_directions, dtype=float)
        n_m = len(bvalues)
        seq = _by_te(TE, n_m, lambda rows, te: ogse(
            dirs[rows], _rows_of(oscillation_frequency, rows, n_m), _rows_of(gradient_duration, rows, n_m),
            shape=shape, Delta=_rows_of(Delta, rows, n_m), bvalues=bvalues[rows], TE=te, n_t=n_t,
            slew_rate=slew_rate, timing=timing))
        return cls(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_btensor_ste(cls, bvalues, gradient_duration, TE=None, n_t=1000, slew_rate=np.inf, timing=None,
                         min_b_shell_distance=50e6, b0_threshold=10e6):
        """Spherical tensor encoding (``b_delta = 0``) through :func:`dmipy_sim.sequences.ste`: three sequential
        self-refocused pairs, one per axis, over ``gradient_duration``. ``gradient_directions`` is nominal
        ([0, 0, 1]); the encoding is isotropic -- read ``scheme.btensor()``."""
        from dmipy_sim.sequences import ste
        seq = ste(gradient_duration, bvalues=bvalues, TE=TE, n_t=n_t, slew_rate=slew_rate, timing=timing)
        return cls(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_btensor_pte(cls, bvalues, plane_normal, gradient_duration, TE=None, n_t=1000, slew_rate=np.inf,
                         timing=None, min_b_shell_distance=50e6, b0_threshold=10e6):
        """Planar tensor encoding (``b_delta = -0.5``) through :func:`dmipy_sim.sequences.pte`: two sequential
        self-refocused pairs in the plane normal to ``plane_normal``; ``gradient_directions`` is the first
        in-plane axis (nominal) -- read ``scheme.btensor()``."""
        from dmipy_sim.sequences import pte
        seq = pte(plane_normal, gradient_duration, bvalues=bvalues, TE=TE, n_t=n_t, slew_rate=slew_rate, timing=timing)
        return cls(seq, min_b_shell_distance, b0_threshold)

    @classmethod
    def from_btensor_waveform(cls, G, dt, *, echo_idx=None, TE=None, timing=None,
                              min_b_shell_distance=50e6, b0_threshold=10e6):
        """A precomputed b-tensor gradient (e.g. a dmipy-design output; the PHYSICAL gradient) as a spin echo
        through :func:`dmipy_sim.sequences.from_btensor_waveform`: the 180 at ``echo_idx`` (TE/2 by default,
        the only position at which the static field refocuses at the echo), the budget it was built to."""
        from dmipy_sim.sequences import from_btensor_waveform
        seq = from_btensor_waveform(G, dt, echo_idx=echo_idx, TE=TE, timing=timing)
        return cls(seq, min_b_shell_distance, b0_threshold)

    # ── what the waveform-integrating models read ──────────────────────────────────────────────────────────
    def gamma_lm(self, l_max=4):
        """The angular power spectrum of the effective gradient per measurement,
        ``Gamma_lm = integral |G(t)|^2 Y_lm(Ghat(t)) dt`` for l = 0 and 2 (six coefficients, ordered Y00, Y2-2,
        Y2-1, Y20, Y21, Y22): what the cylinder GPA reads in the fast-eigenmode limit. Riemann sum on each
        measurement's own grid."""
        if not hasattr(self, '_gamma_lm_cache'):
            self._gamma_lm_cache = {}
        if l_max in self._gamma_lm_cache:
            return self._gamma_lm_cache[l_max]
        n_m = self.number_of_measurements
        result = np.zeros((n_m, 6), dtype=np.float64)
        for m in range(n_m):
            G_m, dt = self.waveform_of(m)                      # (n_t, 3)
            G_mag = np.linalg.norm(G_m, axis=-1)
            nonzero = G_mag > 0.0
            if not np.any(nonzero):
                continue
            Ghat = np.zeros_like(G_m)
            Ghat[nonzero] = G_m[nonzero] / G_mag[nonzero, None]
            G2 = G_mag ** 2
            x, y, z = Ghat[:, 0], Ghat[:, 1], Ghat[:, 2]
            Y = np.stack([np.full(G_m.shape[0], 1.0 / np.sqrt(4.0 * np.pi)),
                          np.sqrt(15.0 / (4.0 * np.pi)) * x * y,
                          np.sqrt(15.0 / (4.0 * np.pi)) * y * z,
                          np.sqrt(5.0 / (16.0 * np.pi)) * (2.0 * z ** 2 - x ** 2 - y ** 2),
                          np.sqrt(15.0 / (4.0 * np.pi)) * x * z,
                          np.sqrt(15.0 / (16.0 * np.pi)) * (x ** 2 - y ** 2)])
            result[m] = (Y * G2).sum(axis=1) * dt
        self._gamma_lm_cache[l_max] = result
        return result

    def btensor(self):
        """The b-tensor ``B_ij = integral q_i q_j dt`` per measurement ``(n_m, 3, 3)`` (s/m²), in acquisition order.

        A colinear single-direction encoding (PGSE / PGSTE) has the exact rank-1 tensor ``b n n^T`` and reads
        it from the declared b (the quadrature of the discretised waveform carries an O(1/n_t) error that would
        otherwise make the anisotropic Gaussian models disagree with the analytic b); every other encoding is the
        object's own integral of its effective gradient.
        """
        if hasattr(self, '_btensor_cache'):
            return self._btensor_cache
        if getattr(self, '_colinear_pgse', False):
            self._btensor_cache = super(AcquisitionScheme, self).btensor()
            return self._btensor_cache
        B = np.empty((self.number_of_measurements, 3, 3), dtype=np.float64)
        for seq, rows in zip(self.protocol, self.protocol.rows):
            B[rows] = np.asarray(seq.btensor(), np.float64)
        self._btensor_cache = B
        return B

    def to_gradient_array(self, n_t=1000):
        """``(G_eff, dt)`` of the square PGSE with this scheme's b, directions, delta and Delta on an ``n_t`` grid,
        as the analytical layer integrates it (:func:`dmipy_sim.sequences.to_gradient_array`); one grid, so one
        sequence's delta / Delta must be uniform."""
        from dmipy_sim.sequences import to_gradient_array
        if len(self.protocol) != 1:
            raise ValueError("to_gradient_array() is one grid: take it per sequence of a multi-TE protocol")
        return to_gradient_array(self.protocol[0], n_t=n_t)

    # ── unions ─────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def concatenate(cls, schemes):
        """The scheme of several schemes' measurements, one after the other: a Protocol of all their sequences
        (each keeps its own grid; ``_G`` holds them per step on the finest one), the rows
        offset so the acquisition order is the concatenation order."""
        from dmipy_sim.acquisition.scanner_sequence import Protocol
        if not schemes:
            raise ValueError("concatenate() requires at least one scheme.")
        seqs, rows, offset = [], [], 0
        for s in schemes:
            for seq, r in zip(s.protocol, s.protocol.rows):
                seqs.append(seq)
                rows.append(np.asarray(r) + offset)
            offset += s.number_of_measurements
        return cls(Protocol(seqs, rows=rows), schemes[0].min_b_shell_distance, schemes[0].b0_threshold)

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
        # Effective diffusion time tau = Delta - delta/3, same convention as the
        # full PGSEAcquisitionScheme, so tau-dependent isotropic models (e.g. the
        # S3 Callaghan sphere) work in the spherical-mean path instead of raising
        # AttributeError on a missing `tau`.
        self.tau = None
        if deltas is not None and Deltas is not None:
            self.tau = np.asarray(Deltas, dtype=float) \
                - np.asarray(deltas, dtype=float) / 3.0
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
        True when TE defaulted to the minimum.
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
