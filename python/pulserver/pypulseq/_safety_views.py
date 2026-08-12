"""PNS and gradient-spectrum views, over the timeline or over one TR.

Over the timeline these are upstream PyPulseq's own functions, run on a window
of blocks. Under ``tr`` they come from the C safety core -- the same code the
interpreter runs at predownload -- so a picture here and a verdict there
cannot disagree.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import _results, _safety
from ._common import _UPSTREAM_WINDOW_WIDTH, _span
from ._structure import _Structure


class SafetyViewsMixin:
    """PNS and gradient-spectrum views. Mixed into :class:`Sequence`."""

    def calculate_pns(
        self,
        hardware: object,
        time_range: list[float] | None = None,
        do_plots: bool = True,
        *,
        tr: str | int | None = None,
        compat: bool = True,
    ):
        """Peripheral nerve stimulation over the sequence, or over one TR.

        Parameters
        ----------
        hardware : str or pathlib.Path or types.SimpleNamespace or dict
            Which nerve model to use, and its coefficients. A Siemens ``.asc``
            path or a per-axis namespace of the kind
            :func:`pypulseq.utils.safe_pns_prediction.safe_example_hw`
            returns selects **SAFE**, upstream's model. A mapping carrying
            ``chronaxie`` (seconds) and ``rheobase`` (Hz/m/s) selects the
            **Irnich** rheobase/chronaxie model, which is what the GE gate
            applies.
        time_range : list of float, optional
            Two timepoints in seconds bounding what to look at. Only with
            ``tr=None``.
        do_plots : bool, default True
            Draw the gradient waveform and the PNS response. The response
            panel is marked at 100 % of the stimulation threshold and at the
            80 % margin, whichever nerve model produced it -- upstream draws
            neither, only the peak the sequence happened to reach.
        tr : {"worst_case"} or int, optional
            Analyse one repetition time rather than the timeline.

            ``None``, the default, is upstream PyPulseq exactly: the sequence
            as written, played once from rest, zero-padded.

            ``"worst_case"`` is the waveform ``pulseg_check_safety`` judges --
            **not** any TR the scanner plays, but the per-sample maximum over
            every instance of one -- evaluated periodically, with the history
            wrapped round from the end of the TR so the nerve model is warmed
            up rather than starting from rest. This is the number the
            interpreter gates on.

            An integer is that TR instance as it really plays, signed
            amplitudes and all, evaluated the same periodic way. Its use is
            checking the claim ``"worst_case"`` rests on: that no instance
            exceeds the envelope.

        Returns
        -------
        ok : bool
            Whether peak PNS stays under the threshold everywhere.
        pns_norm : numpy.ndarray
            ``(N,)`` PNS over all axes, normalised to 1.
        pns_components : numpy.ndarray
            ``(N, 3)`` PNS per gradient axis, normalised to 1.
        t_pns : numpy.ndarray
            ``(N,)`` the time axis, in seconds.

        Notes
        -----
        Under ``tr=None`` this is upstream's
        :func:`pypulseq.Sequence.calc_pns.calc_pns` called on the blocks
        asked for, so a PyPulseq script gets PyPulseq's numbers.

        Under ``tr=``, the response comes from the same C code the scanner
        runs, and the returned arrays run past one TR: the circularly wrapped
        history the model needed is reported rather than trimmed away, which
        is what makes ``ok`` here the gate's own verdict.
        """
        if tr is None:
            from pypulseq.Sequence.calc_pns import calc_pns

            first, last = self._blocks_over(*_span(time_range if time_range else (0.0, np.inf)))
            answer = calc_pns(
                self._upstream_window(first, last),
                hardware,
                time_range=time_range,
                do_plots=do_plots,
            )
            if do_plots:
                # Upstream draws through pyplot and leaves its PNS panel
                # current, so the thresholds go on afterwards rather than
                # forking its plotting.
                _safety.overlay_pns_thresholds()
            return answer if compat else _results.Pns(*answer)

        if time_range is not None:
            raise ValueError(
                "calculate_pns(): time_range selects part of the timeline and tr selects a "
                "repetition time, which is not on it -- pass one or the other"
            )

        structure = self._structure_for("calculate_pns")
        mode, index = structure.resolve(tr)
        result = _native_pns(structure, hardware, mode, index)

        # The C core reports per-axis percentage of threshold; upstream
        # normalises to 1. Same quantity, hundredfold apart.
        components = 0.01 * np.stack(
            [np.asarray(result[f"slew_{axis}"], dtype=float) for axis in "xyz"], axis=-1
        )
        norm = np.sqrt((components**2).sum(axis=1))
        raster = _safety.SAFETY_RASTER_FRACTION * float(self.system.grad_raster_time)
        times = np.arange(components.shape[0]) * raster

        if do_plots:
            _plot_pns(structure.waveform(tr), components, raster)

        verdict = (bool(np.all(norm < 1)), norm, components, times)
        return verdict if compat else _results.Pns(*verdict)

    def calculate_gradient_spectrum(
        self,
        max_frequency: float = 2000.0,
        window_width: float = 0.05,
        frequency_oversampling: float = 3.0,
        time_range: list[float] | None = None,
        plot: bool = True,
        combine_mode: str = "max",
        use_derivative: bool = False,
        acoustic_resonances: list[dict] | None = None,
        *,
        tr: str | int | None = None,
        resonance_lines: bool = False,
        bands: list | None = None,
        compat: bool = True,
    ):
        """The gradient spectrum of the sequence, or of one TR.

        Parameters
        ----------
        max_frequency : float, default 2000.0
            Highest frequency to report, in Hz.
        window_width : float, default 0.05
            Length of each transformed window, in seconds. **Ignored under
            ``tr``**, where the window is the repetition time, so that what
            comes back is one spectrum rather than a spectrogram. Passing a
            value there warns rather than being quietly overridden.
        frequency_oversampling : float, default 3.0
            Zero-padding factor along frequency; higher is smoother.
        time_range : list of float, optional
            Two timepoints in seconds bounding what to look at. Only with
            ``tr=None``.
        plot : bool, default True
            Draw the spectrograms.
        combine_mode : {"max", "mean", "rss", "none"}, default "max"
            How to collapse the windows into one spectrogram. Under ``tr``
            there is only ever one window, so this decides nothing except
            whether the single column is kept ( ``"none"`` ) or dropped.
        use_derivative : bool, default False
            Transform the slew rate rather than the gradient.
        acoustic_resonances : list of dict, optional
            Resonances to mark, as ``{'frequency': ..., 'bandwidth': ...}``.
            See :func:`~._safety.bands_to_resonances`.
        tr : {"worst_case"} or int, optional
            Analyse one repetition time rather than the timeline. ``None``,
            the default, is upstream PyPulseq exactly. ``"worst_case"`` is the
            per-sample maximum over every TR instance -- the waveform
            ``pulseg_check_safety`` judges. An integer is that instance as it
            really plays. See :meth:`calculate_pns` for the full account.
        resonance_lines : bool, default False
            Also compute the C safety core's acoustic line spectrum, draw it
            over the spectrogram, and return it. Needs ``tr``.
        bands : list of tuple, optional
            Forbidden bands as ``(freq_min_hz, freq_max_hz,
            max_amplitude_hz_per_m)``, which decide which lines count as
            violations. Read only when ``resonance_lines`` is set; defaults
            to whatever ``acoustic_resonances`` describes, or to nothing.

        Returns
        -------
        spectrograms : list of numpy.ndarray
            One per gradient axis.
        spectrogram_rss : numpy.ndarray
            The axes combined in root-sum-square.
        frequencies : numpy.ndarray
            The frequency axis, in Hz.
        times : numpy.ndarray
            The time axis, meaningful only for ``combine_mode="none"``.
        resonances : ~._safety.MechResonances
            **Only when ``resonance_lines`` is set**, appended as a fifth
            element: the line spectrum at the TR harmonics ``k / T_TR``, in
            equivalent-drive units, and which of them violate a band.

        Notes
        -----
        The transform is always upstream PyPulseq's -- under ``tr`` it is
        upstream's own code run over the waveform the C core extracted, so
        what changes is which waveform is transformed and over what window,
        never how.

        **Under ``tr`` this is a spectrum, not a spectrogram.** A repetition
        time is transformed in one window because it is periodic, and a
        periodic waveform has no time-varying spectrum to chart: it has
        energy only at multiples of ``1 / T_TR``. Cutting it into shorter
        windows would only smear neighbouring harmonics into each other.

        Those harmonics are what ``resonance_lines`` draws over the result,
        on **its own vertical axis** -- see
        :class:`~._safety.MechResonances` for why the two scales must not be
        shared.
        """
        from pypulseq.Sequence.calc_grad_spectrum import calculate_gradient_spectrum

        if acoustic_resonances is None:
            acoustic_resonances = []

        if tr is None:
            if resonance_lines:
                raise ValueError(
                    "calculate_gradient_spectrum(): resonance_lines needs tr -- a line "
                    "spectrum exists only for a repetition time, not for a stretch of timeline"
                )
            first, last = self._blocks_over(*_span(time_range if time_range else (0.0, np.inf)))
            spectrum = calculate_gradient_spectrum(
                self._upstream_window(first, last),
                max_frequency=max_frequency,
                window_width=window_width,
                frequency_oversampling=frequency_oversampling,
                time_range=time_range,
                plot=plot,
                combine_mode=combine_mode,
                use_derivative=use_derivative,
                acoustic_resonances=acoustic_resonances,
            )
            return spectrum if compat else _results.GradientSpectrum(*spectrum)

        if time_range is not None:
            raise ValueError(
                "calculate_gradient_spectrum(): time_range selects part of the timeline and "
                "tr selects a repetition time, which is not on it -- pass one or the other"
            )

        structure = self._structure_for("calculate_gradient_spectrum")
        waveform = structure.waveform(tr)

        if window_width != _UPSTREAM_WINDOW_WIDTH:
            warnings.warn(
                f"calculate_gradient_spectrum(): window_width={window_width} is ignored under "
                "tr -- a repetition time is transformed whole, in one window",
                stacklevel=2,
            )

        spectrum = calculate_gradient_spectrum(
            waveform,
            max_frequency=max_frequency,
            # The window IS the TR, so this is one spectrum rather than a
            # spectrogram. A periodic waveform has energy only at multiples
            # of 1/T_TR, and it is the transform over exactly one period that
            # resolves them; a shorter window would smear neighbouring
            # harmonics together and a longer one does not exist to cut.
            window_width=structure.tr_duration,
            frequency_oversampling=frequency_oversampling,
            time_range=None,
            plot=plot,
            combine_mode=combine_mode,
            use_derivative=use_derivative,
            acoustic_resonances=acoustic_resonances,
        )
        if not resonance_lines:
            return spectrum if compat else _results.GradientSpectrum(*spectrum)

        if bands is None:
            bands = [
                (
                    resonance["frequency"] - 0.5 * resonance["bandwidth"],
                    resonance["frequency"] + 0.5 * resonance["bandwidth"],
                    0.0,
                )
                for resonance in acoustic_resonances
            ]
        resonances = _resonance_lines(structure, tr, max_frequency, bands)

        if plot and combine_mode != "none":
            _safety.overlay_resonance_lines(resonances, max_frequency=max_frequency)

        if not compat:
            return _results.GradientSpectrum(*spectrum, resonance_lines=resonances)
        # Upstream's four-tuple with a fifth element on the end. This is the
        # shape compat=False exists to retire -- it changes the length of the
        # caller's unpack -- and it survives only because it is what
        # resonance_lines already returned before the flag existed.
        return (*spectrum, resonances)


def _native_pns(structure: _Structure, hardware: object, mode: int, index: int) -> dict:
    """One TR's PNS response, straight out of the C safety core."""
    from .._ext._pulseg_wrapper import _calc_pns, _calc_pns_safe

    if mode != _safety.AMPLITUDE_MODES["actual"]:
        index = 0

    if _safety.is_safe_hardware(hardware):
        gx, gy, gz = _safety.safe_coefficients(hardware)
        return _calc_pns_safe(structure.collection, 0, index, gx, gy, gz)

    chronaxie_us, rheobase, alpha = _safety.irnich_coefficients(hardware)
    return _calc_pns(structure.collection, 0, index, chronaxie_us, rheobase, alpha)

def _plot_pns(waveform: _safety.TRSequence, components: np.ndarray, raster: float) -> None:
    """Upstream's two PNS figures, drawn over a TR waveform.

    The same pair :func:`pypulseq.Sequence.calc_pns.calc_pns` draws, from
    the same two calls -- the gradient trace off the PPoly upstream itself
    built, and ``safe_plot`` on the components -- plus the thresholds,
    which upstream draws in neither mode. ``raster`` is the safety core's,
    half the gradient raster, not the sequence's.
    """
    import matplotlib.pyplot as plt
    from pypulseq.utils.safe_pns_prediction import safe_plot

    plt.figure()
    for gradient in waveform.get_gradients():
        if gradient is not None:
            plt.plot(gradient.x[1:-1], gradient.c[1, :-1])
    plt.title("gradient wave form, in Hz/m")

    plt.figure()
    safe_plot(components * 100, raster)
    _safety.overlay_pns_thresholds()

def _resonance_lines(
    structure: _Structure, tr, max_frequency: float, bands: list
) -> _safety.MechResonances:
    """The C safety core's acoustic line spectrum for one TR."""
    from .._ext._pulseg_wrapper import _calc_mech_resonances

    _, index = structure.resolve(tr)
    spectra = _calc_mech_resonances(
        structure.collection,
        0,
        index,
        # A resolution fine enough that the harmonic grid the core
        # tabulates reaches max_frequency; the lines themselves land at
        # k / T_TR regardless of what is asked for here.
        target_resolution_hz=1.0 / structure.tr_duration,
        max_freq_hz=float(max_frequency),
        forbidden_bands=[tuple(float(value) for value in band[:3]) for band in bands],
    )
    return _safety.MechResonances.from_spectra(spectra, structure.tr_duration, bands)

