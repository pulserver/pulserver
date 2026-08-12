"""k-space, waveforms and event times.

The trajectory is the C core's; the waveform views are upstream PyPulseq's own,
run over a window of blocks.
"""

from __future__ import annotations

import numpy as np
import pypulseq as pp

from . import _results
from ._common import _per_axis, _rf_use
from ._pulseqpp import to_upstream
from ._results import AdcTimes, RfTimes, Waveforms, WaveformsAndTimes


class AnalysisMixin:
    """k-space, waveform and event-time views. Mixed into :class:`Sequence`."""

    def calculate_kspace(
        self,
        trajectory_delay: float | list[float] | np.ndarray = 0.0,
        gradient_offset: float | list[float] | np.ndarray = 0.0,
        *,
        time_range: list[float] | None = None,
        frame: str = "physical",
        sample_window_average: bool = False,
        dense: bool = True,
        compat: bool = True,
    ):
        """Where every ADC sample sits in k-space.

        Returns upstream's five-tuple, ``(k_traj_adc, k_traj, t_excitation,
        t_refocusing, t_adc)``, with ``k_traj_adc`` and ``k_traj`` shaped
        ``(3, n)`` in 1/m and the times in seconds.

        The arithmetic is ``csrc/src/pulseq/pulseq_ktraj.c``, the same code the
        interpreter links. Cost follows the number of distinct gradients, not
        the length of the scan.

        Parameters
        ----------
        trajectory_delay : float or array-like, optional
            Gradient timing compensation in seconds, one value or one per
            axis. Shifts the gradient time base only, never the ADC or RF
            times -- those are synchronised with each other by construction.
        gradient_offset : float or array-like, optional
            A constant background gradient in Hz/m, one value or one per axis.
        time_range : list of float, optional
            ``[start, stop]`` in seconds, to analyse part of the sequence.
            The whole of it by default. The blocks this touches are the ones
            decoded, and the times come back on the sequence's own clock.
        frame : {'physical', 'logical'}, optional
            Whether to resolve rotation extensions into the answer.
            ``'physical'``, the default, is what upstream PyPulseq and Pulseq's
            MATLAB ``calculateKspacePP`` both return and what a reconstruction
            needs. ``'logical'`` leaves the rotation out, which is the frame
            :class:`~pulserver.pypulseq.TransformFOV` works in.
        sample_window_average : bool, optional
            Give each sample the k averaged over its dwell rather than k at
            the window's midpoint. An ADC sample integrates for a whole dwell,
            so the average is the coordinate it physically belongs to; the two
            differ by ``dwell**2 / 24 * dg/dt`` and so agree exactly wherever
            the gradient is flat. Off by default, because PyPulseq, MRpro and
            mri-nufft all sample at the midpoint and matching them is what
            makes the answer comparable. Turn it on for a gridder.
        dense : bool, optional
            Also build ``k_traj``. It is the one output whose size grows with
            the duration of the scan rather than with the acquisition, and the
            only one computed by upstream rather than here -- see the note
            below. Pass ``False`` to skip it and get an empty array back.

        Returns
        -------
        tuple
            ``(k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc)``.

        Notes
        -----
        **``k_traj`` comes from upstream PyPulseq, the other four from the C
        core.** The dense trajectory is a picture of the sequence -- what a
        plot draws -- and being able to hand it to code written against
        upstream matters more than computing it quickly. The C core's own
        answer is on the gradient *breakpoint* grid, which describes the same
        curve in five to ten times fewer points; that is the better
        representation and the wrong one to return from a function whose
        contract is upstream's. It is still available through
        :meth:`_kspace`, together with its time base ``t_ktraj``, which this
        tuple has nowhere to put.

        Nothing a reconstruction needs is on that path: ``k_traj_adc`` is
        where the samples actually are, and it is the C core's, agreeing with
        upstream to 2e-13 on a GRE and 2e-12 on an EPI.

        Sequences upstream cannot read -- anything carrying rotation or RF-shim
        extensions -- have no ``k_traj``, and ask for one raises rather than
        quietly returning the breakpoint grid in its place.

        See Also
        --------
        auto_label : the encoding counters derived from this trajectory.
        """
        blocks = self._window_for(time_range)
        result = self._kspace(
            trajectory_delay=trajectory_delay,
            gradient_offset=gradient_offset,
            blocks=blocks,
            frame=frame,
            sample_window_average=sample_window_average,
            # The breakpoint-grid trajectory has nowhere to go in upstream's
            # tuple, so it is only asked for when there is somewhere to put it.
            dense=not compat,
        )

        k_traj = np.zeros((3, 0))
        if dense:
            # to_upstream does not resolve rotation or RF-shim extensions, so
            # for a sequence carrying either it would hand back a logical-frame
            # k_traj beside a physical-frame k_traj_adc -- two frames in one
            # tuple, and nothing to say which is which. Refuse instead.
            if self._native.num_rotations() > 0 or self._native.num_rf_shims() > 0:
                raise NotImplementedError(
                    "calculate_kspace: k_traj comes from upstream PyPulseq, which cannot "
                    "read the rotation or RF-shim extensions this sequence carries -- it "
                    "would come back in the logical frame beside a physical-frame "
                    "k_traj_adc. Use dense=False for the ADC samples, or _kspace() for "
                    "the breakpoint-grid trajectory, which does resolve them."
                )
            first, last = blocks
            upstream = to_upstream(self, first=first, last=(None if last == 0 else last))
            k_traj = upstream.calculate_kspace(
                trajectory_delay=trajectory_delay,
                gradient_offset=gradient_offset,
            )[1]

        if compat:
            return (
                result["k_adc"],
                k_traj,
                result["t_excitation"],
                result["t_refocusing"],
                result["t_adc"],
            )
        return _results.KSpace(
            k_traj_adc=result["k_adc"],
            k_traj=k_traj,
            t_excitation=result["t_excitation"],
            t_refocusing=result["t_refocusing"],
            t_adc=result["t_adc"],
            k_traj_breakpoints=result["k_traj"],
            t_breakpoints=result["t_ktraj"],
            k_center=result["k_center"],
            readout_center_sample=result["readout_center_sample"],
        )

    def _kspace(
        self,
        *,
        trajectory_delay=0.0,
        gradient_offset=0.0,
        blocks=None,
        frame="physical",
        sample_window_average=False,
        dense=True,
    ) -> dict:
        """Everything the C core reports, not just upstream's five entries.

        Kept separate because :meth:`calculate_kspace` has to return exactly
        upstream's tuple, and the derived echo positions, the k-space centre
        and the repeat-key statistics have nowhere in it to go.

        ``blocks`` is the 1-based inclusive pair the public methods resolve
        their ``time_range`` into, not a window of its own.
        """
        if frame not in ("physical", "logical"):
            raise ValueError(f"frame must be 'physical' or 'logical', not {frame!r}")

        first, last = (1, 0) if blocks is None else blocks
        return self._native.calculate_kspace(
            _per_axis(trajectory_delay, "trajectory_delay"),
            _per_axis(gradient_offset, "gradient_offset"),
            first,
            last,
            frame == "physical",
            bool(sample_window_average),
            bool(dense),
        )

    def auto_label(
        self,
        *,
        # -- MATLAB Pulseq's autoLabel parameters, under Python names --------
        time_range: list[float] | None = None,
        use_labels: dict | None = None,
        use_aux: dict | None = None,
        skip_apply: bool = False,
        mirror_fourier: bool = False,
        reflect: list[int] | None = None,
        reorder: list[int] | None = None,
        sort_slices: str = "ascending",
        no_plots: bool = True,
        # -- Pulserver's own, on top -----------------------------------------
        trajectory_delay: float | list[float] | np.ndarray = 0.0,
        repeat_dims: list[str | tuple[str, int]] | None = None,
        skip: list[str] | None = None,
    ) -> tuple[dict, dict]:
        """Recover the encoding counters from the sequence's own trajectory.

        A ``.seq`` written elsewhere carries no ``LABELSET`` extensions, so
        nothing downstream knows which line, partition, slice or repetition an
        acquisition belongs to. It is all still there, written into where the
        readouts sit in k-space, and this reads it back out.

        Same labels as Pulseq's MATLAB ``autoLabel``, by a cheaper route: that
        one walks every ADC sample three times over, and here the echo search
        is memoized per distinct readout and the rest reduces to one point per
        readout, so nothing scales with the number of samples.

        Every ``autoLabel`` parameter is accepted, under the Python spelling
        of its name and in its own order; Pulserver's additions come after
        them. Two defaults differ, and both are called out below --
        ``sort_slices`` and ``no_plots``.

        Parameters
        ----------
        time_range : list of float, optional
            ``[start, stop]`` in seconds. MATLAB's ``blockRange`` in the same
            position, in the unit PyPulseq windows everything else by.
        use_labels : dict, optional
            Skip detection and apply these counters instead -- the labels
            half of a previous call's return value, or a set computed some
            other way. Keys are counter names, values one entry per ADC in
            acquisition order.

            For applying one detection to several variants of a sequence, and
            for correcting a counter by hand without recomputing the rest.
            Cannot be combined with ``reflect``, ``reorder`` or
            ``mirror_fourier``, which only affect detection -- MATLAB refuses
            that combination too, and it would silently do nothing.
        use_aux : dict, optional
            The definitions to write, in the same spirit: the ``aux`` half of
            a previous return. Usable on its own or alongside ``use_labels``.
        skip_apply : bool, optional
            Return the counters without writing them onto the sequence. By
            default they are written, as ``SET`` label extensions on each ADC
            block where the value changes, and the derived definitions
            (``kSpaceCenterLine``, ``SliceThickness`` and the rest) go into
            ``[DEFINITIONS]``.
        mirror_fourier : bool, optional
            Negate every Fourier-encoding direction at once -- readout, phase
            and partition -- for a reconstruction that inverse-transforms
            where this assumes a forward transform.

            Not the same as ``reflect=[0, 1, 2]``, in the one way that
            matters: the slice positions and slice-select gradients are left
            alone, so slice ordering is unaffected. Applied before
            ``reflect``, and freely combined with it.
        reflect : list of int, optional
            Axes (0, 1, 2) whose k, slice positions and gradients to negate
            before deriving anything. Applied before ``reorder``.
        reorder : list of int, optional
            A permutation of the axes, as source indices: ``[1, 0, 2]`` swaps
            x and y.
        sort_slices : {"ascending", "descending", "acquisition"}, optional
            How ``SLC`` is assigned. ``SlicePositions[SLC]`` is the position
            of slice ``SLC`` under all three, so a reconstruction reading the
            pair together is right either way; what changes is which index a
            slice is given.

            **The default differs from MATLAB's**, which is
            ``"acquisition"``. A geometric index is what makes the slice
            table usable as a stack: an interleaved acquisition (0, 2, 4, 1,
            3) hands the reconstruction a shuffled volume under arrival order
            and an ordered one under ``"ascending"``. Pass ``"acquisition"``
            for MATLAB's numbering exactly. ``"descending"`` is what
            ``autoLabel``'s own notes recommend for a Siemens interpreter.
        no_plots : bool, optional
            **The default differs from MATLAB's**, which is ``False``.
            ``autoLabel`` draws diagnostic figures; nothing here does, so
            there is nothing to suppress and ``True`` is the only truthful
            value. Passing ``False`` raises rather than quietly drawing
            nothing -- it is a request for output that will not appear.
        trajectory_delay : float or array-like, optional
            As for :meth:`calculate_kspace`.
        repeat_dims : sequence of str, optional
            The dimensions the repetition counter is standing in for, named
            by you, **outermost loop first** -- ``["REP", "ECO"]``.

            Where a readout sits in k says which line, partition and slice it
            is. It cannot say which echo of a train, which frame of a time
            series or which saturation state it is, because all of those
            revisit the same k-space position -- so by default they are
            counted together as ``REP``.

            Only the names are needed. How large each dimension is, is
            written in the acquisition order and read back from it: a
            dimension nested inside the k-space loop brings a position back
            after a short gap, one outside it only after a whole pass. Pass
            ``("ECO", 2)`` in place of a name to pin a size, and it is
            checked against what was read rather than believed.

            Repeats that are not a rectangle -- some positions revisited and
            others not, as with an EPI's navigators -- have no nest to read
            and raise. A single name never does: it takes the whole count,
            which is ``REP`` under a name that means something.
        skip : list of str, optional
            Counters to leave alone -- derived neither into the answer nor
            onto the sequence.

            For a sequence that labelled some of its own axes as it was
            built and wants the geometric ones filled in around them.
            Labels this does not derive at all (``ECO``, ``SET``, ``AVG``,
            anything custom) already survive an ``auto_label`` pass
            untouched and need no mention. ``REP`` is the one that does:
            it is derived by default, so a design loop that separated its
            own contrasts or frames should pass ``skip=["REP"]`` or its own
            labelling is overwritten by a bare repeat count.

        Returns
        -------
        labels : dict
            Counter name to an array with one value per ADC, in acquisition
            order. Only the counters that vary are present -- a single-slice
            scan has no ``SLC``.
        aux : dict
            The derived definitions.

        Raises
        ------
        RuntimeError
            If the readouts do not share a direction. These are Cartesian
            encoding counters, and a non-Cartesian trajectory has no honest
            value for them -- which is what MATLAB's ``autoLabel`` also says.
            Also if the repeats do not form a rectangle that ``repeat_dims``
            can name, or if a size you pinned contradicts the acquisition
            order.

        Notes
        -----
        ``SLC`` is a geometric index: slices are ranked by the position their
        excitation's frequency offset puts them at, so ``SlicePositions[SLC]``
        is where slice ``SLC`` sits whatever order the scan visited them in.
        Those offsets are read as authored, and :class:`TransformFOV` scaling
        rewrites the slice-select gradient without touching them -- so label
        first and transform second.
        """
        if not no_plots:
            raise ValueError(
                "auto_label(): no_plots=False asks for the diagnostic figures MATLAB's "
                "autoLabel draws, and nothing here draws any. Leave it at True and plot "
                "from the returned labels if you need a picture."
            )
        if sort_slices not in ("ascending", "descending", "acquisition"):
            raise ValueError(
                f"auto_label(): sort_slices must be 'ascending', 'descending' or "
                f"'acquisition', got {sort_slices!r}"
            )

        first, last = self._window_for(time_range)

        # Detection-only options against a caller who has skipped detection.
        # MATLAB raises on the same combination, and for the same reason: it
        # would look like it did something.
        if (use_labels is not None or use_aux is not None) and (
            reflect or reorder or mirror_fourier
        ):
            raise ValueError(
                "auto_label(): reflect, reorder and mirror_fourier only affect detection, "
                "so they cannot be combined with use_labels or use_aux."
            )

        if use_labels is not None or use_aux is not None:
            labels = dict(use_labels or {})
            aux = dict(use_aux or {})
            if not skip_apply:
                blocks = [
                    index
                    for index in range(first, (last or self._native.num_blocks()) + 1)
                    if self._native.block_events()[index - 1][4] != 0
                ]
                ordered = [
                    (name, [int(v) for v in np.atleast_1d(values)])
                    for name, values in labels.items()
                ]
                for name, values in ordered:
                    if len(values) != len(blocks):
                        raise ValueError(
                            f"auto_label(): use_labels['{name}'] has {len(values)} values "
                            f"for {len(blocks)} ADCs in range"
                        )
                self._native.apply_labels(blocks, ordered, aux)
                for key, value in aux.items():
                    self.set_definition(
                        key, value.tolist() if hasattr(value, "tolist") else value
                    )
                self._touch()
            return labels, aux

        reflect_mask = [False, False, False]
        for axis in reflect or ():
            if axis not in (0, 1, 2):
                raise ValueError(f"reflect axes must be 0, 1 or 2, not {axis!r}")
            reflect_mask[axis] = True

        order = [0, 1, 2]
        if reorder is not None:
            if sorted(reorder) != list(range(len(reorder))) or len(reorder) not in (2, 3):
                raise ValueError(f"reorder must permute the first 2 or 3 axes, got {reorder!r}")
            order[: len(reorder)] = list(reorder)

        dims = []
        for entry in repeat_dims or ():
            # A bare name is the ordinary case; a (name, size) pair pins one
            # down. Strings are iterable, so they have to be caught first --
            # unpacking "AB" would otherwise succeed and mean nothing.
            if isinstance(entry, str):
                dims.append((entry, 0))
                continue
            try:
                name, size = entry
            except (TypeError, ValueError):
                raise ValueError(
                    f"repeat_dims entries are names, or (name, size) pairs to pin a size, "
                    f"got {entry!r}"
                ) from None
            dims.append((str(name), int(size)))

        result = self._native.auto_label(
            first,
            last,
            reflect_mask,
            order,
            _per_axis(trajectory_delay, "trajectory_delay"),
            not skip_apply,
            dims,
            [str(name) for name in (skip or ())],
            bool(mirror_fourier),
            sort_slices,
        )
        aux = result["aux"]
        if not skip_apply:
            # The C++ side wrote these onto the native sequence, which is right
            # for a C++ caller and invisible here: this class keeps its own
            # definitions and pushes them across when the sequence is written,
            # so anything only the native side knows would be overwritten on
            # the way out. Mirroring them is what makes them survive.
            for key, value in aux.items():
                self.set_definition(key, value.tolist() if hasattr(value, "tolist") else value)
            self._touch()
        return result["labels"], aux

    def plot_kspace(
        self,
        *,
        time_range: list[float] | None = None,
        plane: str | None = None,
        show_trajectory: bool = True,
        plot_now: bool = True,
    ):
        """Draw the k-space the ADC samples visit.

        Parameters
        ----------
        time_range : list of float, optional
            Restrict to the blocks in this window, in seconds.
        plane : {"xy", "xz", "yz"}, optional
            Project onto two axes. By default the scan chooses: a trajectory
            confined to one plane is drawn in it, and anything else in 3D.
        show_trajectory : bool, default True
            Draw the continuous path between samples as well as the samples.
        plot_now : bool, default True
            Show the figure before returning.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt

        result = self.calculate_kspace(time_range=time_range, dense=show_trajectory, compat=False)
        adc = np.asarray(result.k_traj_adc, dtype=float)
        if adc.size == 0:
            raise ValueError("plot_kspace(): the window holds no ADC samples")

        axes_used = [a for a in range(3) if np.ptp(adc[a]) > 1e-9 * max(np.ptp(adc), 1e-12)]
        if plane is None:
            plane = "".join("xyz"[a] for a in axes_used[:2]) if len(axes_used) <= 2 else None
        labels = {"x": 0, "y": 1, "z": 2}

        figure = plt.figure(figsize=(5.5, 5.0))
        if plane is None:
            axis = figure.add_subplot(projection="3d")
            if show_trajectory:
                path = np.asarray(result.k_traj, dtype=float)
                axis.plot(path[0], path[1], path[2], lw=0.4, color="0.7")
            axis.scatter(adc[0], adc[1], adc[2], s=1.5)
            axis.set_xlabel("$k_x$ [1/m]")
            axis.set_ylabel("$k_y$ [1/m]")
            axis.set_zlabel("$k_z$ [1/m]")
        else:
            if len(plane) != 2 or any(c not in labels for c in plane):
                raise ValueError(f"plot_kspace(): plane must be two of x, y, z, got {plane!r}")
            first, second = labels[plane[0]], labels[plane[1]]
            axis = figure.add_subplot()
            if show_trajectory:
                path = np.asarray(result.k_traj, dtype=float)
                axis.plot(path[first], path[second], lw=0.4, color="0.7")
            axis.scatter(adc[first], adc[second], s=1.5)
            axis.set_xlabel(f"$k_{plane[0]}$ [1/m]")
            axis.set_ylabel(f"$k_{plane[1]}$ [1/m]")
            axis.set_aspect("equal", adjustable="datalim")

        figure.tight_layout()
        if plot_now:
            plt.show()
        return figure

    def calculate_kspacePP(
        self,
        # Present only so the signature matches upstream's; nothing reads them.
        trajectory_delay: float | list[float] | np.ndarray = 0,  # noqa: ARG002
        gradient_offset: float | list[float] | np.ndarray = 0,  # noqa: ARG002
    ):
        """Deprecated upstream; raises instead of forwarding."""
        raise DeprecationWarning(
            "Sequence.calculate_kspacePP has been deprecated, use calculate_kspace instead"
        )

    def waveforms(
        self,
        append_RF: bool = False,
        time_range: list[float] | None = None,
        *,
        compat: bool = True,
    ):
        """The gradient waveforms, decompressed onto one time axis per channel.

        Parameters
        ----------
        append_RF : bool, optional
            Append the complex RF waveform as a fourth channel.
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's return -- a list of ``(2, n)`` arrays -- by default.
            ``False`` returns a :class:`~._results.Waveforms` instead, whose
            channels are named rather than positional.

        Notes
        -----
        An arbitrary gradient stored on the centres raster has its raster-edge
        samples reconstructed here, which upstream leaves as a ``TODO``. A
        trapezoid that was converted to a shape therefore comes back with its
        corners where the sequence put them, rather than rounded over a raster
        interval -- exactly, on a fixture, against upstream's 1.25% at the
        corner, and in four samples rather than eighty-two. See
        :class:`~._pulseqpp.RestoringSequence`.
        """
        first, last = self._window_for(time_range)
        channels = self._upstream_window(first, last).waveforms(append_RF=append_RF)

        if compat:
            return channels
        return Waveforms(
            gx=channels[0],
            gy=channels[1],
            gz=channels[2],
            rf=channels[3] if append_RF and len(channels) > 3 else None,
        )

    def waveforms_and_times(
        self,
        append_RF: bool = False,
        time_range: list[float] | None = None,
        *,
        compat: bool = True,
    ):
        """The waveforms, plus when the RF pulses and ADC samples happen.

        Parameters
        ----------
        append_RF : bool, optional
            Append the complex RF waveform as a fourth gradient channel.
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's five-tuple ``(wave_data, tfp_excitation, tfp_refocusing,
            t_adc, fp_adc)`` by default. ``False`` returns a
            :class:`~._results.WaveformsAndTimes`.

        Notes
        -----
        **Upstream's tuple cannot say everything the sequence knows**, and
        ``compat=False`` is where the rest of it comes out:

        - *Every* RF use, not two. Upstream sorts RF into excitation and
          refocusing and silently drops inversion, saturation, preparation and
          "other" -- an inversion pulse does not appear in its answer at all.
        - ``pm_adc``, the per-sample ADC phase modulation, which MATLAB returns
          as a sixth output and PyPulseq does not return at all.
        - The echo centres, which neither returns. MATLAB reconstructs
          ``2*t_refocusing - t_excitation`` in ``calcMomentsBtensor`` and
          carries its own ``TODO: fixme for double-refocused sequences``; the
          value here is the ADC sample nearest k-space zero, found by the C
          core walking the real trajectory. It is computed on first read, not
          with the rest, because it needs that trajectory.
        """
        first, last = self._window_for(time_range)
        window = self._upstream_window(first, last)

        channels = window.waveforms(append_RF=append_RF)
        # The window carries the time in front of it as a lead-in block
        # numbered 0, so walking it from zero already gives absolute times.
        rf = self._rf_times_of(window, 0.0)
        adc = self._adc_times_of(window, 0.0, block_span=(first, last))

        if compat:
            return (
                channels,
                rf.of("excitation", "undefined").tfp,
                rf.of("refocusing").tfp,
                adc.t,
                adc.fp,
            )
        return WaveformsAndTimes(
            waveforms=Waveforms(
                gx=channels[0],
                gy=channels[1],
                gz=channels[2],
                rf=channels[3] if append_RF and len(channels) > 3 else None,
            ),
            rf=rf,
            adc=adc,
        )

    def rf_times(self, time_range: list[float] | None = None, *, compat: bool = True):
        """When each RF pulse reaches its centre, and with what phase.

        Parameters
        ----------
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's ``(t_excitation, fp_excitation, t_refocusing,
            fp_refocusing)`` by default, which describes two of Pulseq's seven
            RF uses and drops the rest. ``False`` returns a
            :class:`~._results.RfTimes` covering all of them.
        """
        first, last = self._window_for(time_range)
        window = self._upstream_window(first, last)
        pulses = self._rf_times_of(window, 0.0)

        if not compat:
            return pulses

        # Upstream counts an untagged pulse as an excitation.
        excitation = pulses.of("excitation", "undefined")
        refocusing = pulses.of("refocusing")
        return (
            list(excitation.t),
            np.vstack((excitation.freq_offset, excitation.phase_offset)),
            list(refocusing.t),
            np.vstack((refocusing.freq_offset, refocusing.phase_offset)),
        )

    def adc_times(self, time_range: list[float] | None = None, *, compat: bool = True):
        """When every ADC sample is taken.

        Parameters
        ----------
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's ``(t_adc, fp_adc)`` by default, where ``fp_adc`` is one
            row per ADC *event* carrying its raw frequency and phase offsets.
            ``False`` returns an :class:`~._results.AdcTimes`, which adds the
            per-*sample* phase -- ppm terms, phase modulation and accumulated
            ``2*pi*f*t`` folded in, the number a demodulator wants.
        """
        first, last = self._window_for(time_range)
        window = self._upstream_window(first, last)
        samples = self._adc_times_of(window, 0.0, block_span=(first, last))
        return samples if not compat else (samples.t, samples.fp)

    def get_gradients(
        self,
        trajectory_delay: float | list[float] | np.ndarray = 0,
        gradient_offset: float | list[float] | np.ndarray = 0,
        time_range: list[float] | None = None,
    ) -> list:
        """The gradients as :class:`scipy.interpolate.PPoly` piecewise polynomials.

        Upstream's, evaluated on this class's waveforms -- so the raster-edge
        reconstruction :meth:`waveforms` performs is in them, and in everything
        built on them.
        """
        first, last = self._window_for(time_range)
        return self._upstream_window(first, last).get_gradients(
            trajectory_delay=trajectory_delay,
            gradient_offset=gradient_offset,
        )

    # -- the walk behind rf_times / adc_times ----------------------------


    def _rf_times_of(self, window: pp.Sequence, elapsed: float) -> RfTimes:
        """Walk a window's RF pulses into one flat table, use tags kept.

        The centre is :func:`pypulseq.calc_rf_center`'s, and the phase carries
        the ``2*pi*f*t_centre`` term to it -- upstream's convention and
        MATLAB's, so ``compat=True`` reproduces upstream exactly.
        """
        from pypulseq.calc_rf_center import calc_rf_center

        gamma_b0 = self.system.gamma * self.system.B0
        times: list[float] = []
        frequencies: list[float] = []
        phases: list[float] = []
        uses: list[str] = []
        blocks: list[int] = []

        for number in window.block_events:
            block = window.get_block(number)
            rf = getattr(block, "rf", None)
            if rf is not None:
                centre = calc_rf_center(rf)[0]
                frequency = rf.freq_offset + rf.freq_ppm * 1e-6 * gamma_b0
                phase = rf.phase_offset + rf.phase_ppm * 1e-6 * gamma_b0

                times.append(elapsed + rf.delay + centre)
                frequencies.append(frequency)
                phases.append(phase + 2 * np.pi * frequency * centre)
                uses.append(_rf_use(rf))
                blocks.append(number)
            elapsed += window.block_durations[number]

        return RfTimes(
            t=np.asarray(times, dtype=float),
            freq_offset=np.asarray(frequencies, dtype=float),
            phase_offset=np.asarray(phases, dtype=float),
            use=tuple(uses),
            block=np.asarray(blocks, dtype=int),
        )

    def _adc_times_of(
        self, window: pp.Sequence, elapsed: float, *, block_span: tuple[int, int]
    ) -> AdcTimes:
        """Walk a window's ADC events into sample times and per-sample phase."""
        gamma_b0 = self.system.gamma * self.system.B0
        sample_times: list[np.ndarray] = []
        sample_phases: list[np.ndarray] = []
        modulations: list[np.ndarray] = []
        frequencies: list[float] = []
        phases: list[float] = []
        blocks: list[int] = []
        counts: list[int] = []

        for number in window.block_events:
            block = window.get_block(number)
            adc = getattr(block, "adc", None)
            if adc is not None:
                count = int(adc.num_samples)
                # Samples sit half a dwell into their window -- Siemens' and
                # Pulseq's shared convention, not a midpoint approximation.
                within = (np.arange(count) + 0.5) * adc.dwell
                frequency = adc.freq_offset + adc.freq_ppm * 1e-6 * gamma_b0
                phase = adc.phase_offset + adc.phase_ppm * 1e-6 * gamma_b0

                modulation = getattr(adc, "phase_modulation", None)
                if modulation is None or len(modulation) == 0:
                    modulation = np.zeros(count)
                modulation = np.asarray(modulation, dtype=float).ravel()

                sample_times.append(elapsed + adc.delay + within)
                sample_phases.append(phase + modulation + 2 * np.pi * frequency * within)
                modulations.append(modulation)
                # Upstream's fp_adc is the raw event offsets, no ppm folded in.
                frequencies.append(adc.freq_offset)
                phases.append(adc.phase_offset)
                blocks.append(number)
                counts.append(count)
            elapsed += window.block_durations[number]

        def _stack(pieces, width=None):
            if pieces:
                return np.concatenate(pieces)
            return np.zeros(0) if width is None else np.zeros((0, width))

        return AdcTimes(
            t=_stack(sample_times),
            freq_offset=np.asarray(frequencies, dtype=float),
            phase_offset=np.asarray(phases, dtype=float),
            phase_modulation=_stack(modulations),
            sample_phase=_stack(sample_phases),
            block=np.asarray(blocks, dtype=int),
            num_samples=np.asarray(counts, dtype=int),
            _echoes=lambda: self._echo_centers(block_span),
        )

    def _echo_centers(self, blocks: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """``(sample index, time)`` of the k-space centre of each readout.

        The C core already reports which sample of each readout is nearest
        k-space zero -- ``readout_center_sample``, which it derives while
        integrating the trajectory. Reading it back is cheaper and more honest
        than re-deriving it here, and it is the same number the recon and the
        scanner use.
        """
        first, last = blocks
        result = self._kspace(blocks=(first, last), dense=False)

        centers = np.asarray(result["readout_center_sample"], dtype=int)
        counts = np.asarray(result["readout_samples"], dtype=int)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(int)
        t_adc = np.asarray(result["t_adc"], dtype=float)

        absolute = starts + centers
        inside = (absolute >= 0) & (absolute < t_adc.size)
        times = np.full(absolute.shape, np.nan)
        times[inside] = t_adc[absolute[inside]]
        return centers, times

