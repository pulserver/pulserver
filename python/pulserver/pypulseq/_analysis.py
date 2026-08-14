"""k-space, waveforms and event times.

The trajectory is the C core's; the waveform views are upstream PyPulseq's own,
run over a window of blocks.
"""

from __future__ import annotations

import numpy as np
import pypulseq as pp

from . import _results
from .._labels import COUNTER_LABELS, FRAME_COUNTERS, MRD_FLAGS, canonical_label
from ._common import _per_axis, _rf_use
from ._pulseqpp import to_upstream
from ._results import AdcTimes, RfTimes, Waveforms, WaveformsAndTimes

#: The boundary flags, as the counter each pair belongs to.
BOUNDARY_FLAGS = {
    name: (f"FIRST{name}", f"LAST{name}")
    for name in (*FRAME_COUNTERS, "LIN", "PAR", "SEG")
    if f"LAST{name}" in MRD_FLAGS
}

#: Every flag :func:`_boundary_flags` can produce.
BOUNDARY_NAMES = {flag for pair in BOUNDARY_FLAGS.values() for flag in pair} | {"LASTSCAN"}

#: The labels detection writes, and so the only ones it can be told to skip.
DETECTED_LABELS = frozenset({"NOISE", "SLC", "REV", "LIN", "PAR", "REP"})


def _boundary_flags(counters: dict, wanted: set[str], n_adc: int) -> dict:
    """Derive the boundary flags a set of per-ADC counters implies.

    Parameters
    ----------
    counters : dict
        Counter name to one value per ADC, in acquisition order.
    wanted : set of str
        Canonical flag names to derive. Anything else is left out.
    n_adc : int
        Number of ADCs, so ``LASTSCAN`` can be placed without a counter.

    Returns
    -------
    dict
        Flag name to a 0/1 ``numpy`` array, one value per ADC.

    Notes
    -----
    A frame -- one image -- is a setting of every frame counter at once, and it
    is complete when every encoding position inside it has arrived. So a frame
    counter's boundary is read within the other frame counters, and an encoding
    counter's within all of them; inside that group the first and last
    occurrence of each value are the boundary.

    Deliberately not read off the loop nesting. A scan looping lines outer and
    slices inner visits every ``(LIN, SLC)`` pair exactly once, so a nesting-
    derived ``LASTSLC`` would fire on every acquisition rather than at the four
    places a four-slice scan finishes a slice.
    """
    frames = [name for name in FRAME_COUNTERS if name in counters]
    flags = {}

    for name, (first_flag, last_flag) in BOUNDARY_FLAGS.items():
        if name not in counters or not ({first_flag, last_flag} & wanted):
            continue
        enclosing = [other for other in frames if other != name]
        keys = list(zip(*[counters[other] for other in enclosing], counters[name], strict=True))

        first_at: dict = {}
        last_at: dict = {}
        for index, key in enumerate(keys):
            first_at.setdefault(key, index)
            last_at[key] = index

        for flag, positions in ((first_flag, first_at), (last_flag, last_at)):
            if flag in wanted:
                values = np.zeros(len(keys), dtype=int)
                values[list(positions.values())] = 1
                flags[flag] = values

    if "LASTSCAN" in wanted and n_adc:
        values = np.zeros(n_adc, dtype=int)
        values[-1] = 1
        flags["LASTSCAN"] = values

    return flags


def _label_names(value: bool | list[str] | None, argument: str, allowed: set[str]) -> set[str]:
    """Resolve a ``True``/``False``/list-of-names argument into a set of names."""
    if value is True:
        return set(allowed)
    if not value:
        return set()
    names = {canonical_label(str(name)) for name in value}
    unknown = names - allowed
    if unknown:
        raise ValueError(
            f"auto_label(): {argument} does not name {sorted(unknown)}; it takes "
            f"{sorted(allowed)}"
        )
    return names


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
        boundary_flags: bool | list[str] = True,
        overwrite: bool | list[str] = False,
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

        **The prescription is read when the file states one.** ``FOV`` and
        ``Matrix``, together, say what one step of ``LIN`` and ``PAR`` is and
        how many of them the encoded axis holds, so the counters come back as
        positions on that matrix. Without them the step has to be inferred
        from the sampled positions and index zero from the lowest one sampled,
        which is right only for a scan that reaches the grid on its own: an
        accelerated scan with no autocalibration block has no adjacent pair to
        read the step off, and a partial-Fourier scan never visits the low
        edge. Set both definitions *before* calling this if the sequence
        states them; a prescription the readouts do not land on is ignored, so
        a non-Cartesian scan is unaffected either way.

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

            A counter the sequence already sets is protected without being
            named here, so this is for suppressing one the sequence does not
            set and does not want: deriving ``REP`` on a scan whose repeats
            mean nothing, say.
        boundary_flags : bool or list of str, optional
            Which ``FIRST``/``LAST`` flags to derive. ``True``, the default,
            derives every one whose counter is known; a list restricts it, and
            ``False`` derives none. Names may be given in either spelling, so
            ``["LASTSLC"]`` and ``["ACQ_LAST_IN_SLICE"]`` ask the same thing.

            They come from the counters and the order the scan acquires them
            in, so nothing about the trajectory is needed and they are derived
            for authored counters as readily as for detected ones.
        overwrite : bool or list of str, optional
            Labels to write over even though the sequence already sets them.
            ``False``, the default, writes none of them; a list names the
            exceptions, and ``True`` writes them all.

            This decides what is *written*, not what is derived: everything is
            derived and returned either way, so reading a counter back to
            compare it against the one a design authored costs nothing and
            changes nothing. Leaving it alone is what makes this safe to run on
            a sequence that labelled itself -- detection fills the gaps and
            leaves the rest as authored.

        Returns
        -------
        labels : dict
            Counter or flag name to an array with one value per ADC, in
            acquisition order. Only the counters that vary are present -- a
            single-slice scan has no ``SLC``. Everything derived is here,
            including labels that were not written because the sequence
            already carries them.
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

        wanted = _label_names(boundary_flags, "boundary_flags", BOUNDARY_NAMES)
        forced = _label_names(
            overwrite, "overwrite", BOUNDARY_NAMES | set(COUNTER_LABELS) | DETECTED_LABELS
        )

        # The definitions are kept on this side until a write needs them, and
        # detection reads `FOV` and `Matrix` to place the counters on the grid
        # the sequence was prescribed on.
        self._publish_definitions()

        first, last = self._window_for(time_range)

        # What the sequence says about itself, which nothing here may
        # contradict: a design that labelled its own axes and boundaries knows
        # things a trajectory cannot be read for.
        authored = self._authored_labels(time_range)
        protected = set(authored) - forced

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
            labels = {name: np.atleast_1d(values) for name, values in (use_labels or {}).items()}
            aux = dict(use_aux or {})
            counters = {**{n: v for n, v in authored.items() if n in COUNTER_LABELS}, **labels}
            labels.update(_boundary_flags(counters, wanted, self._num_adc(time_range)))
            if not skip_apply:
                # What was passed in is an instruction and goes down as given;
                # only the flags derived around it answer to `overwrite`.
                self._write_labels(
                    {
                        name: values
                        for name, values in labels.items()
                        if name in (use_labels or {}) or name not in protected
                    },
                    time_range,
                    aux,
                )
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
            # Detect, never write: what gets written is decided here, where
            # what the sequence already says is known. Protection cannot be a
            # `skip`, because a counter that is not derived takes the
            # definitions read off it down with it -- `kSpaceCenterLine` is
            # the line of the central readout, and there is no central readout
            # without `LIN`.
            False,
            dims,
            [canonical_label(str(name)) for name in (skip or ())],
            bool(mirror_fourier),
            sort_slices,
        )
        aux = result["aux"]
        labels = result["labels"]

        # A boundary is read off every counter the scan has, whoever wrote it,
        # so detection and authorship are one picture here.
        counters = {**{n: v for n, v in authored.items() if n in COUNTER_LABELS}, **labels}
        labels.update(_boundary_flags(counters, wanted, self._num_adc(time_range)))

        if not skip_apply:
            self._write_labels(
                {name: values for name, values in labels.items() if name not in protected},
                time_range,
                aux,
            )
        return labels, aux

    def _num_adc(self, time_range: list[float] | None) -> int:
        """How many blocks in the window hold an ADC."""
        return len(self._adc_blocks(time_range))

    def _adc_blocks(self, time_range: list[float] | None) -> list[int]:
        """The 1-based indices of the blocks in the window that hold an ADC."""
        first, last = self._window_for(time_range)
        events = self._native.block_events()
        return [
            index
            for index in range(first, (last or self._native.num_blocks()) + 1)
            if events[index - 1][4] != 0
        ]

    def _authored_labels(self, time_range: list[float] | None) -> dict:
        """One value per ADC for every label the sequence already sets.

        Empty when it sets none, which is the usual state of a ``.seq`` written
        elsewhere -- and the cheap check for it, so that case never pays for a
        walk over the blocks.
        """
        if not (self._native.num_label_set() or self._native.num_label_inc()):
            return {}
        return self.evaluate_labels(evolution="adc", time_range=time_range)

    def _write_labels(self, labels: dict, time_range: list[float] | None, aux: dict) -> None:
        """Write one value per ADC for each named label onto the blocks."""
        blocks = self._adc_blocks(time_range)
        ordered = [
            (name, [int(v) for v in np.atleast_1d(values)]) for name, values in labels.items()
        ]
        for name, values in ordered:
            if len(values) != len(blocks):
                raise ValueError(
                    f"auto_label(): use_labels['{name}'] has {len(values)} values "
                    f"for {len(blocks)} ADCs in range"
                )
        self._native.apply_labels(blocks, ordered, aux)
        for key, value in aux.items():
            self.set_definition(key, value.tolist() if hasattr(value, "tolist") else value)
        self._touch()

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

