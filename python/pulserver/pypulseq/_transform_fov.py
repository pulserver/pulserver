"""Moving, turning and resizing the imaging volume of a built sequence.

A port of Pulseq's MATLAB ``mr.TransformFOV``: an object that carries a
scaling, a translation and a rotation, and applies them to a sequence that
has already been designed. The three are applied in that order -- scale,
translate, rotate -- because scaling changes the gradient amplitudes the
translation's phase math reads, while a rotation, expressed as an extension,
changes nothing either of them sees.

**A translation is a phase, and the phase is computed in C++.** MATLAB walks
the sequence a block at a time, building a piecewise polynomial per gradient
and carrying a running ``prior_phase_cycle`` from block to block by hand.
The engine underneath this class already does the equivalent for a whole
sequence in one pass: :func:`pulseq::block_k_origins` tracks where absolute k
stands at the start of every block -- reset by an excitation, negated by a
refocusing pulse -- which *is* that running phase, written as a k-space
position rather than a scalar. So the shift needs nothing from Python but the
block range to apply it over.

**A rotation is an annotation, not a redesign.** Only
``use_rotation_extension=True`` is supported: the rotation is attached to each
block as a ``ROTATIONS`` extension and resolved by whatever plays the
sequence, rather than being baked into new gradient waveforms. That keeps one
waveform per distinct trajectory no matter how many orientations use it,
which is the difference between a large non-Cartesian scan being affordable
and not. It is also the case this class exists to serve: a module that points
somewhere other than the main prescription -- a fat saturation slab, say --
followed by ``NOPOS``/``NOROT`` on the blocks that must not be transformed
again downstream.

**A scale is one multiplication.** Gradients are stored as a normalised shape
beside a scalar amplitude, so scaling multiplies the amplitude and leaves the
waveform -- and therefore the shape library -- untouched.

Labels gate all three, as they do in MATLAB: ``NOSCL``, ``NOPOS`` and
``NOROT`` are read as they are set, stay set until set again, and exempt the
block that carries them.
"""

from __future__ import annotations

__all__ = ["TransformFOV"]

from typing import TYPE_CHECKING

import numpy as np
import pypulseq as pp

from scipy.spatial.transform import Rotation

from ._opts import default_system

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._sequence import Sequence


#: The labels that exempt a block from each transformation.
_NOSCL, _NOPOS, _NOROT = "NOSCL", "NOPOS", "NOROT"


class TransformFOV:
    """A scaling, translation and rotation to apply to a built sequence.

    Parameters
    ----------
    rotation : numpy.ndarray, optional
        A ``(3, 3)`` rotation matrix, taking the sequence's logical frame to
        the frame it should play in.
    translation : tuple of float, optional
        ``(dx, dy, dz)`` along the logical readout, phase and slice axes, in
        **millimetres** -- the units a prescription states them in.
    scale : tuple of float, optional
        Per-axis *gradient* scaling. The field of view along an axis is
        divided by its factor, which is Pulseq's convention; a factor of zero
        flattens that axis.
    transform : numpy.ndarray, optional
        A ``(4, 4)`` homogeneous matrix, as an alternative to giving
        ``rotation`` and ``translation`` separately. Its translation part is
        in the rotated (world) frame, per the usual reading of a homogeneous
        transform, and is converted here. Cannot be combined with either of
        them; ``scale`` may still be given alongside it.
    use_rotation_extension : bool, default True
        Whether the rotation is carried as a ``ROTATIONS`` extension. Only
        ``True`` is implemented -- see the module docstring for why. The
        parameter is kept so the signature matches MATLAB's, and passing
        ``False`` says so rather than quietly doing something else.
    server_mode : bool, default False
        Where the ADC's share of the shift is applied. ``False`` bakes it
        into the ADC alongside the RF, so the file needs nothing downstream --
        what to write when sharing a ``.seq`` with another toolbox. ``True``
        bakes only the RF and instead stores each readout's base k-space
        trajectory, leaving the ADC side to a consumer of ours; that keeps one
        shape per distinct trajectory rather than one per readout.
    prior_phase_cycle : float, default 0.0
        Accepted for signature parity with MATLAB, where it lets a caller
        chain two transform objects by hand. It is not used: the phase this
        would carry is exactly what absolute k already tracks across the whole
        sequence, so chaining needs nothing passed between calls.
    system : pypulseq.Opts, optional
        Hardware limits. Defaults to :attr:`pypulseq.Opts.default`.

    Raises
    ------
    ValueError
        If nothing to do was given, or ``transform`` was combined with
        ``rotation`` or ``translation``.

    Examples
    --------
    Shift the whole volume 10 mm along the slice axis:

    >>> import pulserver.pypulseq as pp
    >>> shift = pp.TransformFOV(translation=(0.0, 0.0, 10.0))
    >>> moved = shift.apply_to_sequence(seq)             # doctest: +SKIP

    Point one module somewhere else, leaving the rest of the scan alone:

    >>> from scipy.spatial.transform import Rotation
    >>> tilt = pp.TransformFOV(                          # doctest: +SKIP
    ...     rotation=Rotation.from_euler("y", 30, degrees=True).as_matrix(),
    ...     translation=(0.0, 0.0, 25.0),
    ... )
    >>> tilt.apply_to_sequence(seq, time_range=[0.0, 0.02], in_place=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        rotation: np.ndarray | None = None,
        translation: tuple[float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
        transform: np.ndarray | None = None,
        use_rotation_extension: bool = True,
        prior_phase_cycle: float = 0.0,
        system: pp.Opts | None = None,
        server_mode: bool = False,
    ) -> None:
        if transform is not None:
            if rotation is not None or translation is not None:
                raise ValueError(
                    "TransformFOV(): `transform` already carries a rotation and a "
                    "translation; give it or give those two, not both"
                )
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4):
                raise ValueError(f"TransformFOV(): `transform` must be (4, 4), got {matrix.shape}")
            rotation = matrix[:3, :3]
            # The homogeneous matrix states its offset in the rotated frame;
            # everything below works in the logical one.
            translation = tuple(rotation.T @ matrix[:3, 3])
        elif rotation is None and translation is None and scale is None:
            raise ValueError(
                "TransformFOV(): give at least one of `rotation`, `translation`, "
                "`scale` or `transform`"
            )

        if not use_rotation_extension:
            raise NotImplementedError(
                "TransformFOV(use_rotation_extension=False) would bake the rotation into "
                "new gradient waveforms, which is not implemented here -- it would cost one "
                "waveform per orientation. Use pulserver.pypulseq.rotate3D directly if that "
                "is really what you want."
            )

        self.rotation = None if rotation is None else np.asarray(rotation, dtype=float)
        if self.rotation is not None and self.rotation.shape != (3, 3):
            raise ValueError(f"TransformFOV(): `rotation` must be (3, 3), got {self.rotation.shape}")

        self.translation = None if translation is None else tuple(float(v) for v in translation)
        self.scale = None if scale is None else tuple(float(v) for v in scale)
        self.use_rotation_extension = use_rotation_extension
        self.server_mode = server_mode
        self.prior_phase_cycle = prior_phase_cycle
        self.system = default_system(system)

        for name, value in (("translation", self.translation), ("scale", self.scale)):
            if value is not None and len(value) != 3:
                raise ValueError(f"TransformFOV(): `{name}` must be three numbers, got {len(value)}")

    def apply_to_sequence(
        self,
        seq: Sequence,
        *,
        time_range: list[float] | None = None,
        in_place: bool = False,
    ) -> Sequence:
        """Apply the transformation to ``seq``.

        Parameters
        ----------
        seq : Sequence
            The sequence to transform.
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default. The
            blocks this window touches are the ones transformed; the rest are
            left exactly as they were, but they still count towards where k
            stands inside it -- a shift applied to part of a scan gives those
            blocks the same phase they would have had from shifting all of it.
        in_place : bool, default False
            Transform ``seq`` itself rather than a copy. The default matches
            MATLAB's ``applyToSeq``, which hands back a new sequence.

        Returns
        -------
        Sequence
            The transformed sequence -- ``seq`` itself when ``in_place``.

        Raises
        ------
        ValueError
            If ``time_range`` is not a two-element window in increasing order.
        """
        target = seq if in_place else seq._clone()

        count = target.num_blocks
        first, last = (1, count) if count == 0 else target._window_for(time_range)

        if count == 0:
            return target

        exempt = _label_gates(target, first, last)

        # Scale first: it changes the amplitudes the translation integrates.
        if self.scale is not None:
            for run_first, run_last in _runs(exempt[_NOSCL], first, last):
                target._native.apply_fov_scale(*self.scale, first=run_first, last=run_last)

        if self.translation is not None:
            shift_m = tuple(value * 1e-3 for value in self.translation)
            for run_first, run_last in _runs(exempt[_NOPOS], first, last):
                target._native.apply_fov_shift(
                    *shift_m,
                    bake_adc=not self.server_mode,
                    first=run_first,
                    last=run_last,
                )

        if self.rotation is not None:
            turn = Rotation.from_matrix(self.rotation)
            for block in range(first, last + 1):
                if not exempt[_NOROT][block - first]:
                    _compose_rotation(target._native, block, turn)

        if self.server_mode and self.translation is not None:
            target._native.attach_base_trajectory()

        # The waveforms moved, so anything derived from them has to be
        # rebuilt -- the scan structure the safety analyses and plot(tr=)
        # cache, above all.
        target._touch()
        return target


# %% local subroutines


def _label_gates(seq: Sequence, first: int, last: int) -> dict[str, list[bool]]:
    """Which of ``first..last`` each of the three labels exempts.

    One pass over the blocks' extension chains. The labels are sticky: a value
    set in one block holds until another block sets it, and the block that
    carries the setting is itself exempt -- both are MATLAB's behaviour. State
    starts clear for every call, matching ``applyToSeq``'s reset.
    """
    state = {_NOSCL: False, _NOPOS: False, _NOROT: False}
    gates: dict[str, list[bool]] = {name: [] for name in state}

    native = seq._native
    for block in range(first, last + 1):
        node = native.get_block(block)[5]
        while node:
            type_id, reference, node = native.extension_row(node)
            if native.extension_type_name(type_id) != "LABELSET":
                continue
            value, label_id = native.label_set_row(reference)
            name = native.label_name(label_id)
            if name in state:
                state[name] = bool(value)
        for name, flagged in state.items():
            gates[name].append(flagged)
    return gates


def _runs(exempt: list[bool], first: int, last: int) -> list[tuple[int, int]]:
    """The maximal runs of blocks in ``first..last`` that are not exempt.

    Contiguous runs rather than single blocks because the C++ side takes a
    range: a scan with no exemptions at all is then one call, not one per
    block.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for offset, flagged in enumerate(exempt):
        block = first + offset
        if flagged:
            if start is not None:
                runs.append((start, block - 1))
                start = None
        elif start is None:
            start = block
    if start is not None:
        runs.append((start, last))
    return runs


def _compose_rotation(native, block: int, turn: Rotation) -> None:
    """Attach ``turn`` to ``block``, composed with any rotation already there.

    Only the extension chain is rebuilt. The block's RF, gradient and ADC ids
    are carried over untouched, so nothing is re-registered and no shape is
    duplicated -- which is the reason for working here rather than through
    :meth:`Sequence.get_block`, whose decoded events would have to be
    registered afresh.

    ``turn`` is applied *after* whatever the block already carried, so the
    total is what baking the rotations into the gradients in that order would
    give. (MATLAB composes these in the opposite order and marks the line with
    a "check left or right rotation" TODO, so there is no settled convention
    to follow; this one is at least self-consistent.)
    """
    rf, gx, gy, gz, adc, head, duration = native.get_block(block)

    chain = []
    node = head
    while node:
        type_id, reference, node = native.extension_row(node)
        chain.append([type_id, reference])

    rotations = native.extension_type_id("ROTATIONS")
    for link in chain:
        if link[0] == rotations:
            existing = Rotation.from_quat(native.rotation_row(link[1]), scalar_first=True)
            link[1] = native.register_rotation(
                (turn * existing).as_quat(canonical=True, scalar_first=True)
            )
            break
    else:
        chain.append(
            [
                rotations,
                native.register_rotation(turn.as_quat(canonical=True, scalar_first=True)),
            ]
        )

    tail = 0
    for type_id, reference in reversed(chain):
        tail = native.append_extension(type_id, reference, tail)

    native.set_block(block, rf, gx, gy, gz, adc, tail, duration)
