"""Playing a finished sequence more than once, written into the block table."""

from __future__ import annotations

__all__ = ["tile"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._sequence import Sequence


def tile(
    seq: Sequence,
    reps: int,
    *,
    label: str = "AVG",
    strip_once: bool = True,
    in_place: bool = False,
) -> Sequence:
    """Repeat ``seq`` ``reps`` times, as :func:`numpy.tile` repeats an array.

    A ``.seq`` describes one pass, and playing it several times is normally
    left to the interpreter, which takes the count from outside the file (on a
    GE scanner, ``opnex``). This resolves it here instead: afterwards the block
    table *is* the scan, in the order it plays, and nothing downstream has to
    be told how many times to read it.

    Whether a subsequence carries its repetitions is stated by whether this
    was called on it, and nothing is written into the file to say so -- a
    subsequence that should be acquired once, an EPI calibration say, simply
    is not tiled.

    Which blocks belong to a single pass is what the ``ONCE`` flag says:

    =========  ======================================================
    ``ONCE``   plays
    =========  ======================================================
    ``1``      on the first repetition only -- a catalyst, a startup
               transient, the dummy repetitions that reach steady state
    ``0``      on every repetition -- the body of the scan
    ``2``      on the last repetition only -- a cooldown, a ramp-down
    =========  ======================================================

    Only the block table grows. A repetition plays the *same* events, so every
    library is untouched -- a million-block scan repeated twice is two million
    rows of six integers and a duration, and not one extra gradient.

    The sequence is deduplicated first, which is not only tidiness: reading
    ``ONCE`` means walking each block's extension chain, and a scan that has
    not been collapsed holds one label row per use rather than one per distinct
    label. Measured on a 512-cubed bSSFP -- 786 465 blocks, a catalyst and a
    closing rewind -- collapsing first takes the pass from 0.75 s to 0.25 s.
    A sequence already deduplicated pays nothing for the check.

    Parameters
    ----------
    seq : Sequence
        The sequence to repeat. Finished: call this last, after the readouts
        are placed and any transform applied.
    reps : int
        How many times the body plays, at least 1. ``1`` is not a no-op: it
        resolves the flags, leaving a file whose block table is the whole of
        what plays.
    label : str, default "AVG"
        Counter stamped with the repetition index, or ``""`` for none. ``AVG``
        because the repetition an interpreter adds is a signal average -- the
        same acquisition, sampled again. ``REP`` is the frame counter of a
        dynamic series and means something else.
    strip_once : bool, default True
        Drop the ``ONCE`` labels once resolved -- they now describe a table
        they no longer fit.
    in_place : bool, default False
        Repeat ``seq`` itself rather than a copy. The default returns a copy,
        matching :func:`numpy.tile` and :meth:`Sequence.remove_duplicates`.

    Returns
    -------
    Sequence
        The repeated sequence -- ``seq`` itself when ``in_place``.

    Raises
    ------
    ValueError
        If ``reps`` is below 1.
    RuntimeError
        If a block carries an ``ONCE`` value outside ``{0, 1, 2}``, if ``reps``
        exceeds 1 with every block marked ``ONCE`` (there is no body to
        repeat), or if the sequence already writes ``label`` -- two meanings on
        one counter is not resolved quietly.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> seq = pp.Sequence(pp.Opts())
    >>> seq.add_block(pp.make_delay(1e-3), pp.make_label("ONCE", "SET", 1))
    1
    >>> seq.add_block(pp.make_delay(2e-3), pp.make_label("ONCE", "SET", 0))
    2
    >>> pp.tile(seq, 3).num_blocks     # the prep once, the body three times
    4

    See Also
    --------
    Sequence.remove_duplicates : the other whole-sequence pass, run before
        writing.
    """
    target = seq if in_place else seq._clone()
    target.remove_duplicates(in_place=True)
    target._expand_repeats(reps, label=label, strip_once=strip_once)
    return target
