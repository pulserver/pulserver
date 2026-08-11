"""Custom label support for Sequence."""

__all__ = [
    "COUNTER_LABELS",
    "FLAG_LABELS",
    "STICKY_FLAGS",
    "get_supported_labels",
    "make_label",
]

from types import SimpleNamespace

#: Counters: per-acquisition integer indices, one ISMRMRD ``EncodingCounters``
#: field each. These are what a :class:`~pulserver.ScanLoop` axis emits — the
#: reconstruction sorts data by them, and the interpreter derives the
#: ``FIRST_IN_*``/``LAST_IN_*`` MRD flags from their observed range.
#:
#: Every one of them maps to data. That is what separates them from the flags
#: below, half of which are directives to the interpreter and never reach a
#: reconstruction at all.
COUNTER_LABELS = (
    "SLC",
    "SEG",
    "REP",
    "AVG",
    "SET",
    "ECO",
    "PHS",
    "LIN",
    "PAR",
    "ACQ",
)

#: Flags: sticky booleans (and two sticky ids) describing *how a block is
#: played or classified*, not where its data belongs. These are what a
#: :class:`~pulserver.SequenceModule` emits, through
#: :meth:`~pulserver.SequenceModule.set_state`.
#:
#: They divide again, and the division matters more than it looks. ``NAV``,
#: ``REV``, ``SMS``, ``REF``, ``IMA``, ``NOISE`` and ``OFF`` **classify the
#: data** and become acquisition flags a reconstruction reads. ``PMC``,
#: ``NOROT``, ``NOPOS``, ``NOSCL``, ``ONCE``, ``TRID`` and ``MODULE``
#: **instruct the interpreter** and stop there: nothing downstream of the
#: scanner ever sees them, so a sequence cannot use one to say something to
#: its own reconstruction.
FLAG_LABELS = (
    "NAV",
    "REV",
    "SMS",
    "REF",
    "IMA",
    "NOISE",
    "PMC",
    "NOROT",
    "NOPOS",
    "NOSCL",
    "OFF",
    "ONCE",
    "TRID",
    "MODULE",
)

#: Flags whose meaning spans more than the module that sets them, so they are
#: *not* auto-reset at the module's last block: ``ONCE`` delimits a whole
#: prep/cooldown section, ``MODULE`` groups consecutive modules under one
#: safety id, ``TRID`` names a repeating TR block for the interpreter.
STICKY_FLAGS = ("ONCE", "MODULE", "TRID")

_SUPPORTED_LABELS = COUNTER_LABELS + FLAG_LABELS


def get_supported_labels() -> tuple[str, ...]:
    """Return every counter and flag understood by Pulserver.

    This extends the Pulseq/PyPulseq set with ``OFF`` (discard an acquisition)
    and ``MODULE`` (mark module scope), both consumed by Pulserver's
    interpreter.

    The set splits in two, and the split is the design toolbox's division of
    labour: :data:`COUNTER_LABELS` say *where an acquisition belongs* and come
    from a :class:`~pulserver.ScanLoop` axis;
    :data:`FLAG_LABELS` say *how a block is played or classified*. Both are
    written through :meth:`pulserver.SequenceModule.set_state`, which tells them
    apart by name.

    A third distinction cuts across the second and is the one worth knowing
    before choosing a label: whether it **maps to data**. Every counter does,
    and so do the flags that classify an acquisition — those become fields a
    reconstruction reads. The remaining flags are instructions to the
    interpreter and go no further than the scanner, so a sequence that needs
    to tell its own reconstruction something cannot say it with one of those.

    .. list-table:: Counters — one ISMRMRD ``idx`` field each, set by a scan loop
       :header-rows: 1
       :widths: 12 88

       * - Label
         - Meaning
       * - ``LIN``
         - Line counter in 2D and 3D acquisitions — the in-plane
           phase-encoding step (``kspace_encode_step_1``).
       * - ``PAR``
         - Partition counter: the phase-encoding step in the second,
           through-slab direction of a 3D sequence (``kspace_encode_step_2``).
       * - ``SLC``
         - Slice counter, or slab counter for a 3D multi-slab sequence.
       * - ``ECO``
         - Echo counter in a multi-echo sequence. Owned by the readout, which
           knows its own train.
       * - ``PHS``
         - Cardiac or respiratory phase counter.
       * - ``REP``
         - Repetition counter — the frame index of a dynamic acquisition.
       * - ``AVG``
         - Averaging counter. What
           :meth:`pulserver.pypulseq.Sequence.expand_repeats` stamps when it
           writes the repetitions into the block table.
       * - ``SET``
         - Flexible counter with no firm assignment — in practice the usual
           home of a non-echo contrast dimension (inversion time, b-value,
           saturation offset).
       * - ``SEG``
         - Segment counter, for a segmented FLASH or EPI.
       * - ``ACQ``
         - Spectroscopic acquisition counter.

    .. list-table:: Flags that classify the data — these reach a reconstruction
       :header-rows: 1
       :widths: 12 88

       * - Label
         - Meaning
       * - ``NAV``
         - Navigator data flag; routes data to its own encoding space.
       * - ``REV``
         - The readout direction is reversed.
       * - ``SMS``
         - Simultaneous-multislice acquisition (group or band index).
       * - ``REF``
         - Parallel-imaging reference / auto-calibration data.
       * - ``IMA``
         - Parallel-imaging imaging data *within* the ACS region.
       * - ``NOISE``
         - Noise-adjust scan, e.g. for parallel-imaging acceleration.
       * - ``OFF``
         - Pulserver's own: discard the acquisition downstream — an ADC that
           is played, so timing is unchanged, but whose data is dropped.

    .. list-table:: Flags that instruct the interpreter — these stop at the scanner
       :header-rows: 1
       :widths: 12 88

       * - Label
         - Meaning
       * - ``NOROT``
         - Ignore the FOV rotation prescribed on the UI for these blocks.
       * - ``NOPOS``
         - Ignore the FOV offset prescribed on the UI for these blocks.
       * - ``NOSCL``
         - Ignore the FOV scaling prescribed on the UI for these blocks.
       * - ``PMC``
         - Mark blocks that can or should be prospectively motion-corrected.
       * - ``ONCE``
         - Three-state: alter the sequence when playing multiple repeats.
           ``0`` plays on every repetition, ``1`` on the first only
           (preparation), ``2`` on the last only (cooldown). Resolved into the
           block table by
           :meth:`pulserver.pypulseq.Sequence.expand_repeats`.
       * - ``TRID``
         - Counter marking the beginning of a repeatable module such as a TR.
           Modules with different timing take different ``TRID`` values.
       * - ``MODULE``
         - Pulserver's own: sticky structural/safety module-group identifier.

    Notes
    -----
    ``ONCE`` and ``TRID`` are counters in Pulseq's own taxonomy rather than
    booleans, and are grouped with the flags here because the split this
    module draws is by *what a label is for*, not by what it holds: neither
    says where an acquisition belongs, and neither reaches a reconstruction.

    See Also
    --------
    COUNTER_LABELS, FLAG_LABELS, STICKY_FLAGS : the same split as constants.
    pulserver.SequenceModule.set_state : emit counters and flags on a module.
    pulserver.ScanLoop.label_state : the counter values from a loop's axes.
    """
    return _SUPPORTED_LABELS


def make_label(label: str, type: str, value: int | bool | float) -> SimpleNamespace:  # noqa: A002
    """Create a label event without validating *label* against the built-in list.

    Identical to :func:`pypulseq.make_label` but does **not** raise
    :exc:`ValueError` for unknown label strings, allowing user-defined labels to
    be passed to :meth:`pulserver.pypulseq.Sequence.add_block`. The label is
    auto-registered and retrievable via
    :attr:`pulserver.pypulseq.Sequence.custom_labels`.

    Labels are how a sequence tells the reconstruction *what* each acquisition
    is — which line, partition, echo, slice, average. Upstream restricts them
    to the Pulseq built-in set; this version accepts any string, so a sequence
    can carry its own bookkeeping (a bin index, a preparation state) through to
    the reconstruction without abusing a built-in counter.

    Parameters
    ----------
    label : str
        Arbitrary label string, built-in or custom.
    type : {'SET', 'INC'}
        ``'SET'`` assigns ``value``; ``'INC'`` adds it to the running counter.
    value : int or bool or float
        Counter or flag value; coerced to ``int``.

    Returns
    -------
    types.SimpleNamespace
        Label event with ``type`` ``'labelset'`` or ``'labelinc'``.

    Raises
    ------
    ValueError
        If ``type`` is neither ``'SET'`` nor ``'INC'``.

    Examples
    --------
    >>> from pulserver.pypulseq import make_label
    >>> event = make_label("LIN", "SET", 12)
    >>> event.type, event.label, event.value
    ('labelset', 'LIN', 12)

    A custom label passes through unvalidated, and is retrievable afterwards:

    >>> import pulserver.pypulseq as pp
    >>> label = make_label("BIN", "SET", 3)
    >>> delay = pp.make_delay(1e-3)
    >>> seq = pp.Sequence(pp.Opts(), 1, (delay, label))
    >>> seq.add_block(delay, label)
    >>> sorted(seq.custom_labels)
    ['BIN']

    See Also
    --------
    get_supported_labels : the built-in label set.
    """
    out = SimpleNamespace()
    if type == "SET":
        out.type = "labelset"
    elif type == "INC":
        out.type = "labelinc"
    else:
        raise ValueError("Invalid type. Must be one of 'SET' or 'INC'.")
    out.label = label
    out.value = int(value)
    return out
