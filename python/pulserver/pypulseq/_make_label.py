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
#: :meth:`~pulserver.SequenceModule.set_flags`.
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
    :data:`FLAG_LABELS` say *how a block is played or classified* and come from
    :meth:`pulserver.SequenceModule.set_flags`.

    .. list-table:: Counters — one ISMRMRD ``idx`` field each, set by a scan loop
       :header-rows: 1
       :widths: 14 86

       * - Label
         - Meaning
       * - ``LIN``
         - In-plane phase-encoding line index (``kspace_encode_step_1``).
       * - ``PAR``
         - Through-plane partition index (``kspace_encode_step_2``).
       * - ``SLC``
         - Slice index.
       * - ``ECO``
         - Echo/contrast index. Owned by the readout, which knows its own train.
       * - ``PHS``
         - Cardiac/respiratory phase, or phase-cycle index.
       * - ``REP``
         - Repetition index — the frame counter of a dynamic acquisition.
       * - ``SET``
         - Acquisition-set index — the usual home of a non-echo contrast
           dimension (inversion time, b-value, saturation offset).
       * - ``AVG``
         - Signal-average index.
       * - ``SEG``
         - Segment or shot-within-repetition index.
       * - ``ACQ``
         - Acquisition index.

    .. list-table:: Flags — sticky block properties, set by a sequence module
       :header-rows: 1
       :widths: 14 86

       * - Label
         - Meaning
       * - ``NOROT``
         - Suppress geometric rotation for the labelled block.
       * - ``NOPOS``
         - Suppress position-dependent frequency translation.
       * - ``NOSCL``
         - Suppress geometric gradient scaling.
       * - ``PMC``
         - Prospective-motion-correction acquisition flag.
       * - ``NAV``
         - Navigator acquisition flag; routes data to its own encoding space.
       * - ``REV``
         - Reversed readout-polarity flag.
       * - ``SMS``
         - Simultaneous-multislice group or band index.
       * - ``REF``
         - Calibration/reference acquisition flag.
       * - ``IMA``
         - Imaging acquisition flag.
       * - ``NOISE``
         - Noise-only acquisition flag.
       * - ``OFF``
         - Discard the associated acquisition downstream — an ADC that is
           played (so timing is unchanged) but whose data is dropped.
       * - ``ONCE``
         - Section marker: ``1`` = preparation (played once, leading), ``2`` =
           cooldown (played once, trailing), ``0`` = steady-state body.
       * - ``TRID``
         - Trigger identifier naming a repeating TR block.
       * - ``MODULE``
         - Sticky structural/safety module-group identifier.

    See Also
    --------
    COUNTER_LABELS, FLAG_LABELS, STICKY_FLAGS : the same split as constants.
    pulserver.SequenceModule.set_flags : emit flags with the right scope.
    pulserver.ScanLoop.labels : emit counters from a loop's axes.
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
    >>> seq = pp.Sequence()
    >>> seq.add_block(pp.make_delay(1e-3), make_label("BIN", "SET", 3))
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
