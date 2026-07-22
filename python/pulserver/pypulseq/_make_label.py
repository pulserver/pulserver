"""Custom label support for Sequence."""

__all__ = ["get_supported_labels", "make_label"]

from types import SimpleNamespace

_SUPPORTED_LABELS = (
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
    "ONCE",
    "TRID",
    "OFF",
    "MODULE",
)


def get_supported_labels() -> tuple[str, ...]:
    """Return every counter and flag understood by Pulserver.

    This extends the Pulseq/PyPulseq set with ``OFF`` (discard an acquisition)
    and ``MODULE`` (mark module scope), both consumed by Pulserver's
    interpreter.

    .. list-table:: Supported labels
       :header-rows: 1
       :widths: 14 86

       * - Label
         - Meaning
       * - ``SLC``
         - Slice index.
       * - ``SEG``
         - Segment or shot-within-repetition index.
       * - ``REP``
         - Repetition index.
       * - ``AVG``
         - Signal-average index.
       * - ``SET``
         - Acquisition-set index.
       * - ``ECO``
         - Echo index.
       * - ``PHS``
         - Phase-cycle or phase-contrast index.
       * - ``LIN``
         - In-plane phase-encoding line index.
       * - ``PAR``
         - Through-plane partition index.
       * - ``ACQ``
         - Acquisition index.
       * - ``NAV``
         - Navigator acquisition flag.
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
       * - ``PMC``
         - Prospective-motion-correction acquisition flag.
       * - ``NOROT``
         - Suppress geometric rotation for the labelled module.
       * - ``NOPOS``
         - Suppress position-dependent frequency translation.
       * - ``NOSCL``
         - Suppress geometric gradient scaling.
       * - ``ONCE``
         - Execute or emit the labelled block only once.
       * - ``TRID``
         - Trigger identifier.
       * - ``OFF``
         - Discard the associated acquisition downstream.
       * - ``MODULE``
         - Sticky structural/safety module-group identifier.
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
