"""Custom label support for Sequence."""

__all__ = [
    "COUNTER_LABELS",
    "ENCODING_COUNTERS",
    "FLAG_LABELS",
    "FRAME_COUNTERS",
    "MRD_COUNTERS",
    "MRD_FLAGS",
    "SCANNER_FLAGS",
    "STICKY_FLAGS",
    "canonical_label",
    "get_supported_labels",
    "make_label",
]

from types import SimpleNamespace

from ._events import convert as _convert
from .._labels import (
    COUNTER_LABELS,
    ENCODING_COUNTERS,
    FLAG_LABELS,
    FRAME_COUNTERS,
    MRD_COUNTERS,
    MRD_FLAGS,
    SCANNER_FLAGS,
    STICKY_FLAGS,
    canonical_label,
)

_SUPPORTED_LABELS = COUNTER_LABELS + FLAG_LABELS


def get_supported_labels() -> tuple[str, ...]:
    """Return every counter and flag understood by Pulserver.

    The set is ISMRMRD's, under Pulseq's spellings. Every counter is an
    ``EncodingCounters`` field and every flag but one is an
    ``ismrmrd.ACQ_*`` bit, plus the handful of names that steer the
    interpreter and have no ISMRMRD meaning. Names may be written in either
    spelling — :func:`~pulserver.pypulseq.make_label` translates the ISMRMRD
    one — and :data:`~pulserver.pypulseq.MRD_COUNTERS` and
    :data:`~pulserver.pypulseq.MRD_FLAGS` are the map.

    Two divisions run through the set. The first is the design toolbox's
    division of labour: :data:`COUNTER_LABELS` say *where an acquisition
    belongs*, :data:`FLAG_LABELS` say *how a block is played or classified*.
    The second is the one worth knowing before choosing a label — whether it
    **maps to data**. Every counter does, and so does every flag with an
    ISMRMRD constant beside it; the rest are instructions to the interpreter
    and go no further than the scanner, so a sequence that needs to tell its
    own reconstruction something cannot say it with one of those.

    .. list-table:: Counters — one ISMRMRD ``idx`` field each
       :header-rows: 1
       :widths: 10 26 64

       * - Label
         - ISMRMRD
         - Meaning
       * - ``LIN``
         - ``kspace_encode_step_1``
         - Line counter in 2D and 3D acquisitions — the in-plane
           phase-encoding step. Also spelled ``k1``, as MRPro spells it.
       * - ``PAR``
         - ``kspace_encode_step_2``
         - Partition counter: the phase-encoding step in the second,
           through-slab direction of a 3D sequence. Also ``k2``.
       * - ``SLC``
         - ``slice``
         - Slice counter, or slab counter for a 3D multi-slab sequence.
       * - ``ECO``
         - ``contrast``
         - Echo counter in a multi-echo sequence. Owned by the readout, which
           knows its own train.
       * - ``PHS``
         - ``phase``
         - Cardiac or respiratory phase counter.
       * - ``REP``
         - ``repetition``
         - Repetition counter — the frame index of a dynamic acquisition.
       * - ``AVG``
         - ``average``
         - Averaging counter. What
           :meth:`pulserver.pypulseq.Sequence.expand_repeats` stamps when it
           writes the repetitions into the block table.
       * - ``SET``
         - ``set``
         - Flexible counter with no firm assignment — in practice the usual
           home of a non-echo contrast dimension (inversion time, b-value,
           saturation offset).
       * - ``SEG``
         - ``segment``
         - Segment counter, for a segmented FLASH or EPI.
       * - ``USER0``-``USER7``
         - ``user[0]``-``user[7]``
         - Eight counters with no assigned meaning, for an axis the standard
           set has no name for — an SMS band index, a preparation state, a
           respiratory bin.
       * - ``ACQ``
         - —
         - Spectroscopic acquisition counter. Pulseq's own; ISMRMRD has
           nowhere to put it, so it does not reach a reconstruction.

    Nothing maps to ``kspace_encoding_step_0``. It appears in the ISMRMRD
    header's ``encodingLimits`` as the readout extent, but there is no such
    ``EncodingCounters`` field and no ``ENCODE_STEP0`` flag: the readout is the
    samples *inside* one acquisition, not something counted across them. So
    ``k1`` and ``k2`` are labels and ``k0`` is refused rather than accepted as
    a custom one, which would be a name that quietly does nothing.

    .. list-table:: Boundary flags — when a unit is complete
       :header-rows: 1
       :widths: 22 34 44

       * - Label
         - ISMRMRD
         - Meaning
       * - ``FIRSTLIN`` / ``LASTLIN``
         - ``ACQ_{FIRST,LAST}_IN_ENCODE_STEP1``
         - First and last acquisition of a ``LIN``.
       * - ``FIRSTPAR`` / ``LASTPAR``
         - ``ACQ_{FIRST,LAST}_IN_ENCODE_STEP2``
         - … of a ``PAR``.
       * - ``FIRSTSLC`` / ``LASTSLC``
         - ``ACQ_{FIRST,LAST}_IN_SLICE``
         - … of an ``SLC``, so everything the slice will ever contain has
           arrived.
       * - ``FIRSTSEG`` / ``LASTSEG``
         - ``ACQ_{FIRST,LAST}_IN_SEGMENT``
         - … of a ``SEG``. A reconstruction with work to do the moment a block
           of acquisitions is complete — calibrating coil sensitivities from an
           autocalibration block, say — triggers on this rather than waiting
           for the scan.
       * - ``FIRSTECO`` / ``LASTECO``
         - ``ACQ_{FIRST,LAST}_IN_CONTRAST``
         - … of an ``ECO``.
       * - ``FIRSTPHS`` / ``LASTPHS``
         - ``ACQ_{FIRST,LAST}_IN_PHASE``
         - … of a ``PHS``.
       * - ``FIRSTREP`` / ``LASTREP``
         - ``ACQ_{FIRST,LAST}_IN_REPETITION``
         - … of a ``REP``.
       * - ``FIRSTAVG`` / ``LASTAVG``
         - ``ACQ_{FIRST,LAST}_IN_AVERAGE``
         - … of an ``AVG``.
       * - ``FIRSTSET`` / ``LASTSET``
         - ``ACQ_{FIRST,LAST}_IN_SET``
         - … of a ``SET``.
       * - ``LASTSCAN``
         - ``ACQ_LAST_IN_MEASUREMENT``
         - The final acquisition of the scan.

    These are a function of the counters and the order the scan acquires them
    in, so a sequence need not work them out:
    :meth:`pulserver.pypulseq.Sequence.auto_label` derives every one whose
    counter varies, and leaves alone any the sequence set itself.

    .. list-table:: Classifying flags — what kind of data this is
       :header-rows: 1
       :widths: 16 40 44

       * - Label
         - ISMRMRD
         - Meaning
       * - ``REF``
         - ``ACQ_IS_PARALLEL_CALIBRATION``
         - Parallel-imaging reference / auto-calibration data.
       * - ``IMA``
         - ``ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING``
         - Calibration data that is also imaging data — an autocalibration
           block inside the imaging k-space rather than beside it.
       * - ``NOISE``
         - ``ACQ_IS_NOISE_MEASUREMENT``
         - Noise-adjust scan, e.g. for parallel-imaging acceleration.
       * - ``NAV``
         - ``ACQ_IS_NAVIGATION_DATA``
         - Navigator data; routes data to its own encoding space.
       * - ``REV``
         - ``ACQ_IS_REVERSE``
         - The readout direction is reversed.
       * - ``PHASECORR``
         - ``ACQ_IS_PHASECORR_DATA``
         - Phase-correction readouts, as an EPI plays before its train.
       * - ``DUMMYSCAN``
         - ``ACQ_IS_DUMMYSCAN_DATA``
         - Acquired while the magnetisation is still approaching steady state.
       * - ``RTFEEDBACK``
         - ``ACQ_IS_RTFEEDBACK_DATA``
         - Feeds a real-time control loop rather than an image.
       * - ``HPFEEDBACK``
         - ``ACQ_IS_HPFEEDBACK_DATA``
         - High-priority feedback, on the same principle.
       * - ``COILCORR``
         - ``ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA``
         - Surface-coil correction scan.
       * - ``PHSTAB`` / ``PHSTABREF``
         - ``ACQ_IS_PHASE_STABILIZATION``, ``…_REFERENCE``
         - Phase-stabilisation readout, and the reference it is measured
           against.
       * - ``USERFLAG1``-``USERFLAG8``
         - ``ACQ_USER1``-``ACQ_USER8``
         - Eight flags with no assigned meaning, for a classification the
           standard set has no name for.

    ``SMS`` is Pulseq's multiband marker and belongs to neither group. ISMRMRD
    has no multiband field at all: converters index the excited *group* in
    ``slice``, mark the single-band reference ``ACQ_IS_PARALLEL_CALIBRATION``,
    and put a band index in ``user``. A band index that has to reach a
    reconstruction goes in a ``USER*`` counter.

    ``COMPRESS1``-``COMPRESS4`` (``ACQ_COMPRESSION1``-``4``) describe how the
    data was packed on the way out. Whatever compresses it sets them; a
    sequence has no way to know and no reason to write one.

    .. list-table:: Flags that steer the interpreter — these stop at the scanner
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
       * - ``OFF``
         - Pulserver's own: discard the acquisition downstream — an ADC that
           is played, so timing is unchanged, but whose data is dropped.
       * - ``ONCE``
         - Three-state: alter the sequence when playing multiple repeats.
           ``0`` plays on every repetition, ``1`` on the first only
           (preparation), ``2`` on the last only (cooldown). Resolved into the
           block table by
           :meth:`pulserver.pypulseq.Sequence.expand_repeats`.
       * - ``TRID``
         - Sticky id naming the repeating unit a block belongs to — a TR, or
           a whole contrast of a multi-contrast scan. Units with different
           timing take different ``TRID`` values. Pulseq defines it as "an
           integer ID of the TR (sequence segment) used by the GE interpreter
           (and some others) to optimize the execution on the scanner", and
           MATLAB's ``seq.addTRID('fat_suppression')`` is how a design names
           one. Pulserver reads it as **the safety group**: a hyper-TR made
           of several contrasts is checked per contrast against the 10 s SAR
           limit and as a whole against the 6 min one, rather than as one
           long lump.

    Notes
    -----
    ``ONCE`` and ``TRID`` are counters in Pulseq's own taxonomy rather than
    booleans, and are grouped with the flags here because the split this
    module draws is by *what a label is for*, not by what it holds: neither
    says where an acquisition belongs, and neither reaches a reconstruction.

    See Also
    --------
    COUNTER_LABELS, FLAG_LABELS, SCANNER_FLAGS, STICKY_FLAGS : the same splits
        as constants.
    MRD_COUNTERS, MRD_FLAGS : the ISMRMRD map, as dictionaries.
    canonical_label : resolve one name from either spelling.
    pulserver.pypulseq.Sequence.auto_label : derive counters and boundary flags.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> labels = pp.get_supported_labels()
    >>> "LIN" in labels, "TRID" in labels
    (True, True)

    Only the canonical spellings are listed; the ISMRMRD one is accepted
    everywhere a name is taken, and resolves to it:

    >>> "kspace_encode_step_1" in labels
    False
    >>> pp.canonical_label("kspace_encode_step_1")
    'LIN'

    The two splits partition what is here. A counter carries an ISMRMRD
    field:

    >>> set(pp.COUNTER_LABELS) <= set(labels)
    True
    >>> pp.MRD_COUNTERS["LIN"]
    'kspace_encode_step_1'

    A scanner flag carries none, which is what makes it unable to reach a
    reconstruction:

    >>> set(pp.SCANNER_FLAGS) & set(pp.MRD_FLAGS)
    set()
    >>> pp.STICKY_FLAGS
    ('ONCE', 'TRID')
    """
    return _SUPPORTED_LABELS


def make_label(label: str, type: str, value: int | bool | float):  # noqa: A002
    """Create a label event without validating *label* against the built-in list.

    Identical to :func:`pypulseq.make_label` but does **not** raise
    :exc:`ValueError` for unknown label strings, allowing user-defined labels to
    be passed to :meth:`pulserver.pypulseq.Sequence.add_block`. The label is
    auto-registered, written to the file, and read back by name;
    :meth:`pulserver.pypulseq.Sequence.evaluate_labels` reports its value.
    Recognising it is the interpreter's business, and one it does not
    recognise it ignores.

    Labels are how a sequence tells the reconstruction *what* each acquisition
    is — which line, partition, echo, slice, average. Upstream restricts them
    to the Pulseq built-in set; this version accepts any string, so a sequence
    can carry its own bookkeeping (a bin index, a preparation state) through to
    the reconstruction without abusing a built-in counter.

    Parameters
    ----------
    label : str
        Label name, in any spelling: Pulseq's (``"LIN"``), ISMRMRD's
        (``"kspace_encode_step_1"``, ``"ACQ_LAST_IN_SLICE"``), MRPro's
        (``"k1"``, ``"k2"``), or a custom string, which passes through
        untouched.
    type : {'SET', 'INC'}
        ``'SET'`` assigns ``value``; ``'INC'`` adds it to the running counter.
    value : int or bool or float
        Counter or flag value; coerced to ``int``.

    Returns
    -------
    LabelEvent
        Label event with ``type`` ``'labelset'`` or ``'labelinc'``, its fields
        in slots. Its ``label`` is the canonical name, which is what the
        ``.seq`` file carries.

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

    An ISMRMRD name means the label it maps to:

    >>> make_label("kspace_encode_step_1", "SET", 12).label
    'LIN'
    >>> make_label("k2", "SET", 3).label
    'PAR'
    >>> make_label("ACQ_LAST_IN_SLICE", "SET", 1).label
    'LASTSLC'

    A custom label passes through unvalidated, and is readable afterwards
    beside the built-in ones:

    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> seq = pp.Sequence(system)
    >>> _ = seq.add_block(
    ...     pp.make_delay(1e-3), make_label("BIN", "SET", 3), make_label("LIN", "SET", 2)
    ... )
    >>> seq.evaluate_labels() == {"BIN": 3, "LIN": 2}
    True

    See Also
    --------
    get_supported_labels : the built-in label set.
    canonical_label : the name translation, on its own.
    pulserver.pypulseq.Sequence.evaluate_labels : label values at a block.
    """
    out = SimpleNamespace()
    if type == "SET":
        out.type = "labelset"
    elif type == "INC":
        out.type = "labelinc"
    else:
        raise ValueError("Invalid type. Must be one of 'SET' or 'INC'.")
    out.label = canonical_label(label)
    out.value = int(value)
    return _convert(out)
