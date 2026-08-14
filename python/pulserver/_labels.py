"""The label vocabulary, shared by the design side and the reconstruction side.

A label is how a sequence says something about an acquisition, and almost
everything it can say has an ISMRMRD field waiting for it: a counter becomes an
``EncodingCounters`` index, a flag becomes an ``ismrmrd.ACQ_*`` bit. The
mapping lives here, in one module with no imports, so that
:mod:`pulserver.pypulseq` (which writes the labels) and
:mod:`pulserver.recon` (which reads the flags back) cannot drift apart.

Pulserver's own spellings are canonical -- ``LIN``, ``SLC``, ``LASTSLC`` -- and
these are what a ``.seq`` file carries. MRD spellings are accepted anywhere a
label name is taken and translated by :func:`canonical_label`, so a script
written against ISMRMRD can say ``kspace_encode_step_1`` and mean ``LIN``.
"""

from __future__ import annotations

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
]

#: Counters: per-acquisition integer indices, one ISMRMRD ``EncodingCounters``
#: field each. A counter says which unit an acquisition belongs to; the
#: ``FIRST``/``LAST`` flags say when a unit ends.
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
    "USER0",
    "USER1",
    "USER2",
    "USER3",
    "USER4",
    "USER5",
    "USER6",
    "USER7",
)

#: The counters that say *which image* an acquisition belongs to. A frame is
#: one setting of all of them at once, and it is complete when every encoding
#: position inside it has arrived.
FRAME_COUNTERS = ("SLC", "ECO", "PHS", "REP", "AVG", "SET")

#: The counters that say *where inside a frame* an acquisition sits.
ENCODING_COUNTERS = ("LIN", "PAR", "SEG")

#: Counter to ISMRMRD ``EncodingCounters`` field.
#:
#: ``ACQ`` is absent: it is Pulseq's own spectroscopic counter and MRD has
#: nowhere to put it. So is the readout direction -- ``kspace_encoding_step_0``
#: exists in the header's ``encodingLimits`` as the readout extent, but there
#: is no such ``EncodingCounters`` field and no ``ENCODE_STEP0`` flag, because
#: the readout is the samples inside one acquisition rather than something
#: counted across acquisitions.
#:
#: The user counters are named as ``encodingLimits`` names them; on an
#: acquisition they are one array, so ``USER3`` is ``idx.user[3]``.
MRD_COUNTERS = {
    "LIN": "kspace_encode_step_1",
    "PAR": "kspace_encode_step_2",
    "AVG": "average",
    "SLC": "slice",
    "ECO": "contrast",
    "PHS": "phase",
    "REP": "repetition",
    "SET": "set",
    "SEG": "segment",
    "USER0": "user_0",
    "USER1": "user_1",
    "USER2": "user_2",
    "USER3": "user_3",
    "USER4": "user_4",
    "USER5": "user_5",
    "USER6": "user_6",
    "USER7": "user_7",
}

#: Flag to ISMRMRD acquisition-flag constant, covering every flag ISMRMRD
#: defines. Four groups, and which one a flag is in decides who may set it:
#:
#: * **boundary** -- ``FIRST``/``LAST`` per counter, plus ``LASTSCAN``. A pure
#:   function of the counters and the acquisition order, which is why
#:   :meth:`pulserver.pypulseq.Sequence.auto_label` can derive them.
#: * **classifying** -- what kind of data this acquisition is. Only the
#:   sequence knows, so only the sequence can say.
#: * **transport** -- how the data was packed on the way out. Set by whatever
#:   compresses it; a sequence has no way to know and no reason to write one.
#: * **free** -- ``USERFLAG1``-``USERFLAG8``, for a classification with no
#:   standard name.
MRD_FLAGS = {
    # boundary
    "FIRSTLIN": "ACQ_FIRST_IN_ENCODE_STEP1",
    "LASTLIN": "ACQ_LAST_IN_ENCODE_STEP1",
    "FIRSTPAR": "ACQ_FIRST_IN_ENCODE_STEP2",
    "LASTPAR": "ACQ_LAST_IN_ENCODE_STEP2",
    "FIRSTAVG": "ACQ_FIRST_IN_AVERAGE",
    "LASTAVG": "ACQ_LAST_IN_AVERAGE",
    "FIRSTSLC": "ACQ_FIRST_IN_SLICE",
    "LASTSLC": "ACQ_LAST_IN_SLICE",
    "FIRSTECO": "ACQ_FIRST_IN_CONTRAST",
    "LASTECO": "ACQ_LAST_IN_CONTRAST",
    "FIRSTPHS": "ACQ_FIRST_IN_PHASE",
    "LASTPHS": "ACQ_LAST_IN_PHASE",
    "FIRSTREP": "ACQ_FIRST_IN_REPETITION",
    "LASTREP": "ACQ_LAST_IN_REPETITION",
    "FIRSTSET": "ACQ_FIRST_IN_SET",
    "LASTSET": "ACQ_LAST_IN_SET",
    "FIRSTSEG": "ACQ_FIRST_IN_SEGMENT",
    "LASTSEG": "ACQ_LAST_IN_SEGMENT",
    "LASTSCAN": "ACQ_LAST_IN_MEASUREMENT",
    # classifying
    "NOISE": "ACQ_IS_NOISE_MEASUREMENT",
    "REF": "ACQ_IS_PARALLEL_CALIBRATION",
    "IMA": "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",
    "REV": "ACQ_IS_REVERSE",
    "NAV": "ACQ_IS_NAVIGATION_DATA",
    "PHASECORR": "ACQ_IS_PHASECORR_DATA",
    "HPFEEDBACK": "ACQ_IS_HPFEEDBACK_DATA",
    "DUMMYSCAN": "ACQ_IS_DUMMYSCAN_DATA",
    "RTFEEDBACK": "ACQ_IS_RTFEEDBACK_DATA",
    "COILCORR": "ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA",
    "PHSTABREF": "ACQ_IS_PHASE_STABILIZATION_REFERENCE",
    "PHSTAB": "ACQ_IS_PHASE_STABILIZATION",
    # transport
    "COMPRESS1": "ACQ_COMPRESSION1",
    "COMPRESS2": "ACQ_COMPRESSION2",
    "COMPRESS3": "ACQ_COMPRESSION3",
    "COMPRESS4": "ACQ_COMPRESSION4",
    # free
    "USERFLAG1": "ACQ_USER1",
    "USERFLAG2": "ACQ_USER2",
    "USERFLAG3": "ACQ_USER3",
    "USERFLAG4": "ACQ_USER4",
    "USERFLAG5": "ACQ_USER5",
    "USERFLAG6": "ACQ_USER6",
    "USERFLAG7": "ACQ_USER7",
    "USERFLAG8": "ACQ_USER8",
}

#: Flags the interpreter consumes. These are the ones that change how the
#: scanner plays the sequence rather than what the data is, so they have no MRD
#: counterpart and stop at the scanner.
SCANNER_FLAGS = ("PMC", "NOROT", "NOPOS", "NOSCL", "OFF", "ONCE", "TRID")

#: Flags: booleans (and two ids) describing how a block is played or
#: classified, rather than where its data belongs.
#:
#: ``SMS`` belongs to neither group. It carries a multiband group index, and
#: ISMRMRD has no multiband field of any kind -- converters index the excited
#: group in ``slice``, mark the single-band reference
#: ``ACQ_IS_PARALLEL_CALIBRATION``, and put a band index in ``user``. A band
#: index that has to reach a reconstruction goes in a ``USER*`` counter.
FLAG_LABELS = (*MRD_FLAGS, "SMS", *SCANNER_FLAGS)

#: Flags whose meaning spans more than the module that sets them, so they are
#: *not* auto-reset at the module's last block: ``ONCE`` delimits a whole
#: prep/cooldown section, and ``TRID`` names a repeating unit -- a TR, or a
#: whole contrast of a multi-contrast scan -- for the interpreter.
STICKY_FLAGS = ("ONCE", "TRID")

_CANONICAL = frozenset(COUNTER_LABELS) | frozenset(FLAG_LABELS)

#: Every spelling that is not the canonical one, mapped to the one that is.
#:
#: ``K1``/``K2`` are MRPro's names for the two phase-encoding steps, and mean
#: what its ``AcqIdx`` means by them.
_ALIASES = {
    **{mrd.upper(): name for name, mrd in MRD_COUNTERS.items()},
    **{mrd: name for name, mrd in MRD_FLAGS.items()},
    **{mrd[len("ACQ_") :]: name for name, mrd in MRD_FLAGS.items()},
    "K1": "LIN",
    "K2": "PAR",
}

#: Names that look like labels and cannot be, with the reason. Refused rather
#: than passed through as custom labels, because each is a name someone reaches
#: for expecting it to work, and a custom label that nothing consumes is a
#: silent way to not work.
_REFUSED = {
    "K0": (
        "k0 is the readout direction, which is the samples inside one "
        "acquisition rather than something counted across them -- so there is "
        "no counter to label, in ISMRMRD or in MRPro's AcqIdx. k1 and k2 are "
        "counters and translate to LIN and PAR. The readout extent is stated "
        "once in the header, as encodingLimits.kspace_encoding_step_0."
    ),
}


def canonical_label(name: str) -> str:
    """Return the Pulserver spelling of one label name.

    Accepts the canonical name, the ISMRMRD field or constant name (with or
    without the ``ACQ_`` prefix), or MRPro's ``k1``/``k2``, in any case.
    Anything that matches nothing is returned unchanged, which is what lets a
    sequence carry a label of its own invention.

    Parameters
    ----------
    name : str
        A label name in any accepted spelling.

    Returns
    -------
    str
        The canonical name, or ``name`` unchanged.

    Raises
    ------
    ValueError
        If *name* is one that reads as a label but cannot be one -- ``k0``.

    Examples
    --------
    >>> from pulserver._labels import canonical_label
    >>> canonical_label("kspace_encode_step_1")
    'LIN'
    >>> canonical_label("k1"), canonical_label("k2")
    ('LIN', 'PAR')
    >>> canonical_label("ACQ_LAST_IN_SLICE"), canonical_label("LAST_IN_SLICE")
    ('LASTSLC', 'LASTSLC')
    >>> canonical_label("LIN"), canonical_label("BIN")
    ('LIN', 'BIN')
    """
    upper = name.upper()
    if upper in _CANONICAL:
        return upper
    if upper in _REFUSED:
        raise ValueError(f"{name!r} is not a label: {_REFUSED[upper]}")
    return _ALIASES.get(upper, name)
