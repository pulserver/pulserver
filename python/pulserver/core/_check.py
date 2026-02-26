"""Safety and consistency checks for SequenceCollection."""

__all__ = ['check']

from typing import Sequence as SequenceType

from ._extension._pulseqlib_wrapper import _check_consistency, _check_safety
from ._sequence import SequenceCollection


def check(
    seq: SequenceCollection,
    *,
    forbidden_bands: SequenceType[dict] | None = None,
    pns_chronaxie_us: float = 0.0,
    pns_rheobase: float = 0.0,
    pns_alpha: float = 0.0,
    pns_threshold_percent: float = 100.0,
) -> None:
    """Run consistency and safety checks on a sequence.

    First checks internal consistency (segment boundaries, RF
    periodicity, label tables), then runs safety checks (gradient
    amplitude/slew, acoustic forbidden bands, PNS).

    On failure, raises :class:`RuntimeError` with a diagnostic
    message describing the first violation found.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to check.
    forbidden_bands : list of dict, optional
        Acoustic forbidden-band specifications.  Each dict must have
        keys ``'freq_min_hz'``, ``'freq_max_hz'``, and
        ``'max_amplitude'`` (Hz/m).
    pns_chronaxie_us : float
        PNS nerve model chronaxie (µs).  Set > 0 to enable PNS check.
    pns_rheobase : float
        PNS nerve model rheobase (Hz/m/s).
    pns_alpha : float
        PNS nerve model alpha parameter.
    pns_threshold_percent : float
        PNS threshold as percentage (100 = 100 %).

    Raises
    ------
    RuntimeError
        If a consistency or safety violation is detected.
    """
    _check_consistency(seq._cseq)

    bands = list(forbidden_bands) if forbidden_bands else []
    skip_pns = pns_chronaxie_us <= 0.0

    _check_safety(
        seq._cseq,
        forbidden_bands=bands,
        pns_chronaxie_us=pns_chronaxie_us,
        pns_rheobase=pns_rheobase,
        pns_alpha=pns_alpha,
        pns_threshold_percent=pns_threshold_percent,
        skip_pns=skip_pns,
    )
