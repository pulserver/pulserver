"""Binary cache serialization / deserialization for SequenceCollection."""

__all__ = ['serialize', 'deserialize']

from pathlib import Path

from ._extension._pulseqlib_wrapper import _save_cache, _load_cache
from ._sequence import SequenceCollection


def serialize(
    seq: SequenceCollection,
    path: str | Path,
) -> None:
    """Serialize a loaded sequence collection to a binary cache file.

    The cache is a compact binary representation of the fully parsed
    and segmented collection.  It can be reloaded much faster than
    re-parsing the original ``.seq`` text, which is useful for
    interactive workflows where the same sequence is opened repeatedly.

    Parameters
    ----------
    seq : SequenceCollection
        A fully loaded sequence.
    path : str or Path
        Output file path (e.g. ``'my_sequence.bin'``).

    Raises
    ------
    RuntimeError
        If the C library fails to write the cache.
    """
    _save_cache(seq._cseq, str(path), 0)


def deserialize(
    seq: SequenceCollection,
    path: str | Path,
) -> None:
    """Restore a sequence collection from a previously serialized binary cache.

    This mutates the collection in-place, replacing its internal state
    with the data from the cache file.  The cache must have been created
    from the same ``.seq`` source (the C library verifies a size check).

    Parameters
    ----------
    seq : SequenceCollection
        An already-constructed collection (provides the target buffer).
    path : str or Path
        Path to the binary cache file.

    Raises
    ------
    RuntimeError
        If the cache file is missing, corrupt, or mismatched.
    """
    _load_cache(seq._cseq, str(path))
