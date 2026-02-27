"""Serialize / deserialize a sequence collection (linked .seq files)."""

__all__ = ["serialize", "deserialize"]

import copy
from pathlib import Path

import pypulseq as pp


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def serialize(
    seqs: list[pp.Sequence],
    path: str | Path,
) -> list[Path]:
    """Write a sequence collection as a linked chain of ``.seq`` files.

    Each sequence is written to its own ``.seq`` file.  A ``"next"``
    key is added to the ``[DEFINITIONS]`` section of every file except
    the last, pointing to the filename of the next file in the chain.

    Parameters
    ----------
    seqs : list of pp.Sequence
        Ordered list of pypulseq Sequence objects.
    path : str or Path
        Base file path.  For a single sequence the output is
        ``<path>.seq`` (the ``.seq`` suffix is added if missing).
        For multiple sequences the outputs are
        ``<path>_001.seq``, ``<path>_002.seq``, etc.

    Returns
    -------
    list of Path
        Paths of the written files, in order.

    Raises
    ------
    ValueError
        If *seqs* is empty.
    """
    if not seqs:
        raise ValueError("seqs must be a non-empty list of pp.Sequence objects.")

    base = Path(path)
    if base.suffix.lower() == ".seq":
        base = base.with_suffix("")

    n = len(seqs)
    if n == 1:
        filenames = [base.with_suffix(".seq")]
    else:
        filenames = [
            base.parent / f"{base.name}_{i + 1:03d}.seq" for i in range(n)
        ]

    for i, seq in enumerate(seqs):
        seq_copy = copy.deepcopy(seq)
        if i < n - 1:
            seq_copy.set_definition("next", filenames[i + 1].name)
        seq_copy.write(str(filenames[i]))

    return filenames


def deserialize(path: str | Path) -> list[pp.Sequence]:
    """Read a linked chain of ``.seq`` files into a list of sequences.

    Starting from *path*, each file is read with :pymethod:`pp.Sequence.read`.
    If the ``[DEFINITIONS]`` section contains a ``"next"`` key the
    referenced file (resolved relative to the directory of the current
    file) is loaded as well, and so on until no ``"next"`` key is found.

    Parameters
    ----------
    path : str or Path
        Path to the first ``.seq`` file in the chain.

    Returns
    -------
    list of pp.Sequence
        Ordered list of loaded sequences.

    Raises
    ------
    FileNotFoundError
        If *path* (or any ``"next"`` target) does not exist.
    """
    current = Path(path)
    seqs: list[pp.Sequence] = []

    while current is not None:
        if not current.is_file():
            raise FileNotFoundError(f"Sequence file not found: {current}")

        seq = pp.Sequence()
        seq.read(str(current))
        seqs.append(seq)

        next_name = seq.definitions.get("next")
        if next_name:
            current = current.parent / str(next_name)
        else:
            current = None

    return seqs
