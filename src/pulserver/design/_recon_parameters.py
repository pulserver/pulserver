"""Values a sequence states for its reconstruction, carried to the MRD header by the proxy."""

from __future__ import annotations

from typing import Any

#: Definition listing the names of the definitions the proxy copies into the
#: MRD header, separated by spaces.
RECON_PARAMETERS = "ReconParameters"


def recon_parameters(sequence: Any, **values: int | float | str) -> None:
    """State values for the reconstruction as definitions of ``sequence``.

    Each value becomes the definition of its name, and the name is added to the
    ``ReconParameters`` definition. The proxy writes each listed definition
    into the MRD header as a ``userParameterString`` of the same name, which
    a reconstruction reads with :func:`pulserver.mrd.user_parameter`.

    Parameters
    ----------
    sequence : pypulseqpp.Sequence
        The sequence the scan plays.
    **values : int, float or str
        The values, by name; a name is a Pulseq definition key, with no spaces.

    Raises
    ------
    ValueError
        If a name contains white space.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pulserver.design import recon_parameters
    >>> seq = pp.Sequence()
    >>> recon_parameters(seq, QsmAnatomy=1)
    >>> seq.get_definition("ReconParameters"), seq.get_definition("QsmAnatomy")
    ('QsmAnatomy', [1.0])
    """
    listed = str(sequence.get_definition(RECON_PARAMETERS) or "").split()
    for name, value in values.items():
        if not name or any(ch.isspace() for ch in name):
            raise ValueError(f"recon parameter name {name!r} contains white space")
        sequence.set_definition(key=name, value=value)
        if name not in listed:
            listed.append(name)
    sequence.set_definition(key=RECON_PARAMETERS, value=" ".join(listed))
