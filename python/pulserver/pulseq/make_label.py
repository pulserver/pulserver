"""Custom label support for Sequence."""

__all__ = ["make_label"]

from types import SimpleNamespace


def make_label(label: str, type: str, value: int | bool | float) -> SimpleNamespace:  # noqa: A002
    """Create a label event without validating *label* against the built-in list.

    Identical to :func:`pypulseq.make_label` but does **not** raise
    :exc:`ValueError` for unknown label strings, allowing user-defined labels to
    be passed to :meth:`pulserver.pulseq.Sequence.add_block`. The label is
    auto-registered and retrievable via
    :attr:`pulserver.pulseq.Sequence.custom_labels`.

    Parameters
    ----------
    label:
        Arbitrary label string (built-in or custom).
    type:
        ``'SET'`` or ``'INC'``.
    value:
        Integer counter or flag value.
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
