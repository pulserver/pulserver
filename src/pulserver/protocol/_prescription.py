"""The prescription entries: where the interpreter asks for the field of view to be."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._keys import FloatKey

#: The entries a protocol carries for the scanner's prescription: the
#: translation of the field-of-view centre along the logical readout, phase and
#: slice axes, in mm. The interpreter fills them from its prescription; no
#: sequence argument receives them, and pulserver applies the translation when
#: it builds the IR.
PRESCRIPTION = (
    str(FloatKey.FOV_OFFSET_X),
    str(FloatKey.FOV_OFFSET_Y),
    str(FloatKey.FOV_OFFSET_Z),
)


def prescribed_offset(values: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return the field-of-view offset protocol values carry, in metres.

    Along the logical readout, phase and slice axes; zero along an axis whose
    entry is absent.
    """
    x, y, z = (float(values.get(name, 0.0)) * 1e-3 for name in PRESCRIPTION)
    return x, y, z
