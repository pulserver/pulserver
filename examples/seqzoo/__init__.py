"""Pulserver's sequence zoo: one complete, self-contained sequence per module.

``pyproject.toml`` packages this directory as :mod:`pulserver.seqzoo`. Each
module is a worked example of the whole authoring stack — the
:mod:`pulserver.design` modules it composes, the encoding plan it builds from
:mod:`pulserver.pypulseq`, and the loop that writes the blocks — and each
carries two entry points over one implementation:

``main(...)``
    Explicit keyword controls, returns a :class:`pulserver.pypulseq.Sequence`.
    This is the whole sequence, written in the style of a PyPulseq example
    script, and it is what to read, copy and edit.
``PLUGIN``
    A :class:`pulserver.SequencePlugin` wrapping that same ``main`` in the scanner
    protocol contract, so the bridge can offer it in the UI. Running the module
    as a script builds a ``.seq`` offline from the same controls.

::

    from pulserver.seqzoo import gre_2d

    seq = gre_2d.main(n_x=128, n_y=128, n_slices=5)
    seq.write("gre_2d.seq")

:mod:`pulserver.reczoo` holds the reconstruction that matches each sequence,
under the same module name.
"""

from __future__ import annotations

import importlib

__all__ = [
    "bssfp_2d",
    "bssfp_3d",
    "epi_2d",
    "epi_3d",
    "fse_2d",
    "fse_3d",
    "gre_2d",
    "gre_3d",
    "gre_multiecho_2d",
    "gre_multiecho_3d",
    "gre_radial_2d",
    "gre_spiral_2d",
    "gre_stack_of_spirals_3d",
    "gre_stack_of_stars_3d",
    "mprage_3d",
    "mprage_stack_of_spirals_3d",
    "se_2d",
    "se_3d",
    "se_propeller_2d",
    "zte_3d",
]


def __getattr__(name: str):
    """Import one sequence module on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the sequences this zoo ships."""
    return sorted(__all__)
