"""The sequence half of the zoo: one complete sequence per module.

Every module here is installed into :mod:`pulserver.app` and reached from
there -- ``pulserver.app.gre2D_sequence``, whatever directory it was written
in. Each is a worked example of the whole authoring stack: the
:mod:`pulserver.design` modules it composes, the encoding plan it builds from
:mod:`pulserver.pypulseq`, and the loop that writes the blocks::

    from pulserver.app import gre2D_sequence

    seq = gre2D_sequence(n_x=128, n_y=128, n_slices=5)
    seq.write("gre2D.seq")

The reconstructions live beside them in the same namespace. There is one per
sampling rather than one per sequence, because what a reconstruction needs to
know is how k-space was covered: every 2D Cartesian scan here comes back
through ``cartesian2D_recon``.
"""

from __future__ import annotations

import importlib

__all__ = [
    "bssfp2D_sequence",
    "bssfp3D_sequence",
    "epi2D_sequence",
    "epi3D_sequence",
    "fse2D_sequence",
    "fse3D_sequence",
    "gre2D_sequence",
    "gre3D_sequence",
    "gre_multiecho2D_sequence",
    "gre_multiecho3D_sequence",
    "gre_radial2D_sequence",
    "gre_spiral2D_sequence",
    "gre_stack_of_spirals3D_sequence",
    "gre_stack_of_stars3D_sequence",
    "mprage3D_sequence",
    "mprage_stack_of_spirals3D_sequence",
    "se2D_sequence",
    "se3D_sequence",
    "se_propeller2D_sequence",
    "zte3D_sequence",
]


def __getattr__(name: str):
    """Import one sequence on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the sequences this package ships."""
    return sorted(__all__)
