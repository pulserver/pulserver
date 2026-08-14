"""Reconstruction for :mod:`pulserver.seqzoo.fse_2d`.

An FSE shot fills the same Cartesian grid line by line as its gradient-echo
counterpart -- the train's ordering decides *when* each line arrives, and
the counters already say *where*. The signal modulation along the train is
the sequence's contrast mechanism, not something to undo here. So this is
:class:`pulserver.reczoo.gre_2d.Gre2DRecon` under the name the runtime
pairs with the sequence.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Fse2DRecon"]

from pulserver.reczoo.gre_2d import Gre2DRecon


class Fse2DRecon(Gre2DRecon):
    """Reconstruct a 2D Cartesian fast spin echo, one image per slice."""


PLUGIN = Fse2DRecon()
