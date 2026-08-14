"""Reconstruction for :mod:`pulserver.seqzoo.fse_3d`.

Under the coherent view orderings (linear, radial, radial-adaptive) the
train's signal modulation is smooth in k-space by construction, and the
volume reconstructs exactly as any 3D Cartesian scan does -- so this is
:class:`pulserver.reczoo.gre_3d.Gre3DRecon` under the name the runtime pairs
with the sequence.

Under ``shuffling`` the modulation is deliberately scattered, and a plain
reconstruction returns its average contrast with noise-like artifacts. That
data is meant for a subspace reconstruction: every acquisition carries its
echo index as the ``contrast`` counter, and the sequence declares
``RefocusingFlipAngles``, ``EchoSpacing`` and ``EchoTrainLength`` -- exactly
what :mod:`pulserver.reczoo.subspace_basis` needs to simulate the signal
basis it solves against.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Fse3DRecon"]

from pulserver.reczoo.gre_3d import Gre3DRecon


class Fse3DRecon(Gre3DRecon):
    """Reconstruct a 3D Cartesian fast spin echo, one volume per measurement."""


PLUGIN = Fse3DRecon()
