"""Reconstruction for :mod:`pulserver.app.sequence.mprage3D_sequence`.

Under the coherent view orderings the inversion recovery is smooth in
k-space by construction and the volume reconstructs exactly as any 3D
Cartesian scan does -- so this is
:class:`pulserver.app.recon.gre3D_recon.Gre3DRecon` under the name the runtime
pairs with the sequence. Under ``shuffling`` the recovery is deliberately
scattered for a subspace reconstruction: every acquisition carries its
within-segment index as the ``contrast`` counter, and ``TI``, ``InnerTR``
and ``ViewsPerSegment`` travel in the definitions --
what :mod:`pulserver.app.recon.subspace_basis_recon` simulates its basis from.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Mprage3DRecon"]

from pulserver.app.recon.gre3D_recon import Gre3DRecon


class Mprage3DRecon(Gre3DRecon):
    """Reconstruct a 3D MPRAGE, one volume per measurement."""


PLUGIN = Mprage3DRecon()
