# `pulserver.sequences`

Ready-to-run sequence callbacks for a Python REPL.  They take explicit,
keyword-only sequence controls and return an in-memory
`pulserver.pypulseq.Sequence`; no bridge protocol dictionary or output path is
part of this API.  The bridge plugins under `examples/sequences/` remain the
single implementation: these callbacks invoke those plugins without their
final write side effect.  A returned sequence may therefore be inspected,
combined by an application, or written explicitly with `pulserver.io.write`.

```python
from pulserver.sequences import design_gre_2d

seq = design_gre_2d(te_ms=8.0, tr_ms=250.0, nx=128, ny=128)
```

`pulserver.sequence` is a singular compatibility alias for this namespace.

## Cartesian

GRE, multi-echo GRE, FSE, bSSFP and MPRAGE on a Cartesian (phase-encoded)
readout, each in its 2D and 3D form.

```{eval-rst}
.. autosummary::
   :toctree: generated/sequences
   :nosignatures:

   pulserver.sequences.design_gre
   pulserver.sequences.design_gre_2d
   pulserver.sequences.design_gre_3d
   pulserver.sequences.design_gre_multiecho_2d
   pulserver.sequences.design_gre_multiecho_3d
   pulserver.sequences.design_fse_2d
   pulserver.sequences.design_fse_3d
   pulserver.sequences.design_bssfp_2d
   pulserver.sequences.design_bssfp_3d
   pulserver.sequences.design_mprage_2d
   pulserver.sequences.design_mprage_3d
```

## EPI

EPI callbacks construct one volume.  For structural imaging, use the desired
high-resolution geometry once; for fMRI, invoke the same callback once per
volume with the functional geometry and TR.  This keeps volume scheduling in
the calling sequence rather than coupling it to readout design.  See each
function's own page for its structural and fMRI-volume renderings.

```{eval-rst}
.. autosummary::
   :toctree: generated/sequences
   :nosignatures:

   pulserver.sequences.design_epi_2d
   pulserver.sequences.design_epi_3d
```

## Non-Cartesian

GRE and MPRAGE on a non-Cartesian readout.  `design_gre_noncart_*` selects
its in-plane base waveform (spiral or rosette) through its `trajectory`
argument; `design_gre_radial_*` and `design_gre_mprage_radial_*` are the
plain-trapezoid radial family; ZTE is the half-spoke radial variant with a
near-zero echo time.

```{eval-rst}
.. autosummary::
   :toctree: generated/sequences
   :nosignatures:

   pulserver.sequences.design_gre_noncart_2d
   pulserver.sequences.design_gre_noncart_3d
   pulserver.sequences.design_gre_radial_2d
   pulserver.sequences.design_gre_radial_3d
   pulserver.sequences.design_gre_mprage_radial_2d
   pulserver.sequences.design_gre_mprage_radial_3d
   pulserver.sequences.design_zte_2d
   pulserver.sequences.design_zte_3d
```
