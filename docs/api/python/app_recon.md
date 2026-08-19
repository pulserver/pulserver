# pulserver.app (reconstruction)

Ready-made reconstructions, one per **sampling**. There is deliberately not one
per sequence: what a reconstruction has to know is how k-space was covered, not
which contrast the sequence was after. A spin echo, a gradient echo, an MPRAGE
and a balanced SSFP all leave one Cartesian grid, so one plugin serves them
all, and a radial, spiral or PROPELLER scan differs only in the trajectory its
acquisitions carry.

```{eval-rst}
.. currentmodule:: pulserver.app
```

## Calling one

Every module is callable, and calling it reconstructs an MRD file:

```python
from pulserver.app import cartesian2D_recon

images = cartesian2D_recon("scan.h5")
```

That is not a second code path. A file holds what the scanner sent -- the
header, then the acquisitions in acquisition order -- so
{meth}`pulserver.ReconPlugin.run` opens it and streams it to the plugin in this
process, through the same `startup` / `receive` / `recon` hooks an inline
reconstruction uses. No socket, no port, no server. No plugin implements it and
none overrides it.

Every plugin here is those three hooks and nothing else. `startup` lays out the
buffers the header's encoding spaces describe, `receive` places each
acquisition and routes the boundaries it closes to a named branch, and `recon`
holds the reconstruction of each branch over buffers that are already filled.
There is no local helper between them: what a step needs is a name in
{doc}`recon`, so a plugin reads as the composition it is.

## What happens on the way in

A noise scan measures the receiver, not the object, so it never reaches a
buffer: `receive` keeps it and whitens every readout that follows against it.

Coil compression is the same idea one step later, and where it can happen
depends on what the sequence acquired. A scan whose calibration is a separate
prescan — both EPI plugins — reads the array's principal channels off it when
that prescan closes and leaves the basis in `context.exam`, which is what
carries an artifact from the prescan's stream to the imaging one; every imaging
readout is then compressed as it arrives, and the imaging buffer is allocated
at the virtual channel count. A scan whose calibration is imaging data on the
imaging grid has no such moment — there is no basis before the first line — so
those plugins buffer the full array and compress once, on the way into the
solve.

The module also exposes `PLUGIN`, the configured
{class}`pulserver.ReconPlugin` the scanner drives, which is what to subclass or
re-instantiate with different settings.

## Cartesian

One buffer per encoding space, filled as the lines arrive. Which of three
reconstructions runs is read off the sampling mask rather than declared, by
{func}`pulserver.recon.cartesian_recon`: the coil-wise adjoint when everything
is there, POCS when the readout is truncated, CG-SENSE against NLINV maps when
phase encodes are missing. The plugins branch on the calibration boundary and
the imaging one; echoes are an axis, not a variant.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon

   cartesian2D_recon
   cartesian3D_recon
```

## Non-Cartesian

Density compensation, NLINV sensitivities calibrated from the samples inside
the calibration radius, and a CG-SENSE solve against the trajectory the
acquisitions carry -- which `pulserver` buffers beside the data.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon

   noncartesian2D_recon
   noncartesian3D_recon
   noncartesian_stack_recon
```

`noncartesian_stack_recon` is the factorised case: the partition axis is
Cartesian, so an inverse FFT along z turns the volume into independent planes,
each reconstructed in-plane.

## EPI

The one family that corrects before it places: a reversed EPI line has to be
flipped and phase corrected against the fit its navigator triplet produced
before it belongs on the grid, so `receive` does that as each line arrives and
places the corrected readout. The calibration prescan is a subsequence, so it
is an encoding space of its own and never touches the imaging grid.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon

   epi2D_recon
   epi3D_recon
```

## See also

{doc}`recon` is the toolbox these are written against -- physics operators,
solvers, calibration, the buffer layer. {doc}`app_sequence` is the other half
of the zoo.
