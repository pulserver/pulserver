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

The module also exposes `PLUGIN`, the configured
{class}`pulserver.ReconPlugin` the scanner drives, which is what to subclass or
re-instantiate with different settings.

## Cartesian

One buffer per encoding space, filled as the lines arrive. Which of three
reconstructions runs is read off the sampling mask rather than declared: the
coil-wise adjoint when everything is there, POCS when the readout is truncated,
CG-SENSE against NLINV maps when phase encodes are missing. Echoes are an axis,
not a variant.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon

   cartesian2D_recon
   cartesian3D_recon
```

## Non-Cartesian

Density compensation, adjoint-derived sensitivities, and a CG-SENSE solve
against the trajectory the acquisitions carry -- which `pulserver` buffers
beside the data.

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

The one family that sorts its own stream: an EPI line cannot be placed until
the acquisitions have been partitioned by flag into navigator, reverse-polarity
reference and imaging, and until the odd/even ramp has been fitted from its
slice's navigator triplet.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/app_recon

   epi2D_recon
   epi3D_recon
```

`examples/recon/mrd_epi_preprocessing.py` sits beside these as a worked
adapter for handing the EPI groups to an outside reconstruction backend. It is
not part of the zoo and is not importable from `pulserver.app`.

## See also

{doc}`recon` is the toolbox these are written against -- physics operators,
solvers, calibration, the buffer layer. {doc}`app_sequence` is the other half
of the zoo.
