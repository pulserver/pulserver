# `pulserver.io`

Pulseq serialisation helpers.

`read` is the inverse of `write`: it decodes a `.seq` file — Pulserver's
rotation, RF-shim and custom-label extensions included — back into a
{class}`pulserver.pypulseq.Sequence` whose inspection views are already built,
so it can be plotted or analysed straight away.

`read_esp_bands` parses a vendor ESP lockout table into the mechanical
resonance bands that `Sequence.grad_spectrum` overlays. No ESP table ships
with Pulserver; supply the one for the system you target.

```{eval-rst}
.. autosummary::
   :toctree: generated/io
   :nosignatures:

   pulserver.io.write
   pulserver.io.read
   pulserver.io.read_esp_bands
```
