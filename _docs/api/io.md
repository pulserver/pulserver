# `pulserver.io`

Pulseq serialisation helpers.

`read` is the inverse of `write`: it decodes a `.seq` file — Pulserver's
rotation, RF-shim and custom-label extensions included — back into a
{class}`pulserver.pypulseq.Sequence` whose inspection views are already built,
so it can be plotted or analysed straight away.

`read_esp_bands` and `read_asc_bands` parse a vendor lockout table — a GE ESP
table or a Siemens `.asc` file respectively — into the mechanical resonance
bands that {meth}`~pulserver.pypulseq.Sequence.calculate_gradient_spectrum`
overlays. Both report amplitude limits in mT/m, which is how the vendor tables
state them; `bands_to_hz_per_m` restates them in the Hz/m that method's
`bands=` argument takes. No vendor table of either kind ships with Pulserver;
supply the one for the system you target.

```{eval-rst}
.. autosummary::
   :toctree: generated/io
   :nosignatures:

   pulserver.io.write
   pulserver.io.read
   pulserver.io.read_esp_bands
   pulserver.io.read_asc_bands
   pulserver.io.bands_to_hz_per_m
```
