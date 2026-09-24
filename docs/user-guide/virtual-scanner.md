# Scanning a phantom without a scanner

The virtual scanner plays a stored design's IR cache, acquires an analytic
phantom along the trajectory the cache plays, and sends the series to a
reconstruction proxy as the scanner's reconstruction client does
({doc}`../explanations/virtual-scanner`). A scanner-sequence plugin and a
reconstruction plugin are exercised together, through the production design
calls and proxy.

## Generate a design

Generate the design as the PSD host process does, with the prescribed
field-of-view offset in the protocol block ({doc}`running`):

```bash
printf '[NimPulseqGUI Protocol]\nTE: 5000\nnx: 64\nny: 64\nfov_offset_x: 20.0\n[NimPulseqGUI Protocol End]\n' \
  | pulserver design generate --plugins sequences --plugin gre2d --limits limits.txt --store designs
```

The reply is `GENERATED <id>`.

## Start a proxy

Start a reconstruction proxy on the same store ({doc}`running`):

```bash
python -m pulserver.vre --store designs --port 9002 --plugins recon
```

## Acquire and send the series

```python
from pulserver import virtual

design = "..."  # the identifier the generation replied
sequence = f"designs/{design}/sequence.seq"
phantom = virtual.Phantom(
    [virtual.Ellipse((0.02, 0.0, 0.0), (0.08, 0.06))], coils=4
)
readouts = virtual.acquire(sequence, phantom)
received = virtual.send(("127.0.0.1", 9002), design, readouts, position_mm=(20.0, 0.0, 0.0))
```

The phantom is placed in the logical frame, about the isocentre; the object
above is centred on the prescribed field of view, so it appears at the centre
of the image. `received` holds the images, DICOM datasets and texts the
reconstruction returned; a text beginning `pulserver:` reports a refused or
failed series.

## See also

* {doc}`../explanations/virtual-scanner` — the stand-ins and the signal model.
* {doc}`../api/virtual` — the phantom, the acquisition and the client.
* {doc}`reconstruction-client` — the stream the client sends.
