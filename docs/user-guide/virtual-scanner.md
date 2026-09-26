# Scanning a phantom without a scanner

The virtual scanner plays a stored design's IR cache, acquires an analytic
phantom along the trajectory the cache plays or simulates the cache on the
phantom's isochromats, and sends the series to a reconstruction proxy as the
scanner's reconstruction client does ({doc}`../explanations/virtual-scanner`). A scanner-sequence plugin and a
reconstruction plugin are exercised together, through the production design
calls and proxy.

## Generate a design

Generate the design as the interpreter host process does, with the prescribed
field-of-view offset in the protocol block ({doc}`running`):

```bash
printf '[NimPulseqGUI Protocol]\nTE: 5000\nnx: 64\nny: 64\nfov_offset_x: 20.0\n[NimPulseqGUI Protocol End]\n' \
  | pulserver design generate --plugins sequences --plugin gre2d --limits limits.txt --store designs
```

The reply is `GENERATED <id>`.

## Start a proxy

Start a reconstruction proxy on the same store ({doc}`running`):

```bash
python -m pulserver.proxy --store designs --port 9002 --plugins recon
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

The phantom lies in the physical frame, whose axes are the logical ones under
a prescription without a rotation; the object above is centred on the
prescribed field of view, so it appears at the centre of the image. `received`
holds the images, DICOM datasets and texts the reconstruction returned; a text
beginning `pulserver:` reports a refused or failed series.

## Scan water and fat

An ellipse of fat carries its chemical shift, and the acquisition the
magnet's field strength, at which the shift is resolved; `off_resonance_hz`
adds a frequency offset common to every spin:

```python
water = virtual.Ellipse((0.02, 0.0, 0.0), (0.06, 0.05))
fat = virtual.Ellipse((0.02, 0.07, 0.0), (0.04, 0.01), shift_ppm=-3.45)
readouts = virtual.acquire(
    sequence, virtual.Phantom([water, fat], coils=4), field_t=3.0
)
```

Fat is acquired at its chemical shift, and a fat saturation the design plays
leaves it the magnetization its pulse leaves at that frequency. Generate the
design under limits whose `B0` is the same field: the host resolves the ppm
offsets of the design's RF pulses at it when it builds the IR
({doc}`running`).

## Simulate relaxation and the RF pulses

The analytic acquisition acts on the phantom's density alone. For relaxation,
flip angles and slice profiles, sample the phantom as isochromats and play the
cache on them in pypulseqpp's Bloch simulation; the readouts are sent as
before:

```python
tissue = virtual.Phantom(
    [
        virtual.Ellipse((0.02, 0.0, 0.0), (0.08, 0.06), t1=0.9, t2=0.07),
        virtual.Ellipse((0.02, 0.02, 0.0), (0.02, 0.015), t1=1.8, t2=0.25),
    ],
    coils=4,
)
readouts = virtual.simulate(sequence, tissue.isochromats(1e-3))
```

The isochromats approximate the phantom in k-space below $1/(2\Delta)$ for a
grid spacing $\Delta$, here 1 mm, so choose a spacing several times finer than
the pixel; their number, and the time the simulation takes, grow as
$1/\Delta^2$. A phantom with a chemical shift is sampled at the magnet's field,
`tissue.isochromats(1e-3, field_t=3.0)`. The simulation starts from the
magnetization the isochromats hold, so a second scan of the same isochromats
continues the first: sample them again, or call their `reset`, to start from
equilibrium.

## Stream a scan in real time, with its sound

A scan played against its clock releases each readout once it is acquired, and
the client sends it on; the sound of its gradients can be written as it plays:

```python
import wave

import numpy as np

scan = virtual.Scan(sequence, tissue.isochromats(1e-3))
with wave.open("scan.wav", "wb") as audio:
    audio.setnchannels(2)
    audio.setsampwidth(2)
    audio.setframerate(int(virtual.SAMPLE_RATE))

    def acquired():
        for chunk in scan.chunks(speed=1.0):
            print(f"{chunk.stop:6.2f} s of {scan.duration:.2f} s")
            audio.writeframes(np.round(32767 * chunk.sound.T).astype("<i2").tobytes())
            yield from chunk.readouts

    received = virtual.send(("127.0.0.1", 9002), design, acquired())
```

At `speed=1.0` the scan lasts as long as it would on a scanner, or longer where
the simulation takes longer than the blocks it plays; without `speed`, the
spans come as fast as they are computed.

## Scan from the command line

`pulserver scan` runs a scan without a script. It imports a sequence file or
generates a design from a scanner-sequence plugin, as the design calls do,
plays its IR cache on a phantom in the Bloch simulation, and writes the series
to an ISMRMRD file, streams it to a reconstruction proxy, or both:

```bash
pulserver scan --seq sequence.seq --limits limits.txt \
  --orientation coronal --center 10 -5 3 --mrd raw.h5 --sound scan.wav

pulserver scan --plugins sequences --plugin gre2d --protocol protocol.txt \
  --limits limits.txt --store designs --recon 127.0.0.1:9002 --output images
```

The design call's reply, `IMPORTED <id>` or `GENERATED <id>`, or its `ERROR`
line, is written to standard output, and the scan clock to standard error.

- `--limits` is the `[Limits]` block of {doc}`running`; the phantom's chemical
  shifts are resolved at its `B0`. `--protocol` holds a protocol block, as
  `pulserver design generate` reads it on standard input; without it, the
  plugin's defaults are designed.
- `--orientation` names the rotation from logical to physical axes by the
  physical direction of each logical axis:

  | Orientation | Readout | Phase encoding | Slice selection |
  | --- | --- | --- | --- |
  | `axial` | +x | +y | +z |
  | `coronal` | +x | +z | −y |
  | `sagittal` | +y | +z | +x |

  `--rotation` gives the nine elements of another, row by row, and `--center`
  the field-of-view centre, in mm along the physical axes. They replace the
  field-of-view entries of the protocol block.
- `--phantom` is a JSON file of ellipses, each an object of the fields of
  {class}`~pulserver.virtual.Ellipse`, in its units:

  ```json
  {"ellipses": [
    {"centre": [0.0, 0.0, 0.0], "semi_axes": [0.08, 0.06], "t1": 0.9, "t2": 0.07},
    {"centre": [0.02, 0.02, 0.0], "semi_axes": [0.02, 0.015], "shift_ppm": -3.45}
  ]}
  ```

  Without it, the phantom is seven vials of water around one of fat, with T1
  from 0.3 s to 2.0 s and T2 from 0.04 s to 0.3 s. `--spacing`, in mm, and
  `--coils` set its isochromats and its receive coils.
- `--recon` streams the series to a reconstruction proxy, which looks the
  design up in its store: give that store as `--store`, or the proxy's design
  intake as `--push`. The images go to `images.h5` in `--output`, the DICOM
  datasets to the files they are named by, and a text beginning `pulserver:`
  ends the command with status 1.
- `--speed` plays the scan that many times as fast as a scanner; without it,
  the scan runs as fast as the simulation.

## Prescribe an orientation

Add the nine `fov_rotation_ij` entries to the protocol block the design is
generated from: element (i, j) of the rotation $R$ from logical to physical
axes, the identity's where absent ({doc}`../explanations/protocol`). The design
is checked in the physical frame $R$ gives, where logical axes played together
add on one physical axis, so derate the limits it is designed under for $R$ in
the `design_max_grad` and `design_max_slew` lines of the limits file
({doc}`running`). The acquisition and the client are given the same $R$, and
the client sends $R$ times the offset as the field-of-view centre:

```python
import numpy as np

rotation = np.array([[0.866025, -0.5, 0.0], [0.5, 0.866025, 0.0], [0.0, 0.0, 1.0]])
centre = rotation @ (0.02, 0.0, 0.0)
phantom = virtual.Phantom(
    [virtual.Ellipse((0.0, 0.0, 0.0), (0.08, 0.06))],
    coils=4,
    rotation=rotation,
    position=centre,
)
readouts = virtual.acquire(sequence, phantom, rotation=rotation)  # or virtual.simulate
received = virtual.send(
    ("127.0.0.1", 9002), design, readouts, position_mm=1e3 * centre, rotation=rotation
)
```

The phantom, posed at the prescribed centre with its axes turned by $R$,
appears at the centre of the image as it would at the isocentre without a
rotation, and the images carry the columns of $R$ as their read, phase and
slice directions.

## See also

* {doc}`../explanations/virtual-scanner` — the stand-ins and the signal model.
* {doc}`../api/virtual` — the phantom, the acquisition, the Bloch simulation, the scan clock and the client.
* {doc}`reconstruction-client` — the stream the client sends.
