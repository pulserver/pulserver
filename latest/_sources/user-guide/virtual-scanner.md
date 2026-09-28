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

The spans are simulated in a thread of their own, ahead of the clock. At
`speed=1.0` the clock starts once the simulation, at the rate it has run so
far, stays ahead of it to the end of the scan, which then lasts as long as it
would on a scanner; `preparing`, when given, receives the time left before the
clock starts. A span simulated after its end on the clock holds the clock until
it is. Without `speed`, the spans come as fast as they are simulated.

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

  `--phantom brainweb` scans {class}`~pulserver.virtual.BrainWeb`, BrainWeb's
  normal brain, downloaded on first use by the `brainweb` extra
  (`pip install 'pulserver[brainweb]'`); its `--spacing` is a whole number of
  millimetres. Without `--phantom`, the phantom is seven vials of water around
  one of fat, with T1 from 0.3 s to 2.0 s and T2 from 0.04 s to 0.3 s.
  `--spacing`, in mm, and `--coils` set its isochromats and its receive coils;
  `--coil` scans it with one of the scanner's coils instead, `body`,
  `body/head48` or `head8/head32`, named `transmit/receive`, whose
  sensitivities bartorch samples from BART's coil models, which the `coils`
  extra installs (`pip install 'pulserver[coils]'`). The sensitivities at
  every isochromat are written to a temporary file mapped into memory; where
  the temporary directory is a memory file system, set `TMPDIR` to one on
  disk.
- `--recon` streams the series to a reconstruction proxy, which looks the
  design up in its store: give that store as `--store`, or the proxy's design
  intake as `--push`. The images go to `images.h5` in `--output`, the DICOM
  datasets to the files they are named by, and a text beginning `pulserver:`
  ends the command with status 1.
- `--speed` plays the scan that many times as fast as a scanner, once the
  simulation is far enough ahead, writing the time left before it is to
  standard error; without it, the scan runs as fast as the simulation.

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

## Serve a scanner console

`pulserver console` answers a scanner console over a WebSocket: the design
calls on the interpreter's text blocks, an exam on a subject, and scans of the
designs it generates, played on the subject's phantom:

```bash
pulserver console --plugins sequences --limits limits.txt --store designs \
  --recon 127.0.0.1:9002 --port 8765
```

With `--recon-plugins DIR` in place of `--recon`, each scan is reconstructed in
the console's own process by the plugins of `DIR`, checked and enriched as the
proxy checks and enriches a series ({class}`~pulserver.proxy.LocalReconstruction`),
and no proxy runs. `--origin` names an origin whose browser pages the console
serves, and may be repeated; without it, pages from every origin are served,
and a client that sends no `Origin`, which a browser always sends, is served
either way.

pulserver's image runs such a console by default, with bartorch for the head
coils, BrainWeb's normal brain, so that an exam on `brainweb` downloads
nothing, and the `gre2d` plugin reconstructed by the built-in Cartesian FFT,
`pulserver.recon.handlers.simplefft`. It serves the pages of
`https://pulserver.github.io`, `http://localhost:8000` and
`http://127.0.0.1:8000` on port 8765 of this computer, and plays each scan at
the scanner's speed, `--speed=1`:

```bash
docker run -d --restart unless-stopped --name pulserver \
  -p 127.0.0.1:8765:8765 ghcr.io/pulserver/pulserver
```

Each request is a JSON object carrying `call` and an `id` that every reply to
it repeats:

| `call` | Fields | Replies |
| --- | --- | --- |
| `plugins` | | `plugins`: the plugin names |
| `coils` | | `coils`: each coil's `name` and its `transmit` and `receive` channels |
| `list`, `validate`, `generate`, `import` | `plugin`, `block` | `status` and `reply`, as `pulserver design` answers; `design` for a generated or imported design |
| `exam` | `subject`, `coil` | `localizer`: the axial, coronal and sagittal images of the subject's phantom, as base64 DICOM files |
| `scan` | `design`, `rotation` (nine elements), `centre_mm`, `sound` | at a speed, `preparing` about twice a second until the clock starts, the wall-clock time left in s or `null` before there is an estimate; `clock` and `duration` after each span played, with the span's `sound` when asked, as base64 of 16-bit little-endian stereo samples at `rate` Hz; `dicom` and `name` for each image the reconstruction returns, `text`, then `done` with the status |
| `cancel` | | stops the scan in progress |

{meth}`Console.answer <pulserver.virtual.Console.answer>` answers one request
in process with the same replies, without the `id`, for a console that reaches
pulserver by another route than a WebSocket, such as the messages of a Web
Worker running pulserver beside a browser page.

A subject named `brainweb` is BrainWeb's normal brain; any other is the vials.
An exam is scanned in the coil it names, or in the one it had, `--coil` at
first; a head coil needs the `coils` extra. The exam's scans play on one set of
isochromats, each from equilibrium, so that a scan after the first neither
builds them nor computes again the pulses an earlier scan played. The localizer is drawn from the
phantom's proton density, so an exam can be planned before any scan. MaRGE is
such a console when `MARGE_PULSERVER` holds the address, `ws://127.0.0.1:8765`
here: its sequences are then the plugins, its subject names the phantom, its RF
coil names the coil, and it opens each exam on the localizer. The
browser build of MaRGE in [pulserver/MaRGE](https://github.com/pulserver/MaRGE)
runs it in a browser tab, opened with `?console=ws://127.0.0.1:8765`, and plays
each scan's sound as it streams.

## See also

* {doc}`../explanations/virtual-scanner` — the stand-ins and the signal model.
* {doc}`../api/virtual` — the phantom, the acquisition, the Bloch simulation, the scan clock and the client.
* {doc}`reconstruction-client` — the stream the client sends.
