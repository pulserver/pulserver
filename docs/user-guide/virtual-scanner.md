# Scanning a phantom without a scanner

The virtual scanner plays a stored design's IR cache on a phantom's tissue
with its Fourier engine, and sends the series to a reconstruction proxy as the
scanner's reconstruction client does ({doc}`../explanations/virtual-scanner`). A scanner-sequence plugin and a
reconstruction plugin are exercised together, through the production design
calls and proxy.

A browser build of the virtual scanner's console runs without an installation
at <https://pulserver.github.io/MaRGE/>.

## Generate a design

Generate the design as the interpreter host process does, with the prescribed
field-of-view offset in the protocol block ({doc}`running`):

```bash
printf '[Protocol]\nTE: 5000\nnx: 64\nny: 64\nfov_offset_x: 20.0\n[Protocol End]\n' \
  | pulserver design generate --plugins sequences --plugin gre2d --limits limits.txt --store designs
```

The reply is `GENERATED <id>`.

## Start a proxy

Start a reconstruction proxy on the same store ({doc}`running`):

```bash
pulserver proxy --store designs --port 9002 --plugins recon
```

## Acquire and send the series

```python
from pulserver import virtual

design = "..."  # the identifier the generation replied
sequence = f"designs/{design}/sequence.seq"
phantom = virtual.Phantom(
    [
        virtual.Ellipse((0.02, 0.0, 0.0), (0.08, 0.06), t1=0.9, t2=0.07),
        virtual.Ellipse((0.02, 0.02, 0.0), (0.02, 0.015), t1=1.8, t2=0.25),
    ],
    coils=4,
)
readouts = virtual.simulate(sequence, phantom.tissue(1e-3))
received = virtual.send(
    ("127.0.0.1", 9002),
    design,
    readouts,
    position_mm=(20.0, 0.0, 0.0),
    config="pics",
)
```

{meth}`~pulserver.virtual.Phantom.tissue` samples the phantom as cubes of
uniform magnetization, here 1 mm wide, each with its proton density, T1, T2,
T2′ and frequency. {func}`~pulserver.virtual.simulate` plays the cache on
them with the Fourier engine ({class}`~pulserver.virtual.FourierPlayer`,
{doc}`../explanations/virtual-scanner`): each RF pulse the cache plays turns
a cube through the flip angle its profile gives at the field the cube sees,
the gradients' moments dephase the cubes, they relax between the events, and
the images are encoded at the resolution the trajectory reaches, whatever the
spacing. The
spacing sets how closely the cubes follow the phantom's edges; the number of
cubes grows as $1/\Delta^2$ for a spacing $\Delta$, or as $1/\Delta^3$ for a
phantom that fills a volume. The engine runs on a CUDA device where there is
one, and on the CPU otherwise; `device` names another. It returns one
`(coils, samples)` array per readout, in play order, demodulated as the
playout demodulates. It spoils by the moment the gradients wind across a
voxel between pulses and readouts. `motion` takes a
{class}`~pulserver.virtual.RigidMotion` and `girf` a
{class}`~pulserver.virtual.Girf`; a tissue whose `diffusion` is set loses
$e^{-bD}$ at each readout.

`config` names the reconstruction plugin, here the shipped `pics`, as the
config text of a reconstruction client does ({doc}`reconstruction-client`); the
proxy refuses a series whose config names none.

The phantom lies in the physical frame, whose axes are the logical ones under
a prescription without a rotation; the object above is centred on the
prescribed field of view, so it appears at the centre of the image.
`position_mm` is written as each acquisition's `position`; `send` writes no
`fov_offset_mm` user parameter, so the proxy corrects the samples for the
offset the design was converted at and for no further displacement. `received`
holds the images, DICOM datasets and texts the reconstruction returned; a text
beginning `pulserver:` reports a refused or failed series.

## Scan water and fat

An ellipse of fat carries its chemical shift, and the tissue the magnet's
field strength, at which the shift is resolved; `off_resonance_hz` adds a
frequency offset common to every cube:

```python
water = virtual.Ellipse((0.02, 0.0, 0.0), (0.06, 0.05), t1=1.2, t2=0.08)
fat = virtual.Ellipse(
    (0.02, 0.07, 0.0), (0.04, 0.01), shift_ppm=-3.45, t1=0.35, t2=0.07
)
tissue = virtual.Phantom([water, fat], coils=4).tissue(1e-3, field_t=3.0)
readouts = virtual.simulate(sequence, tissue)
```

A phantom with a chemical shift is refused without `field_t`. Fat precesses
at its chemical shift, and a fat saturation the design plays turns it through
the flip angle its pulse plays at that frequency. Generate the
design under limits whose `B0` is the same field: the host resolves the ppm
offsets of the design's RF pulses at it when it builds the IR
({doc}`running`).

## Stream a scan in real time, with its sound

A scan played against its clock releases each readout once it is acquired, and
the client sends it on; the sound of its gradients can be written as it plays:

```python
import wave

import numpy as np

scan = virtual.Scan(sequence, tissue)
with wave.open("scan.wav", "wb") as audio:
    audio.setnchannels(2)
    audio.setsampwidth(2)
    audio.setframerate(int(virtual.SAMPLE_RATE))

    def acquired():
        for chunk in scan.chunks(speed=1.0):
            print(f"{chunk.stop:6.2f} s of {scan.duration:.2f} s")
            audio.writeframes(np.round(32767 * chunk.sound.T).astype("<i2").tobytes())
            yield from chunk.readouts

    received = virtual.send(("127.0.0.1", 9002), design, acquired(), config="pics")
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
plays its IR cache on a phantom's tissue with the Fourier engine, and writes
the series to an ISMRMRD file, streams it to a reconstruction proxy, or both:

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
  `--spacing`, in mm, 1 mm by default, is the spacing at which the phantom is
  sampled as tissue, and `--coils` sets its receive coils; `--coil` scans it
  with one of the scanner's coils instead, `body`, `body/head48` or
  `head8/head32`, named `transmit/receive`, whose sensitivities bartorch
  samples from BART's coil models, which the `coils` extra installs
  (`pip install 'pulserver[coils]'`). A phantom file gives each ellipse its
  `t2_prime`, in s. BrainWeb's grey and white matter and CSF diffuse.
- `--nod DEGREES PERIOD` turns the subject about the physical x axis
  sinusoidally, `--drift X Y Z` translates it in mm/min, and `--jumps RATE MM
  DEGREES` moves it suddenly at random, `RATE` times a second on average.
  `--girf DELAY_US TIME_CONSTANT_US` plays the gradients through a delay and a
  first-order low pass. The console takes the same options.
- `--recon` streams the series to a reconstruction proxy, which looks the
  design up in its store: give that store as `--store`, or the proxy's design
  intake as `--push`. The images go to `images.h5` in `--output`, the DICOM
  datasets to the files they are named by, and a text beginning `pulserver:`
  ends the command with status 1. `--reconstruction` names the reconstruction
  plugin the proxy reconstructs the series with, independently of the sequence
  the design was generated from. Without it, a shipped `--plugin` is
  reconstructed with the shipped reconstruction paired with it
  ({ref}`shipped-sequences`), and a design of another plugin or imported with
  `--seq` needs it.
- `--speed` plays the scan that many times as fast as a scanner, once the
  simulation is far enough ahead, writing the time left before it is to
  standard error; without it, the scan runs as fast as the simulation.
- `--device` names the torch device the Fourier engine runs on, such as
  `cpu` or `cuda`; a CUDA device where there is one without it.

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
readouts = virtual.simulate(sequence, phantom.tissue(1e-3), rotation=rotation)
received = virtual.send(
    ("127.0.0.1", 9002),
    design,
    readouts,
    position_mm=1e3 * centre,
    rotation=rotation,
    config="pics",
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

`--plugins` and `--recon-plugins` may be repeated, the first directory holding a
plugin supplying it. With `--recon-plugins DIR` in place of `--recon`, each scan
is reconstructed in the console's own process by the plugins of `DIR`, checked
and enriched as the
proxy checks and enriches a series ({class}`~pulserver.proxy.LocalReconstruction`),
and no proxy runs. `--origin` names an origin whose browser pages the console
serves, and may be repeated; without it, pages from every origin are served,
and a client that sends no `Origin`, which a browser always sends, is served
either way. `--fields DIR` takes the coils from the field maps in `DIR`,
`<coil>.npz` for each of `body`, `head8`, `head32` and `head48`, and
`<coil>_vops.npz` for the two that transmit, as mariepy writes them for
BrainWeb's head ({doc}`../explanations/virtual-scanner`). Every exam is then on
BrainWeb, and every design is made under the VOP limits of the exam's transmit
coil; maps solved at another frequency than the Larmor frequency of the limits'
`B0` are refused.

A scan is simulated by the Fourier engine on the phantom's tissue, sampled
`--spacing` apart, in mm, 1 mm by default, once for each exam as it starts.
`--device` acts as it does for `pulserver scan`.

pulserver's image runs such a console by default, with BrainWeb's normal brain
and the field maps solved in it, `--fields=/console/fields`, so that every exam
is on BrainWeb and downloads nothing, and with bartorch for the non-Cartesian
reconstructions. It serves the pages of `https://pulserver.github.io`,
`http://localhost:8000` and `http://127.0.0.1:8000` on port 8765 of this
computer, and plays each scan at the scanner's speed, `--speed=1`:

```bash
docker run -d --restart unless-stopped --name pulserver \
  -p 127.0.0.1:8765:8765 ghcr.io/pulserver/pulserver
```

Options after the image's name are added to its console's: an option of one
value takes the value given last, and `--plugins`, `--recon-plugins` and
`--origin` add to the image's. A console whose exams are sampled 2 mm apart:

```bash
docker run -d --restart unless-stopped --name pulserver \
  -p 127.0.0.1:8765:8765 ghcr.io/pulserver/pulserver \
  --spacing 2
```

Its sequences are the ones listed under {ref}`shipped-sequences`,
reconstructed by the shipped plugins of {doc}`reconstruction-plugins` unless a
scan names another, and each protocol starts at values the image's limits
play.

Directories mounted at `/console/user/plugins` and `/console/user/recon` add
sequences and reconstructions to those pulserver ships. They are searched
first, so a file there takes the place of the shipped file of that name. A
link must resolve inside the container, as a relative link within the mounted
directory does. A file added or changed is used from the next call on:

```bash
docker run -d --restart unless-stopped --name pulserver \
  -p 127.0.0.1:8765:8765 \
  -v "$PWD/sequences:/console/user/plugins:ro" \
  -v "$PWD/recon:/console/user/recon:ro" \
  ghcr.io/pulserver/pulserver
```

Each request is a JSON object carrying `call` and an `id` that every reply to
it repeats:

| `call` | Fields | Replies |
| --- | --- | --- |
| `plugins` | | `plugins`: the scanner-sequence plugin names |
| `recons` | | `recons`: the reconstruction plugin names, the shipped ones and those of `--recon-plugins` |
| `coils` | | `coils`: each coil's `name` and its `transmit` and `receive` channels |
| `version` | | `image`: the digest of the image the console runs in, named by `PULSERVER_IMAGE` as `<repository>@<digest>`; `latest`: the digest the registry publishes for that repository's `latest` tag; each `null` where it is not known |
| `list`, `validate`, `generate`, `import` | `plugin`, `block` | `status` and `reply`, as `pulserver design` answers; `design` for a generated or imported design |
| `exam` | `subject`, `coil` | `localizer`: the axial, coronal and sagittal images of the subject's phantom, as base64 DICOM files |
| `scan` | `design`, `rotation` (nine elements), `centre_mm`, `sound`, `recon` | at a speed, `preparing` about twice a second until the clock starts, the wall-clock time left in s or `null` before there is an estimate; `clock` and `duration` after each span played, with the span's `sound` when asked, as base64 of 16-bit little-endian stereo samples at `rate` Hz; `dicom` and `name` for each image the reconstruction returns, `text`, then `done` with the status |
| `cancel` | | stops the scan in progress |

The `recon` field of a `scan` names the reconstruction plugin of the scan,
independently of the plugin the design was generated from. Without it, a design
of a shipped sequence is reconstructed with the shipped reconstruction paired
with it ({ref}`shipped-sequences`), and any other design, such as an imported
one, is refused with an `error` naming the missing reconstruction. A console
that reconstructs through a proxy lists the shipped reconstructions only, and
uses a name the proxy holds all the same.

{meth}`Console.answer <pulserver.virtual.Console.answer>` answers one request
in process with the same replies, without the `id`, for a console that reaches
pulserver by another route than a WebSocket, such as the messages of a Web
Worker running pulserver beside a browser page.

A subject named `brainweb` is BrainWeb's normal brain, and so is every subject
of a console with field maps; any other is the vials. An exam is scanned in the
coil it names, or in the one it had, `--coil` at first; a head coil of BART's
models needs the `coils` extra. The scans of an exam play on one tissue, each
from equilibrium, sampled while the exam's localizer is viewed. A console
loads BrainWeb once, and computes the field its head adds while the first
exam's localizer is viewed. The localizer is drawn from the
phantom's proton density, so an exam can be planned before any scan. MaRGE is
such a console when `MARGE_PULSERVER` holds the address, `ws://127.0.0.1:8765`
here: its sequences are then the plugins, its subject names the phantom, its RF
coil names the coil, and it opens each exam on the localizer. The
browser build of MaRGE in [pulserver/MaRGE](https://github.com/pulserver/MaRGE)
runs it in a browser tab, opened with `?console=ws://127.0.0.1:8765`, and plays
each scan's sound as it streams; it is published at
<https://pulserver.github.io/MaRGE/>.

## See also

* {doc}`../explanations/virtual-scanner` — the stand-ins and the signal model.
* {doc}`../api/virtual` — the phantom, the Fourier engine, the scan clock and the client.
* {doc}`reconstruction-client` — the stream the client sends.
