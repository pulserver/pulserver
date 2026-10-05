# Reconstruction plugins

A reconstruction plugin is a file, `<name>.py` in a `--plugins` directory of the
reconstruction proxy, holding a {class}`~pulserver.recon.ReconPlugin` subclass
and a module-level `PLUGIN` instance. This one is `bart pics` on ESPIRiT maps:

```python
# recon/gre.py
import torch
import bartorch.tools as bt
from bartorch import apps, priors
from pulserver import recon

class Pics(recon.ReconPlugin):
    def recon(self, context, branch, data):
        kspace = torch.from_numpy(data.data.kspace)  # (coils, y, x)
        maps = bt.ecalib(kspace, maps=1)
        image = apps.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
        return recon.ReconResult(image.abs().numpy())

PLUGIN = Pics()
```

`data` is the {class}`~pulserver.recon.ReconData` of one reconstruction unit,
and `data.data` its k-space, `(coils, ..., readout)`, with the axes
`data.data.axes` names. Readouts are placed by their echo along the readout and
by their encoding counters along the encoded axes
({ref}`reconstruction-placement`); a sequence sets the counters with
{class}`pypulseqpp.sequences.Labels` or `pp.make_label` as in PyPulseq, and a
readout placed over another is warned about.
The proxy runs the plugin in a worker process, one per series, over an MRD
stream enriched from the sequence's design
({doc}`../explanations/reconstruction`). The client names the plugin of a series
in its config text ({doc}`reconstruction-client`), independently of the
scanner-sequence plugin the series was played from.

## Reconstruction units

A *reconstruction unit* is the set of readouts reconstructed together. It
holds the readouts of one branch and one encoding space that share their
slice, contrast, cardiac phase, repetition, set and average counters; the
`segment` and user counters do not separate units. A counter named in `axes`
becomes an axis of the unit's k-space instead, so that the echoes or averages
of an image are one unit.

A unit closes when the flag `triggers` names for its branch has arrived at every
position along its `axes`, and `recon` is called with it once. A unit is
released when `recon` returns: no reference to it remains, so a series holds
only the units open at once, and a plugin that needs the data of an earlier unit
keeps the parts it needs. A unit still open at the last readout of the
measurement (`LAST_IN_MEASUREMENT`) or at the end of the stream is reconstructed
then, under its own branch, in the order the units opened.

| Argument | Sets |
| --- | --- |
| `gadgets` | {class}`~pulserver.recon.Gadget` steps run over each readout before it is placed |
| `triggers` | `{branch: flag}`, the {class}`~pulserver.mrd.AcquisitionFlag` closing a unit of the branch, several combined with `\|`. The first branch is that of every readout `branch_for` does not route elsewhere. The default is `{"imaging": LAST_IN_MEASUREMENT}` |
| `axes` | the counters that are axes of a unit instead of separating units |
| `merge` | the counters that neither separate units nor are axes: a unit waits for its closing flag at each of their values, and their readouts fill one k-space, as the place in a train (`contrast`, `ECO`) of an echo train or an inversion shot does |
| `require_flags`, `reject_flags` | the flags an acquisition must all carry, and those any one of which excludes it |
| `buffered` | whether readouts are placed in `data.data` and `data.ref`; with `False` a plugin reads `data.acquisitions` and a unit closes at its first flag, whatever its `axes` |

`recon` receives the unit as a {class}`~pulserver.recon.ReconData`:

| Attribute | Holds |
| --- | --- |
| `data` | the k-space of the imaging readouts, a {class}`~pulserver.recon.ReconBuffer`, or `None` when the unit placed none |
| `ref` | the k-space of the parallel-imaging calibration readouts, laid out as `data` but holding only the lines of the phase-encode and partition axes its readouts cover, which start at `ref.origin` on the grid of `data`; `None` when the unit has none |
| `counters` | the unit's image counters not in `axes`, by MRD name |
| `waveforms` | the waveforms received since the previous unit closed |
| `acquisitions` | every readout the unit received, as it arrived |

A readout flagged `IS_PARALLEL_CALIBRATION` is placed in `ref` only, one flagged
`IS_PARALLEL_CALIBRATION_AND_IMAGING` in both, and one flagged
`IS_PHASECORR_DATA` in neither unless it is also a calibration readout.

## Hooks

The runtime calls three hooks over one stream:

| Hook | Called |
| --- | --- |
| {meth}`~pulserver.recon.ReconPlugin.startup` | once, before any acquisition |
| {meth}`~pulserver.recon.ReconPlugin.recon` | for each unit, as it closes |
| {meth}`~pulserver.recon.ReconPlugin.finish` | once, after the last unit |

Only `recon` has to be written. The runtime passes every acquisition to
{meth}`~pulserver.recon.ReconPlugin.receive`, which is not overridden: it
applies `require_flags` and `reject_flags`, runs the `gadgets` over the readout,
adds it to the unit of the branch
{meth}`~pulserver.recon.ReconPlugin.branch_for` names, and calls `recon` for each
unit that closes. `branch_for` returns no branch for a noise measurement, nor for
a navigator readout unless `triggers` declares a `"navigator"` branch; every
other readout belongs to the first branch `triggers` declares.

```pycon
>>> import numpy as np
>>> import pulserver.mrd as mrd
>>> import pulserver.recon as recon
>>> class DropNoise(recon.Gadget):
...     def __call__(self, acquisition, data):
...         noise = mrd.has_acquisition_flag(acquisition, "ACQ_IS_NOISE_MEASUREMENT")
...         return None if noise else data
>>> class RootSumOfSquares(recon.ReconPlugin):
...     def __init__(self):
...         super().__init__(
...             gadgets=[DropNoise()],
...             triggers={"imaging": mrd.AcquisitionFlag.LAST_IN_SLICE},
...         )
...     def recon(self, context, branch, data):
...         image = np.fft.fftshift(np.fft.ifft2(data.data.kspace))
...         return recon.ReconResult(np.sqrt(np.sum(np.abs(image) ** 2, axis=0)))
>>> PLUGIN = RootSumOfSquares()

```

A {class}`~pulserver.recon.ReconResult` is packaged as an MRD image, so a
plugin builds no image header. The field of view is that of the unit's encoding
space, the position, orientation and physiology time stamps are those of the
unit's reference acquisition, `data.data.reference`, and the time stamp is the
earliest of the unit's acquisitions. `reference=<acquisition>` takes the
geometry from another acquisition. The MRD image holds the array as returned,
`float32` or `complex64`, with no scale applied.

`dicom=True` sends the result as DICOM instead, and the integer pixels are made
then. The first floating-point image of a series fixes the rescale of its
DICOM pixels, and a result states its own in its attributes:

```python
return recon.ReconResult(
    image,
    dicom=True,
    attributes={"RescaleSlope": 1e-4, "RescaleIntercept": 0.0},
)
```

A value outside the range of the stored integers is clipped and counted; see
{doc}`../explanations/reconstruction`.

Each series runs on its own copy of `PLUGIN`
({meth}`~pulserver.recon.ReconPlugin.spawn`). `context.exam`, an
{class}`~pulserver.recon.ExamCache`, is shared by the series of one exam, such
as for a coil calibration; under the proxy a stored value reaches the next
series pickled, as a copy without its `cleanup`.

`context.device` is the GPU the proxy gave the series, such as `"cuda:0"`, or
`None`. A child process started with the `spawn` method runs functions from
importable modules, not from the plugin file.

## Readout placement

A readout of a Cartesian encoding space is placed so that its echo,
`center_sample`, lies at sample `N // 2` of the readout axis, the samples from
`discard_pre` to `number_of_samples - discard_post` being the ones copied. The
counter of a line is placed at `counter - center + extent // 2` along the
encoded axis, `center` being the counter of the k-space centre in the header's
encoding limits, so the counters of a partial-Fourier or undersampled scan
need not start at 0. A readout or a line that falls outside the buffer raises a
`ValueError`. `data.data.readout` is the first and last sample placed.

A plugin brings its readouts to the reconstruction matrix with two gadgets,
run in this order. {class}`~pulserver.recon.AsymmetricEcho` zero-fills a
partial echo to the full echo that is symmetric about its `center_sample`, and
declares the zeros as discarded so that they are not placed.
{class}`~pulserver.recon.RemoveReadoutOversampling` crops a full echo to the
readout field of view of the reconstruction, the ratio of the header's encoded
to reconstructed field of view along the readout, and needs bartorch (the
`coils` extra). A gadget states the echo of the readout it returns by assigning
`center_sample`, `discard_pre` and `discard_post` of the acquisition it is
given, which is a copy of the one the stream delivered:

```pycon
>>> import ismrmrd
>>> acquisition = ismrmrd.Acquisition()
>>> acquisition.resize(5, 2)
>>> acquisition.center_sample = 1
>>> full = recon.AsymmetricEcho()(acquisition, np.ones((2, 5), dtype=np.complex64))
>>> full.shape[-1], acquisition.discard_pre, acquisition.center_sample
(8, 3, 4)

```

## Non-Cartesian data

A buffer holds the trajectory of its samples beside the k-space, as the
acquisitions carry it: k along the sequence's x, y and z axes, in 1/m.
{meth}`~pulserver.recon.ReconBuffer.grid_trajectory` returns it in the grid
units and layout `bartorch.linop.NUFFT` takes, with the coils of the buffer's
`kspace` as a batch axis of the image:

```python
import torch
from bartorch.linop import NUFFT

buffer = data.data
nufft = NUFFT(
    torch.from_numpy(buffer.grid_trajectory()),
    image_shape=(buffer.coils, *buffer.image_shape),
)
coil_images = nufft.adjoint(torch.from_numpy(buffer.kspace))
```

The partitions of a stack of spokes or spirals are placed by their
`kspace_encode_step_2` counter and carry no kz: the partition axis is
transformed with an FFT before the in-plane NUFFT.

## Coil sensitivities, noise and compression

{func}`~pulserver.recon.coil_maps` returns the coil sensitivities of a unit as
a complex torch tensor `(coils, [z,] y, x)` on `context.device`. They are
estimated by the function the plugin passes, from the unit's calibration
k-space `data.ref`, or taken from the maps the stream or the exam holds
({doc}`../explanations/calibration` states the order and the conditions for
reuse). {class}`~pulserver.recon.Prewhiten` is a gadget and
{class}`~pulserver.recon.CoilCompression` a step called from `recon`. All three
need bartorch, which the `coils` extra installs.

```python
# recon/gre.py
import torch
from bartorch import apps, priors
from pulserver import mrd, recon

class Pics(recon.ReconPlugin):
    def __init__(self):
        super().__init__(
            gadgets=[
                recon.Prewhiten(),
                recon.AsymmetricEcho(),
                recon.RemoveReadoutOversampling(),
            ],
            triggers={"imaging": mrd.AcquisitionFlag.LAST_IN_SLICE},
        )
        self.compression = recon.CoilCompression(8)

    def recon(self, context, branch, data):
        data = self.compression(context, data)
        maps = recon.coil_maps(context, data, estimate=apps.nlinv_maps)
        if data.data is None:  # a unit of calibration readouts only
            return None
        kspace = torch.from_numpy(data.data.kspace).to(context.device)
        image = apps.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
        return recon.ReconResult(image.abs().cpu().numpy())

PLUGIN = Pics()
```

`Prewhiten` consumes the noise readouts of the stream and whitens every other
readout with them, or with the noise covariance a noise series left in the exam.
Without either it passes readouts unchanged and logs a warning once;
`Prewhiten(required=True)` raises instead. `CoilCompression(n)` keeps `n`
virtual channels, at most the channels of the first unit, with the basis of
that unit's calibration k-space (its imaging k-space where it has none), and
projects every later unit onto it. A unit is compressed before its maps are
requested, so that they are estimated in the basis of the data they are used
with.

A calibration unit, whose `data.data` is `None`, stores its maps for its slice,
and the imaging units of that slice take them from the stream. A series that is
only a calibration leaves its maps to the exam by assigning
`context.coil_sensitivities`, and a noise series leaves its whitening with
{meth}`~pulserver.recon.Prewhiten.publish`:

```python
class Calibration(recon.ReconPlugin):
    def recon(self, context, branch, data):
        recon.coil_maps(context, data, estimate=apps.nlinv_maps)
        context.coil_sensitivities = context.coil_maps[data.counters["slice"]]

class Noise(recon.ReconPlugin):
    def __init__(self):
        super().__init__(gadgets=[recon.Prewhiten()])

    def recon(self, context, branch, data):
        return None

    def finish(self, context):
        self.gadget(recon.Prewhiten).publish()
```

The exam holds one set of maps. A later series is given them only where the
coil labels, the whitening, the compression and the geometry equal its own, and
otherwise {class}`~pulserver.recon.MissingCalibration` is raised, naming the
first field that differs with both values. The failure text the client
receives carries the message. `required=False` returns `None` instead of
raising.

## Running a plugin offline

{meth}`~pulserver.recon.ReconPlugin.run` reconstructs an ISMRMRD HDF5 file in
the calling process. A file recorded as the scanner sends it, such as
`pulserver scan --mrd raw.h5` writes, is enriched from its design store first:

```python
images = PLUGIN.run("raw.h5", store="designs")
```

Calling the plugin on an {class}`~pulserver.mrd.AcquisitionBucket` runs
`startup`, passes every acquisition to `receive`, reconstructs the units still
open and runs `finish`, and returns the last output. A header only needs to
describe the encoded space and the receiver channels:

```pycon
>>> from types import SimpleNamespace
>>> matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=8, y=4, z=1))
>>> header = SimpleNamespace(
...     encoding=[SimpleNamespace(encodedSpace=matrix, reconSpace=matrix)],
...     acquisitionSystemInformation=SimpleNamespace(receiverChannels=2),
... )
>>> bucket = mrd.AcquisitionBucket.from_arrays(
...     np.ones((4, 2, 8), dtype=np.complex64),
...     labels={"kspace_encode_step_1": np.arange(4)},
... )
>>> result = PLUGIN(bucket, recon.ReconContext.offline(header))
>>> result.data.shape
(4, 8)

```

Six shipped plugins are complete reconstructions, returning the values of their
transform or solve unscaled. They are searched after every reconstruction
plugin directory, so a client can name one, such as `nufft`, without a file of
its own:

| Plugin | Reconstructs |
| --- | --- |
| `nufft` | The same images as `pics`, for a radial, spiral, PROPELLER or zero-echo-time trajectory or a stack of them, from readouts whitened with the stream's noise measurement (`Prewhiten`); each solved as `bart pics -t -R W` over `bart nlinv -t` sensitivities of the unit's samples near the k-space centre. The readouts of every segment of a unit, the blades of a PROPELLER scan or the shells of a zero-echo-time one, enter one solve; a trajectory that encodes the partition direction without a partition axis is solved as one volume. Needs the `coils` extra |
| `nufft_train` | `nufft`'s images from readouts that number their place in a train of excitations (`contrast`, `ECO`), merged so that the readouts of every place fill one image. Needs the `coils` extra |
| `pics` | One image per slice, contrast, cardiac phase, set and repetition, averages summed, from readouts placed by their encoding counters, whitened with the stream's noise measurement, completed to full echoes and cropped to the reconstruction field of view as they arrive (`Prewhiten`, `AsymmetricEcho`, `RemoveReadoutOversampling`); each image solved as `bart pics -R W` over the coil sensitivities of {func}`~pulserver.recon.coil_maps` (`bart nlinv` of the unit's calibration readouts, else the maps stored for its slice or the exam's, else a fit to the unit's own low-resolution centre), then completed by `bart homodyne` along a partial-Fourier axis. A unit of calibration readouts only makes no image and stores its maps. Needs the `coils` extra |
| `pics_train` | `pics`'s images from readouts that number their place in an echo train or an inversion shot (`contrast`, `ECO`), merged so that the readouts of every place fill one image. Needs the `coils` extra |
| `epi` | `pics`'s images from EPI readouts, each whitened, resampled off its ramps onto the matrix and corrected for its odd/even phase against the shot's navigator as it arrives (`bart` ramp operator, `estimate_epi_phase`). The phase-encode-reversed reference set is an image of its own, and with PyHySCO installed (GPL-3.0, not a dependency) each later image of its slice is corrected for susceptibility distortion against it. Needs the `coils` extra |
| `pmc` | `pics_train`'s images, and a pose from each three-plane navigator (`NAV` or `RTFEEDBACK` readouts): planes gridded with Pipe-Menon density on one thread, coils combined by root sum of squares, registered against the first navigator and filtered by an extended Kalman filter. The pose is published to the scan when its design sets `EnablePmc`. Needs the `coils` extra |

Every shipped plugin whitens its readouts with
{class}`~pulserver.recon.Prewhiten`, which consumes the noise readouts of the
stream and logs a warning once where the stream has none and the exam stores no
noise covariance. A volume is sent
as one image per partition. A plugin file that reexports one under another
name, `from pulserver.recon.handlers.pics import PLUGIN`, is the same
reconstruction.

## See also

* {doc}`../explanations/reconstruction` — enrichment, workers and slots.
* {doc}`../api/recon` — the plugin interface.
* {doc}`../api/mrd` — acquisitions, flags, counters and images.
