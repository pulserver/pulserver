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
    def recon(self, branch, context):
        kspace = torch.from_numpy(self.buffers[0].kspace)  # (coils, y, x)
        maps = bt.ecalib(kspace, maps=1)
        image = apps.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
        return recon.ReconResult(image.abs().numpy())

PLUGIN = Pics()
```

`self.buffers[0].kspace` is the first encoding space, `(coils, ..., readout)`,
with the axes `buffers[0].axes` names. Readouts are placed by their encoding
counters, which a sequence sets with `self.labels(LIN=line)` in its kernel or
`pp.make_label` as in PyPulseq; a readout placed over another is warned about.
The proxy runs the plugin in a worker process, one per series, over an MRD
stream enriched from the sequence's design
({doc}`../explanations/reconstruction`).

## Hooks

The runtime calls three hooks over one stream:

| Hook | Called |
| --- | --- |
| {meth}`~pulserver.recon.ReconPlugin.startup` | once, before any acquisition |
| {meth}`~pulserver.recon.ReconPlugin.receive` | for each accepted acquisition, as it arrives |
| {meth}`~pulserver.recon.ReconPlugin.recon` | for each branch `receive` routes |

Only `recon` has to be written. The default `receive` runs the `chain` of
{class}`~pulserver.recon.Gadget` steps over the readout, places it in
`self.buffers` by its encoding counters, and routes by `branches`: the first
{class}`~pulserver.mrd.AcquisitionFlag` the acquisition carries names the
branch to reconstruct.

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
...             chain=[DropNoise()],
...             branches={mrd.AcquisitionFlag.LAST_IN_SLICE: "imaging"},
...         )
...     def recon(self, branch, context):
...         image = np.fft.fftshift(np.fft.ifft2(self.buffers[0].kspace))
...         return recon.ReconResult(np.sqrt(np.sum(np.abs(image) ** 2, axis=0)))
>>> PLUGIN = RootSumOfSquares()

```

A {class}`~pulserver.recon.ReconResult` is packaged as an MRD image whose
geometry and timing come from a reference acquisition, so a plugin builds no
image header; `dicom=True` sends it as DICOM instead.

Each series runs on its own copy of `PLUGIN`
({meth}`~pulserver.recon.ReconPlugin.spawn`). `context.exam`, an
{class}`~pulserver.recon.ExamCache`, is shared by the series of one exam, such
as for a coil calibration; under the proxy a stored value reaches the next
series pickled, as a copy without its `cleanup`.

`context.device` is the GPU the proxy gave the series, such as `"cuda:0"`, or
`None`. A child process started with the `spawn` method runs functions from
importable modules, not from the plugin file.

## Non-Cartesian data

A buffer holds the trajectory of its samples beside the k-space, as the
acquisitions carry it: k along the sequence's x, y and z axes, in 1/m.
{meth}`~pulserver.recon.ReconBuffer.grid_trajectory` returns it in the grid
units and layout `bartorch.linop.NUFFT` takes, with the coils of the buffer's
`kspace` as a batch axis of the image:

```python
import torch
from bartorch.linop import NUFFT

buffer = self.buffers[0]
nufft = NUFFT(
    torch.from_numpy(buffer.grid_trajectory()),
    image_shape=(buffer.coils, *buffer.image_shape),
)
coil_images = nufft.adjoint(torch.from_numpy(buffer.kspace))
```

The partitions of a stack of spokes or spirals are placed by their
`kspace_encode_step_2` counter and carry no kz: the partition axis is
transformed with an FFT before the in-plane NUFFT.

## Running a plugin offline

{meth}`~pulserver.recon.ReconPlugin.run` reconstructs an ISMRMRD HDF5 file in
the calling process. A file recorded as the scanner sends it, such as
`pulserver scan --mrd raw.h5` writes, is enriched from its design store first:

```python
images = PLUGIN.run("raw.h5", store="designs")
```

Calling the plugin on an {class}`~pulserver.mrd.AcquisitionBucket` runs
`startup`, places every readout and reconstructs once. A header only needs to
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

Three built-in plugins are complete reconstructions, each image scaled to the
header's largest stored value:

| Module | Reconstructs |
| --- | --- |
| `pulserver.recon.handlers.simplefft` | A two-dimensional Cartesian FFT of the lines in arrival order, one image per slice |
| `pulserver.recon.handlers.cartesian` | A Cartesian FFT of the readouts placed by their encoding counters, partitions included; one image per slice, contrast, cardiac phase, set and repetition, averages summed |
| `pulserver.recon.handlers.nufft` | Each coil's least-squares fit of bartorch's NUFFT to its samples, with a Tikhonov term; the same images as `cartesian`, for a radial or spiral trajectory or a stack of them. Needs the `coils` extra |

Each combines its coils as a root sum of squares. A plugin file that reexports
one, `from pulserver.recon.handlers.cartesian import PLUGIN`, names it for a
sequence.

## See also

* {doc}`../explanations/reconstruction` — enrichment, workers and slots.
* {doc}`../api/recon` — the plugin interface.
* {doc}`../api/mrd` — acquisitions, flags, counters and images.
