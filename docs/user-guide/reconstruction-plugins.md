# Reconstruction plugins

A reconstruction plugin is a file in the reconstruction proxy's `--plugins`
directory. It defines a {class}`~pulserver.recon.ReconPlugin` subclass and a
module-level `PLUGIN` instance of it. The proxy runs the plugin in a worker
process, one per series, over an MRD stream whose header and acquisitions carry
what the sequence states about its readouts (see
{doc}`../explanations/reconstruction`).

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
({meth}`~pulserver.recon.ReconPlugin.spawn`), so state set in the hooks belongs
to one series. `context.exam`, an {class}`~pulserver.recon.ExamCache`, is shared
by the series of one exam: a coil calibration computed in one series can be
stored there and read by the next. Under the proxy each series runs in a process
of its own, so a stored value reaches the next series pickled, through the
exam's directory: it comes back as a copy, without its `cleanup`, and a value
that cannot be pickled stays with its series.

`context.device` is the GPU the proxy gave the series, such as `"cuda:0"`, and
`None` on a host without one and offline; a reconstruction puts its tensors
there. A reconstruction may start processes of its own. A child started with
the `spawn` method imports what it runs by module name, which a plugin file
loaded from a path does not have, so the functions it runs come from importable
modules.

## Running a plugin offline

The same hooks run outside the proxy.
{meth}`~pulserver.recon.ReconPlugin.run` reconstructs an ISMRMRD HDF5 file in
the calling process:

```python
images = PLUGIN.run("scan.h5")
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

`pulserver.recon.handlers.simplefft` is a complete plugin: a two-dimensional
Cartesian FFT with a root-sum-of-squares coil combination, one image per slice.

## See also

* {doc}`../explanations/reconstruction` — enrichment, workers and slots.
* {doc}`../api/recon` — the plugin interface.
* {doc}`../api/mrd` — acquisitions, flags, counters and images.
