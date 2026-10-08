# Reconstruction proxy

How the reconstruction runtime places the readouts of a reconstruction unit in
its buffers, and how the proxy runs, schedules and isolates the
reconstructions of each series ({doc}`../../explanations/reconstruction`).

(reconstruction-placement)=
## Placement

The readouts of a Cartesian encoding space are placed as Gadgetron's
acquisition bucket places them.

Along the readout axis, a readout is placed by its echo. The samples from
`discard_pre` to `number_of_samples - discard_post` are copied so that the
sample at `center_sample` lies at sample `N // 2` of the buffer, `N` being the
number of samples of the buffer along the readout. `N` is the number of samples
of the unit's first readout when that is a full echo centred in its samples
(`center_sample == number_of_samples // 2`), and otherwise the matrix size of
the encoded space along the readout. Readouts that sample different parts of
the echo, such as a partial echo and a full one, are aligned by it, and the
samples a gadget added to complete an echo, which the gadget declares as
discarded, are not placed. `ReconBuffer.readout` is the first and last sample
placed. A readout whose placed samples fall outside the buffer is an error.

Along the phase-encoding and partition axes, a counter is placed at
`counter - center + extent // 2`, with `extent` the matrix size of the encoded
space along the axis and `center` the centre of the counter's encoding limit.
The k-space centre lies at `extent // 2` whichever counter the sequence gives
it, so undersampled, partial-Fourier and offset-numbered acquisitions are
placed on one grid. The partition axis is shifted only when the encoded space
has more than one partition. A counter that falls outside the grid once placed
is an error. The calibration buffer is placed by the same rule and holds the
lines its readouts cover; its `origin` is the position of its first line on the
grid of the imaging buffer.

A space with a trajectory has no echo or centre line to place by. Its readouts
are aligned to the end of the readout axis, its counters are the positions of
its views, and the k-space location of each sample is the trajectory the buffer
holds.

Two gadgets bring a Cartesian readout to the form the buffer places.
{class}`~pulserver.recon.AsymmetricEcho` zero-fills a partial echo to the full
echo that is symmetric about its `center_sample` and declares the zeros as
discarded. {class}`~pulserver.recon.RemoveReadoutOversampling` crops a full
echo to the readout field of view of the reconstruction, the ratio of the
encoded to the reconstructed field of view being the oversampling. A gadget
states the echo of the readout it returns by assigning `center_sample`,
`discard_pre` and `discard_post` of the acquisition it is given. That is a copy
of the acquisition: neither the stream's acquisition nor
`ReconData.acquisitions` changes.

## Workers

Each series is reconstructed in its own worker process, with the reconstruction
plugin the client's configuration names. The design names none, so the choice is
independent of the scanner sequence the series was played from, and one design
can be reconstructed by different plugins. A worker reconstructs one series and
exits, which releases the host and GPU memory the reconstruction allocated.
Importing a reconstruction engine takes seconds, so the proxy keeps spare
worker processes that have already imported it.

The series of one exam share a directory on the reconstruction computer. What
a series stores in its exam cache is written there, and a later series of the
exam reads it back, such as a coil calibration it need not compute again. The
exam is the one the header names, and the directory is the host's rather than
one proxy's: a reconstruction computer runs one proxy per acquisition, so the
series of an exam are reconstructed by different processes. It is deleted once
no series of that exam is being reconstructed anywhere on the host.

Three maps have names the series agree on, and a hook reaches them as
attributes of its context: `b0_map` is off-resonance in Hz, `b1_map` the
transmit field as a fraction of what was asked for, and `coil_sensitivities`
the receive sensitivity of each coil, a
{class}`~pulserver.recon.CoilSensitivities`. A calibration hook assigns what it
measured and a later hook reads it, with `None` meaning no series of the exam
has measured it yet.

```python
def recon(self, context, branch, data):
    maps = context.coil_sensitivities  # measured by an earlier series
    image, transmit = parallel_imaging_and_b1_fit(data, maps)
    context.b1_map = transmit          # read by a later one
    return recon.ReconResult(image)
```

A map is stored as measured, in the frame and on the grid of its series.
Nothing resamples, regrids or reslices it for a series acquired at another
position, orientation or matrix; {func}`~pulserver.recon.coil_maps` compares the
coil sensitivities with the unit that needs them, as {doc}`../../explanations/calibration` states.

Those three are the whole vocabulary of the exam's maps, so a misspelt name
raises rather than storing a map where nothing looks for it. An artifact a
plugin carries for itself goes in `context.exam` under a key of its own, as
{obj}`~pulserver.recon.NOISE_COVARIANCE` does for the whitening of a noise
series.

The number of series reconstructed concurrently is bounded by a number of
slots, derived from the available memory unless it is specified. On a host with
GPUs, which the proxy finds from `CUDA_VISIBLE_DEVICES` or `nvidia-smi` without
importing a GPU library, each slot holds one of them, one series per GPU unless
more are allowed, and the reconstruction finds its GPU in `context.device`. A
worker is an ordinary process, so a reconstruction may start processes of its
own. A series that
arrives when every slot is occupied is written to disk, enriched, as it
arrives, and replayed to a worker once a slot is released. The client remains
connected meanwhile, and the images are returned through its connection.

Images, DICOM datasets and text produced by the worker are relayed to the
client as they are produced. Closing either connection closes the other. Once
the client's stream ends, the proxy waits for the worker to close however long
the reconstruction takes, unless it was started with a reconstruction timeout,
past which the worker is terminated and the client told so.

