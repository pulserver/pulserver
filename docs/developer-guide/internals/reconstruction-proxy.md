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

## Enrichment

The header names the design the series was acquired with
({doc}`../../explanations/architecture`). The proxy reads that design's sequence chain and tabulates
its readouts in play order ({class}`~pulserver.proxy.SequenceTable`). The table
is applied to the stream as follows.

- The header receives one encoding space for the imaging readouts of each
  subsequence, and one for its navigator readouts when it has any. The
  reconstruction space of each carries the matrix size and field of view the
  sequence defines. The encoded space of a Cartesian space carries them with
  the readout widened to the full echo: its matrix size along the readout is
  the number of samples of the echo that is symmetric about the readouts'
  `center_sample`, readout oversampling included, and its field of view along
  the readout is the reconstructed one times the ratio of the two matrix
  sizes. The ratio of the encoded to the reconstructed field of view along the
  readout is the readout oversampling. A space with a trajectory keeps the
  matrix size and field of view the sequence defines.
- A subsequence that writes no `PAR` and either defines `SlicePositions` or
  writes more than one `SLC` excites a stack of 2D slices, each an image of its
  own. Its `Matrix` and `FOV` along z describe the stack, so each of its
  spaces is one sample deep along z, over the `SliceThickness` the sequence
  defines, or the stack's field of view shared among its slices where it
  defines none.
- The encoding limits of each space are the minimum and maximum of the
  counters its readouts reach. The `center` of `kspace_encoding_step_1` and
  `kspace_encoding_step_2` is the counter of the k-space centre the sequence
  defines (`kSpaceCenterLine`, `kSpaceCenterPartition`); where it defines
  none, and for every other counter, the centre is the minimum plus half the
  number of positions between the minimum and the maximum, rounded down.
- The header's sequence parameters are the TR, TE, TI and flip angles the
  sequence defines. The TR and TE it does not define are measured by
  pypulseqpp's `Sequence.test_report_dict`: TE from the excitation before the
  closest approach to the k-space centre, and TR between the excitations
  around it. The flip angles it does not define are the distinct values of
  `Sequence.rf_flip_angles`.
- Each acquisition is matched to a table row by its position in the stream and
  receives the encoding counters, the MRD flags, the dwell time and the
  encoding space reference, and the k-space trajectory when the k-space
  location changes across the readout, except on a Cartesian readout sampled
  on the flat top of its readout gradient alone. In a stack of slices, its
  position, the field-of-view centre the client sends, is moved along
  `slice_dir` by the `SlicePositions` entry its `SLC` indexes, to the centre of
  its slice. When the client numbers its
  acquisitions, each `scan_counter` must follow the previous one by one; a gap
  or a repeat stops the series before it is reconstructed, since every later
  row would be shifted.

The flags are those the sequence's labels set, such as `IS_NAVIGATION_DATA`,
and the first-and-last flags (`FIRST_IN_SLICE`, `LAST_IN_SLICE` and the
others), derived from the counters. A reconstruction plugin selects the data a
reconstruction runs on by these flags.

## Trajectory integration

The k-space location of each sample is the integral of the sequence's
gradients, with block rotations applied, in 1/m. The table does not hold it for
the whole scan: it is integrated a run of consecutive readouts at a time, by
pypulseqpp's `Sequence.adc_kspace`, from the last excitation before the run.
An excitation resets k to zero at the pulse centre, so the samples that follow
it do not depend on the gradients played before it, and a run integrated from
there gives the k-space locations of the scan integrated from its first block.
A refocusing pulse reverses k rather than resetting it and does not begin an
integration, nor does an excitation in a block that also holds a readout.

An acquisition carries as its trajectory every axis its encoding space varies
along, up to the last, whether or not its own k moves along it: the line of a
PROPELLER blade that is played unrotated keeps its phase encoding in ky. The
partitions of a stack of spokes or spirals are Cartesian along z and placed by
their `kspace_encode_step_2` counter, so their kz is not part of the
trajectory. A reconstruction takes the trajectory in grid units, k times the
reconstructed field of view, through
{meth}`~pulserver.recon.ReconBuffer.grid_trajectory`.

A Cartesian readout carries a trajectory only when it samples the ramps of its
readout gradient. Sampled on the flat top alone, its samples are equally
spaced in k and placed by the encoding counters and `center_sample`, and an
encoding space whose readouts are all of that kind carries no trajectory.

Tabulating a design integrates every range once, for the echo sample of each
readout and the k-space axes its trajectory spans, which decide the header's
trajectory type. An acquisition's trajectory is integrated again when it is
enriched, from the range holding it, and the table keeps the ranges it
integrated last. The k-space locations the proxy holds therefore do not grow
with the length of the scan.
