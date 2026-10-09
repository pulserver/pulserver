# Raw-data enrichment and routing

```{admonition} TL;DR
:class: tldr

- The proxy replaces the encoding counters, flags, encoding spaces and
  trajectory a client records with those the sequence states, before any
  reconstruction code reads the stream.
- The field-of-view offset is played as frequency and phase offsets; where the
  readout gradient varies across the sampling window, the proxy applies the
  remaining phase.
- A reconstruction unit is the readouts of one branch and encoding space that
  share their image-selecting counters; it closes on the flag its branch
  declares, and its images carry the values the plugin returned.
```

The scanner's reconstruction client streams each series as MRD: a configuration
text, the XML header and the acquisitions. The encoding counters, flags,
encoding spaces and trajectory a vendor client records describe the
interpreter's playout loop, not the Pulseq sequence: the counters of a
non-Cartesian or segmented acquisition, the navigator readouts and the k-space
location of each sample are defined by the sequence. The reconstruction proxy
therefore replaces them with the values the sequence states, before any
reconstruction code reads the stream.

The prescribed field-of-view offset is applied to the sequence when its IR is
built ({doc}`scanner-representation`), as the frequency and phase offsets the
scanner plays at the middle of each sampling window. Under a readout gradient
that holds one value across the window, these are the whole phase of the
offset. Under one that does not, as on a ramp-sampled or non-Cartesian
readout, the phase is not linear in time, and the proxy applies the remainder,

$$
\phi(t) = 2\pi\, \mathbf{c} \cdot \bigl(\mathbf{k}(t) - \mathbf{k}(t_c)
- \dot{\mathbf{k}}(t_c)\,(t - t_c)\bigr),
$$

for the offset $\mathbf{c}$ along the logical axes and the window centre
$t_c$, together with any phase modulation the sequence's ADC events store,
which the IR cache does not carry. The samples are otherwise left as received,
and the trajectory the proxy attaches is that of the sequence as designed, in
the logical frame.

## Enrichment

The header names the design the series was acquired with
({doc}`architecture`). The proxy reads that design's sequence chain and tabulates
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

## Trajectory

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

## Reconstruction units

A reconstruction takes the readouts of one image, or of several images
reconstructed together, such as the echoes of a slice. The stream states where
those readouts end only through the encoding counters and flags that
enrichment supplies. A *reconstruction unit* is the set of readouts of one
branch and one encoding space that share their slice, contrast, cardiac phase,
repetition, set and average counters; a branch is a kind of readout a plugin
reconstructs separately, such as `imaging` or `navigator`. The `segment` and
user counters name positions within a readout train and do not separate units.
A plugin names in its `axes` the counters it takes as axes of the unit's
k-space instead, such as the echoes of a multi-echo acquisition, and these do
not separate units either.

A unit allocates its imaging k-space when its first imaging readout arrives,
from the encoding space the header gives it, and places each readout as
described under {ref}`reconstruction-placement`. The flags of a readout select
its buffer as Gadgetron's acquisition bucket divides readouts: parallel-imaging
calibration readouts are placed in `ref`, readouts flagged as calibration and
imaging in `ref` and `data`, phase-correction readouts in neither, and any
other readout in `data`. The calibration k-space is allocated when the unit
closes, over the lines its readouts cover.

A unit closes when the flag its branch declares has arrived at every position
along its axes. Enrichment sets the last-in flag of a counter within each
combination of the encoding space and the other image-selecting counters, so
`LAST_IN_SLICE` arrives once per echo or average of a slice, and a plugin with
those counters as axes waits for all of them. Readouts of different units may
be interleaved: slices acquired line by line each close at their own flagged
readout.

A closed unit leaves the plugin before its reconstruction runs, and nothing in
the runtime holds it afterwards, so the memory a series holds is that of the
units open at once, not that of the series. A unit still open at the last
readout of the measurement or at the end of the stream is reconstructed then,
under its own branch, in the order the units opened.

## Images

A {class}`~pulserver.recon.ReconResult` becomes an MRD image whose header the
runtime builds from the unit that produced it. The values of the image are those
of the array the plugin returned.

### Header

The field of view is that of the reconstruction space of the unit's encoding
space, `(x, y, z)` in mm, so the images of a unit in the second encoding space
carry the second space's field of view. The matrix is the shape of the returned
array, which is {attr}`~pulserver.recon.ReconBuffer.image_shape`, the
reconstruction matrix of the unit's space, for a plugin that uses it.

The position, the orientation (`read_dir`, `phase_dir`, `slice_dir`), the
patient table position, the physiology time stamps and the user fields are those
of the unit's *reference acquisition*, the imaging acquisition placed nearest the
k-space centre, as Gadgetron selects it for an image header. The centre lies at
`extent // 2` along the phase-encoding axis and, where the encoded space has more
than one partition, the partition axis, as described under
{ref}`reconstruction-placement`; nearness is the Euclidean distance from it in
grid positions. Of acquisitions at the same distance, the first placed is the
reference. The acquisition time stamp of the image is the earliest of the unit's
imaging acquisitions.

A plugin reads the reference acquisition as
{attr}`~pulserver.recon.ReconBuffer.reference`, and takes the geometry from
another acquisition by passing it as `ReconResult.reference`.

### Values

The MRD image holds the values the plugin returned: `float32` for real
floating-point data, `complex64` for complex data, wider types being converted
to these, and integer data in its own type. No scale or normalisation is
applied, so the ratios between the images of a series, such as the decay across
the echoes of a multi-echo acquisition or the ratios between frames, are those
of the reconstruction. The units are the reconstruction's own.

A `(z, y, x)` result becomes one image per partition. Each image states the
smallest and largest finite value of the result, of its magnitude where it is
complex, in the meta attributes `ArrayMinimum` and `ArrayMaximum`.

## Reconstruction server

A proxy given a server to forward to reconstructs no series itself. It
enriches each series as it arrives and sends it on over TCP to that MRD server:
a reconstruction server ({class}`~pulserver.proxy.ReconServer`) on another
computer, or any server that reads the MRD streaming protocol. The server
receives a config file message naming the reconstruction plugin of the series,
or a name the proxy is configured with, then the enriched header and
acquisitions. What it returns is relayed to the client as a worker's output
is, and a reconstruction timeout closes the connection to it. With DICOM
conversion enabled, the proxy converts each image the server returns to DICOM
from the enriched header before relaying it.

A reconstruction server runs each series it receives with the plugin its
config names, in workers, slots and a queue of its own, and enriches nothing:
the series a proxy forwards arrive enriched.

An MRD message carries no length by which a reader can skip a message type it
does not know. A message the proxy has no reader for therefore ends what it
relays, and the client receives a text naming the message's type.

## See also

* {doc}`../user-guide/reconstruction-plugins` — writing a reconstruction.
* {doc}`../user-guide/reconstruction-client` — the MRD stream a reconstruction client sends.
* {doc}`calibration` — coil sensitivity maps, prewhitening and coil compression.
* {doc}`../api/proxy` — the proxy, the reconstruction server and the enrichment interface.
* {doc}`../api/recon` — the reconstruction plugin interface.
* {doc}`/generated/gallery/02-tours/03_fov_offset_enrichment` — enrichment and reconstruction of a simulated series.
