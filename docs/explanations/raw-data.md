# Raw data: what pulserver adds to MRD

```{admonition} TL;DR
:class: tldr

- The scanner streams its raw data as ISMRMRD (MRD), with counters and flags
  that describe the interpreter's playout loop rather than your sequence. The
  reconstruction proxy replaces them with what the sequence states before any
  reconstruction code reads the stream.
- It fills in the encoding spaces, matrix and field of view of the header, and
  for every readout its counters from your Pulseq labels, its flags, its echo
  sample and, where k moves across the readout, its trajectory.
- The samples are left as received, except for the part of the field-of-view
  phase that the scanner's frequency and phase offsets cannot apply.
```

A scanner knows little about a Pulseq sequence. It plays the IR cache and
records samples, so the MRD stream its reconstruction client sends says every
readout is line 0 of slice 0. The sequence knows which line, slice, echo and
navigator each readout is, and where in k-space each sample lies. The
reconstruction proxy reads the design the header names and writes that
information into the stream, so your recon plugin receives MRD as a Gadgetron
reconstruction would from a product sequence. The stream stays ISMRMRD v1.

Lesson 4 of the {doc}`Course <../examples/course>` prints the stream before and
after this step.

## What pulserver does

:::{container} capabilities

- **Sets each readout's encoding counters from the Pulseq labels in force at it.**

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *counters are the labels each readout sees* (`test_enrich.py`); *labels are the values in force at each readout* (`test_mrd_sequence.py`).
- **Sets the first-in and last-in flags from the counters, and the flags your labels set, such as navigator data.** A slice closes once per echo, and only the last readout of a chain ends the measurement.

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *a slice closes once per echo*; *only the last readout of a chain ends the measurement* (`test_enrich.py`).
- **Describes each encoding space in the header from the sequence's definitions.** The encoded readout is the full echo with its oversampling; a stack of 2D slices reaches the reconstruction as slices, not partitions; navigator readouts get an encoding space of their own.

  Code: {mod}`pulserver.proxy`. Tests: *the header describes each encoding space*; *the encoded readout is the full echo with its oversampling*; *the slices of a 2d design reach the reconstruction as slices not partitions*; *navigator readouts form their own encoding space* (`test_enrich.py`).
- **Centres the encoding limits on the k-space centre the sequence defines.**

  Code: {mod}`pulserver.proxy`. Tests: *the limits centre is the centre line the sequence defines*; *the limits centre is the middle of the lines when the sequence defines none* (`test_enrich.py`).
- **Gives each readout the echo sample the design puts at the k-space centre.** Reversed lines are counted from their end, and a spiral-out echo is its first sample.

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *the echo index is the design centre sample*; *reversed lines meet the echo at the mirrored sample*; *a spiral out echo is its first sample* (`test_mrd_sequence.py`).
- **Attaches the trajectory where k moves across the readout: non-Cartesian and ramp-sampled readouts.** A Cartesian readout on its flat top carries none.

  Code: `Sequence.adc_kspace` (pypulseqpp). Tests: *a non cartesian readout carries its absolute k*; *a ramp sampled cartesian readout carries its k as one axis*; *a cartesian readout sampled on its flat top carries no trajectory* (`test_enrich.py`).
- **Integrates k a run of readouts at a time, from the last excitation, so memory does not grow with the scan.** Refocusing pulses reverse k rather than reset it.

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *k integrated a run at a time is k integrated from the first block*; *a spin echo train keeps the k its refocusing pulses reverse*; *a table keeps the k of two runs at most* (`test_mrd_sequence.py`).
- **Applies the field-of-view phase the scanner's offsets cannot, on curving readouts, and a later change of position stated in the header.**

  Code: {mod}`pulserver.proxy`. Tests: *a curving readout carries the phase its receiver cannot apply*; *a steady readout needs no curve*; *a stated position away from the converted shift adds its whole phase* (`test_enrich.py`).
- **Writes TR, TE, TI and flip angles into the header, measuring the ones the sequence does not define.**

  Code: {mod}`pulserver.mrd`. Tests: *te and tr a sequence does not define are the ones pypulseqpp measures*; *the header lists every flip angle the sequence plays* (`test_mrd_sequence.py`, `test_enrich.py`).
- **Refuses a stream that does not match the design: a gap in the scan counters, an acquisition of the wrong length, or a slice counter beyond the slice positions.**

  Code: {mod}`pulserver.proxy`. Tests: *a gap in the scan counters stops the series unreconstructed* (`test_proxy.py`); *an acquisition of the wrong length is refused*; *a slice counter beyond the slice positions is refused* (`test_enrich.py`).

:::

## What the scanner and your plugin do

- **The scanner's reconstruction client** sends what the scanner measured, the
  design identifier and the name of the recon plugin, every readout in play
  order including dummies, noise scans and navigators
  ({doc}`../user-guide/reconstruction-client`).
- **Your recon plugin** sorts the enriched readouts into k-space by their
  counters, as described in {doc}`reconstruction`.

## How it works

### The field-of-view offset

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

### Enrichment

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

### Trajectory

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

## See it run

- {doc}`../generated/gallery/01-course/04_reconstruction_plugin`: the stream of
  `gre2d` before and after enrichment.
- {doc}`/generated/gallery/02-tours/03_fov_offset_enrichment`: enrichment and
  reconstruction of a simulated series at a field-of-view offset.
- {doc}`../user-guide/reconstruction-client`: the messages and header a client
  sends.
- {doc}`../api/proxy`: the enrichment interface.
