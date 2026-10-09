# Raw data: what pulserver adds to MRD

```{admonition} TL;DR
:class: tldr

- The proxy rewrites each MRD stream from the design the header names, before any reconstruction code reads it.
- Every readout gets its counters from your labels, its flags, echo sample and, where k moves, its trajectory.
- The samples are left as received, except for the field-of-view phase the scanner's offsets cannot apply.
```

A scanner plays the IR cache and records samples, so the MRD stream its client
sends says every readout is line 0 of slice 0. The sequence knows which line,
slice, echo and navigator each readout is, and where each sample lies in
k-space. The proxy writes that into the stream, which stays ISMRMRD v1, so your
recon plugin receives what a Gadgetron reconstruction receives from a product
sequence. Lesson 4 of the {doc}`Course <../examples/course>` prints the stream
before and after.

## What pulserver does

:::{container} capabilities

- **Sets each readout's encoding counters from the Pulseq labels in force at it.**

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *counters are the labels each readout sees* (`test_enrich.py`); *labels are the values in force at each readout* (`test_mrd_sequence.py`).
- **Sets the first-in and last-in flags from the counters, and the flags your labels set, such as navigator data.**

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *a slice closes once per echo*; *only the last readout of a chain ends the measurement* (`test_enrich.py`).
- **Describes each encoding space from the sequence: full-echo readout, 2D slices as slices, navigators apart.**

  Code: {mod}`pulserver.proxy`. Tests: *the header describes each encoding space*; *the encoded readout is the full echo with its oversampling*; *the slices of a 2d design reach the reconstruction as slices not partitions*; *navigator readouts form their own encoding space* (`test_enrich.py`).
- **Centres the encoding limits on the k-space centre the sequence defines.**

  Code: {mod}`pulserver.proxy`. Tests: *the limits centre is the centre line the sequence defines*; *the limits centre is the middle of the lines when the sequence defines none* (`test_enrich.py`).
- **Gives each readout the echo sample the design puts at the k-space centre.**

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *the echo index is the design centre sample*; *reversed lines meet the echo at the mirrored sample*; *a spiral out echo is its first sample* (`test_mrd_sequence.py`).
- **Attaches the trajectory to non-Cartesian and ramp-sampled readouts only.**

  Code: `Sequence.adc_kspace` (pypulseqpp). Tests: *a non cartesian readout carries its absolute k*; *a ramp sampled cartesian readout carries its k as one axis*; *a cartesian readout sampled on its flat top carries no trajectory* (`test_enrich.py`).
- **Integrates k a run of readouts at a time, so memory does not grow with the scan.**

  Code: {class}`~pulserver.proxy.SequenceTable`. Tests: *k integrated a run at a time is k integrated from the first block*; *a spin echo train keeps the k its refocusing pulses reverse*; *a table keeps the k of two runs at most* (`test_mrd_sequence.py`).
- **Applies the field-of-view phase the scanner's offsets cannot, and any later position change in the header.**

  Code: {mod}`pulserver.proxy`. Tests: *a curving readout carries the phase its receiver cannot apply*; *a steady readout needs no curve*; *a stated position away from the converted shift adds its whole phase* (`test_enrich.py`).
- **Writes TR, TE, TI and flip angles into the header, measuring the ones the sequence does not define.**

  Code: {mod}`pulserver.mrd`. Tests: *te and tr a sequence does not define are the ones pypulseqpp measures*; *the header lists every flip angle the sequence plays* (`test_mrd_sequence.py`, `test_enrich.py`).
- **Refuses a stream that does not match the design.**

  Code: {mod}`pulserver.proxy`. Tests: *a gap in the scan counters stops the series unreconstructed* (`test_proxy.py`); *an acquisition of the wrong length is refused*; *a slice counter beyond the slice positions is refused* (`test_enrich.py`).

:::

## What the scanner and your plugin do

- **The scanner's reconstruction client**
  design identifier and the name of the recon plugin, every readout in play
  order including dummies, noise scans and navigators
  ({doc}`../user-guide/reconstruction-client`).
- **Your recon plugin**
  counters, as described in {doc}`reconstruction`.

## How it works

### The field-of-view offset

The scanner plays the offset as frequency and phase offsets at the middle of
each window ({doc}`scanner-representation`), which is the whole phase only
under a constant readout gradient. On ramp-sampled and non-Cartesian readouts
the proxy applies the remainder, together with any ADC phase modulation the
file stores:

$$
\phi(t) = 2\pi\, \mathbf{c} \cdot \bigl(\mathbf{k}(t) - \mathbf{k}(t_c)
- \dot{\mathbf{k}}(t_c)\,(t - t_c)\bigr),
$$

with $\mathbf{c}$ the offset along the logical axes and $t_c$ the window centre.

### Enrichment and trajectory

The proxy tabulates the design's readouts in play order
({class}`~pulserver.proxy.SequenceTable`) and matches each acquisition to its
row by position in the stream. k is the integral of the gradients with block
rotations applied, in 1/m, integrated one run of readouts at a time from the
last excitation, which resets it. A Cartesian readout on its flat top is placed
by its counters and `center_sample` alone. A plugin reads the trajectory in grid
units through {meth}`~pulserver.recon.ReconBuffer.grid_trajectory`. The full
rules are in {doc}`../developer-guide/internals/reconstruction-proxy`.

## See it run

- {doc}`../generated/gallery/01-course/04_reconstruction_plugin`: the stream of
  `gre2d` before and after enrichment.
- {doc}`/generated/gallery/02-tours/03_fov_offset_enrichment`: enrichment and
  reconstruction of a simulated series at a field-of-view offset.
- {doc}`../user-guide/reconstruction-client`: the messages and header a client
  sends.
- {doc}`../api/proxy`: the enrichment interface.
