# Sequence representation

Pulseq is a portable description of a sequence, not a scanner instruction
stream. A `.seq` file contains a flat, time-ordered list of blocks. A block
refers to RF, gradient, ADC and extension tables; repeated waveform shapes may
share a table entry, but each use is still a block in the file. The format has
no mandatory TR or segment concept.

## PulSeg: reusable definitions plus instances

[PulSeg](https://github.com/HarmonizedMRI/pulseg) supplies the missing
intermediate representation. Conversion normalises each RF and gradient shape,
deduplicates it, and separates the reusable definition from the way it is
played:

| Layer | Contents | Scales with |
| --- | --- | --- |
| Base block | Normalised RF/gradient definitions and ADC | distinct shapes |
| Virtual segment | Contiguous, loadable list of base blocks | distinct instruction regions |
| Segment instance | amplitude, phase, frequency, rotation and delay values | scan length |
| Execution stream | ordered segment instances | scan length |

Amplitude scaling and rotation do not create a new definition. A Cartesian
phase-encode lobe, a radial spoke and a rotated spiral can therefore reuse one
base shape. Shapes that differ while retaining the same timing become shot
variants of one multishot definition. This is the compression that makes
scanner instruction memory depend on the waveform library rather than on the
matrix, slice count or number of averages.

## Segments are inferred, not authored

Pulserver parses Pulseq directly into this representation; it does not write a
separate PulSeg file. The C library finds segment boundaries only at legal
zero-gradient cuts and assigns a segment to the following excitation. If no
legal cut occurs, the remainder is one segment. Pure delays are split out and
can carry an instance-specific duration. The figures below are the
max-gradient-energy instances selected by that same C-library segmentation
query, not illustrative redraws.

![GRE inferred segment](assets/segments/gre_2d_segment_0.png)

![EPI inferred segment](assets/segments/epi_2d_segment_0.png)

![FSE inferred segment](assets/segments/fse_2d_segment_0.png)

`Sequence.segments` exposes the partition used by the scanner. Each segment
view resolves its maximum-energy instance, so plotting it answers the useful
question: which waveform does this reusable instruction region have to carry?

## Pulserver adds a safety-oriented execution view

Pulserver detects a canonical structural TR when one exists, keeps preparation
and cooldown distinct when they do not match it, and validates the allowed
per-instance variation. Gradient scale, RF phase/frequency, rotations and
adjustable pure delays are represented explicitly. Non-periodic RF amplitude
or shim patterns are rejected because RF-energy limits need a well-defined
window.

The result is cached in sections. Pulse generation reads only definitions and
shapes; scan-time playback additionally reads instance tables, rotations and
the execution stream. Reconstruction reads the independent `TRAJECTORY` and
`SEQDESC` sections. The representation is therefore not compression for its
own sake: it preserves the instance state required to prove safety before a
scan and to reconstruct the acquired data afterwards.

For the conformance mapping, see {doc}`../pulseg_conformance`; for safety and
reconstruction consumers, see {doc}`safety` and {doc}`reconstruction`.
