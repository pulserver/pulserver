# `pulserver.pypulseq`

A drop-in replacement for `pypulseq`. The complete upstream namespace is
re-exported unchanged, so this import covers the whole event layer and
`import pypulseq` alongside it is never necessary:

```python
import pulserver.pypulseq as pp

delay = pp.make_delay(1e-3)
seq = pp.Sequence(pp.Opts())
seq.add_block(delay)
```

Everything upstream documents is available here under the same name and with
the same behaviour — see the [PyPulseq API
reference](https://pypulseq.readthedocs.io/) for it. This page documents only
the **difference**: the objects Pulserver replaces, and the extension events
upstream has no equivalent for. Programmatically that set is
`pulserver.pypulseq.OVERRIDES`; the rest is `pulserver.pypulseq.UPSTREAM`.

This namespace is the event layer *only*. The factories that assemble events
into whole modules and scan loops — RF pulses, readouts, sampling plans, phase
schedules — live in {doc}`pulserver.design <design>`.

A few upstream names are deliberately withheld. `make_adiabatic_pulse` is not
re-exported, because Pulserver's inversion and preparation modules design
their adiabatic pulses internally. `compress_shape`, `decompress_shape` and
`convert` are not re-exported either: upstream exposes them as modules rather
than as part of its authoring vocabulary, and nothing a plugin writes should
need to reach the shape codec directly.

## Replaced objects

`Sequence` keeps its libraries, block table and file formats in C++, and reads
and writes Pulseq 1.5.2 including the rotation, shim and custom-label
extensions upstream cannot decode. `Opts` uses zero dead/ringdown times and
vendor-neutral 2 us RF/ADC and 10 us gradient/block rasters — scanner-specific
code should still pass explicit limits. `make_label` accepts user-defined
label strings the upstream set does not cover, and `get_supported_labels`
documents every label and what it means.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.Opts
   pulserver.pypulseq.Sequence
   pulserver.pypulseq.get_supported_labels
   pulserver.pypulseq.make_label
```

`Sequence` carries upstream's full method surface. `plot` is upstream's own,
drawn over the blocks a `time_range` actually touches — so looking at a tenth
of a second of a long protocol costs a tenth of a second of blocks, and the
axis still reads in time from the start of the sequence. Rotations are
resolved into the gradients and RF shims spread across the transmit channels
first, so what is drawn is what the scanner plays. Unstacked, its two panels
are laid out as one window of three rows by two columns rather than opened as
two.

The rest of the analysis half — `waveforms`, `check_timing`, `test_report`
and the others — is not ported yet and raises `NotImplementedError` with
upstream's signature. The scan structure those rest on belongs to the module
and scan-loop layer above this one.

## K-space and encoding counters

`Sequence.calculate_kspace()` returns upstream's five-tuple, `(k_traj_adc,
k_traj, t_excitation, t_refocusing, t_adc)`, and agrees with upstream PyPulseq
to about 1e-12 relative on the sequences where the two compute the same thing.
The arithmetic is the C library's — `csrc/src/pulseq/pulseq_ktraj.c`, the same
code the interpreter links — rather than a second implementation in Python. It
integrates each distinct gradient *shape* once instead of each block, so the
cost follows the number of distinct gradients rather than the length of the
scan, and it memoizes per readout on the block's event-id tuple, so a scan
that plays one readout a hundred thousand times pays for it once.

Two things upstream does not compute come with it. Rotation extensions are
resolved into the answer by default (`frame="logical"` leaves them out, which
is the frame `TransformFOV` works in), and **every readout's echo position is
derived** rather than assumed: the sample nearest the scan's own closest
approach to k = 0, which stays meaningful for a partial-Fourier or centre-out
readout that never crosses the origin. Where it cannot be derived it is
reported as `-1`, never as `num_samples // 2`.

`Sequence.auto_label()` reads the encoding counters back out of that
trajectory — `NOISE`, `SLC`, `REV`, `LIN`, `PAR`, `REP` — and writes them onto
the sequence as `SET` label extensions, together with the definitions they
imply (`kSpaceCenterLine`, `kSpaceCenterSample`, `SlicePositions`,
`SliceThickness`, `SliceGap`). It is the same set of labels as MATLAB Pulseq's
`autoLabel`, which upstream PyPulseq has no equivalent of, by a route that
never touches an ADC sample: the echo search is memoized per distinct readout
and everything after it reduces to one point per readout. A slice or slab
thickness is measured from the pulse's spectrum over the slice-select
amplitude; where there is no measurable spectrum — a hard pulse, which is not
slice-selective — nothing is reported rather than a rule of thumb. A sequence
whose readouts do not share a direction has no Cartesian counters and raises,
as MATLAB's does.

Three details are worth knowing before you rely on the counters.

`SLC` is a geometric index: slices are ranked by the position their
excitation's frequency offset puts them at, so `SlicePositions[SLC]` is where
slice `SLC` sits however the scan chose to visit them. Gaps, uneven spacing
and interleaved orderings all come out of the offsets themselves — nothing is
assumed about the prescription. Those offsets are read as authored, and
`TransformFOV` scaling rewrites the slice-select gradient without touching
them, so label first and transform second.

`kSpaceCenterSample` is quoted **after** `REV` has been honoured. A bipolar
train's two polarities put their echo one sample apart — 64 and 63 of 128,
being the same point in k reached from opposite ends — so a single number for
the scan is only true in one frame, and the useful frame is the one the
reconstructed data lives in.

Dimensions the trajectory cannot see are named, not guessed. Which echo of a
train, which frame of a time series and which saturation state all revisit the
same k-space position, so by default they are counted together as `REP`.
`repeat_dims=[("ECO", 2), ("REP", 10)]` — ordered fastest-varying first — says
what those visits were and splits the count between them; a declaration the
scan does not fit raises rather than wrapping around.

## Positioning the volume

`TransformFOV` moves, turns and resizes the imaging volume of a sequence that
has already been designed — a port of MATLAB Pulseq's `mr.TransformFOV`, which
upstream PyPulseq has no equivalent of. A translation becomes phase on the RF
and ADC, computed in C++ from absolute k; a rotation is attached as a
`ROTATIONS` extension rather than baked into new waveforms, so one waveform
still serves every orientation; a scale multiplies the amplitude a gradient
row carries and leaves its shape alone. `NOSCL`, `NOPOS` and `NOROT` exempt
the blocks that carry them, and a block range confines the whole thing to one
module. `Sequence.transform_fov()` is the shorthand for its commonest use.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.TransformFOV
```

## Event factories

Upstream's factories, wrapped. Each calls upstream unchanged — same
validation, same defaults, same bug fixes when upstream ships them — and
returns the event with its fields in slots rather than in a dictionary, which
is what a scan of a few million blocks needs.

They quack like the `SimpleNamespace` they replace: `rf.signal` is the complex
waveform at its real amplitude, `grad.waveform` the scaled samples. Underneath,
an RF pulse is a normalised magnitude and phase beside one scalar amplitude, so
`rf.amplitude *= 0.5` is a single write and leaves the registered shape valid —
a variable flip angle train is one magnitude shape at many amplitudes.

Setters do not re-validate: `make_*` checked the event when it built it, and a
loop moving a phase encode from one line to the next is not making a new claim
about the hardware.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_adc
   pulserver.pypulseq.make_arbitrary_grad
   pulserver.pypulseq.make_arbitrary_rf
   pulserver.pypulseq.make_block_pulse
   pulserver.pypulseq.make_delay
   pulserver.pypulseq.make_digital_output_pulse
   pulserver.pypulseq.make_extended_trapezoid
   pulserver.pypulseq.make_gauss_pulse
   pulserver.pypulseq.make_sinc_pulse
   pulserver.pypulseq.make_soft_delay
   pulserver.pypulseq.make_trapezoid
   pulserver.pypulseq.make_trigger
```

## Extension events

Pulseq extension objects with no upstream factory. Both are ordinary block
events: pass them to `seq.add_block` like any other.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_rf_shim
   pulserver.pypulseq.make_rotation
```

## Label constants

The Pulseq label set splits in two. **Counters** say where an acquisition
belongs and come from a scan loop's axes — {meth}`~pulserver.ScanLoop.label_state`
reports the values, {meth}`~pulserver.SequenceModule.set_state` emits them.
**Flags** say how a block is played or classified and come from
{meth}`~pulserver.SequenceModule.set_state`, which scopes them to the module
unless they are in `STICKY_FLAGS`.

`COUNTER_LABELS`, `FLAG_LABELS` and `STICKY_FLAGS` are that split as module
constants: the ten ISMRMRD `EncodingCounters` fields, everything else, and the
flags that outlive the module which set them.
