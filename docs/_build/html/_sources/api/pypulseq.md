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

Three upstream names are deliberately withheld: `compress_shape`,
`decompress_shape` and `convert`. Upstream exposes them as modules rather than
as part of its authoring vocabulary, and nothing a plugin writes should need to
reach the shape codec directly. Reaching for one — or for a MATLAB name with no
counterpart here, such as `addCustomLabel` — raises an `AttributeError` that
gives the reason and points at what to use instead.

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

`check_timing()` is upstream's own checker run over a window of blocks, and
`check_gradient_continuity()` is the C safety core's — the check `add_block`
skips because its cost would fall on every block. Both run from `write()` by
default; a server-mode plugin turns them off, since the scanner repeats them
at predownload. Continuity is judged **after** each block's rotation is
applied, in the physical frame the amplifiers slew in.

Two methods still refuse on purpose: `install` (Siemens scanner transfer) and
`sound` (acoustic preview). Each raises with its reason.

## K-space and encoding counters

`Sequence.calculate_kspace()` returns upstream's five-tuple, `(k_traj_adc,
k_traj, t_excitation, t_refocusing, t_adc)`, and is a drop-in for upstream's:
all five elements match it to 1e-9 or better, and the extra parameters are
keyword-only after upstream's own.

`k_traj_adc` — the one a reconstruction needs — comes from the C library,
`csrc/src/pulseq/pulseq_ktraj.c`, the same code the interpreter links, rather
than from a second implementation in Python. It integrates each distinct
gradient *shape* once instead of each block, so the cost follows the number of
distinct gradients rather than the length of the scan, and it memoizes per
readout on the block's event-id tuple, so a scan that plays one readout a
hundred thousand times pays for it once. Agreement with upstream is 2e-13 on a
GRE and 2e-12 on an EPI.

`k_traj`, the dense trajectory, is **computed by upstream**. It is a picture of
the sequence rather than an input to a reconstruction, and being able to hand
it to code written against upstream matters more than computing it quickly.
The C core's own answer is on the gradient breakpoint grid — the same curve in
five to ten times fewer points — and is available through `_kspace()` together
with its time base, which upstream's tuple has nowhere to put. Pass
`dense=False` to skip it. Sequences upstream cannot read, meaning anything
carrying rotation or RF-shim extensions, raise rather than returning a
logical-frame `k_traj` beside a physical-frame `k_traj_adc`.

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

Every `autoLabel` parameter is accepted, under the Python spelling of its name
and in its own order — `time_range`, `use_labels`, `use_aux`, `skip_apply`,
`mirror_fourier`, `reflect`, `reorder`, `sort_slices`, `no_plots` — with
Pulserver's additions (`trajectory_delay`, `repeat_dims`, `skip`) after them
rather than mixed in. `use_labels` and `use_aux` skip detection and apply a
result computed elsewhere, which is how one detection serves several variants
of a sequence, or how a counter gets corrected by hand without recomputing the
rest; combining them with `reflect`, `reorder` or `mirror_fourier` is refused,
since those only affect detection and would look like they had done something.

Two defaults differ from MATLAB's, both deliberately. `no_plots` is `True`
here because `autoLabel` draws diagnostic figures and nothing here draws any —
passing `False` raises rather than quietly producing nothing. `sort_slices`
defaults to `"ascending"` where MATLAB uses `"acquisition"`; the reason is the
next paragraph. `"acquisition"` reproduces MATLAB's numbering exactly, and
`"descending"` is what `autoLabel`'s own notes recommend for a Siemens
interpreter.

Three details are worth knowing before you rely on the counters.

`SLC` is a geometric index: slices are ranked by the position their
excitation's frequency offset puts them at, so `SlicePositions[SLC]` is where
slice `SLC` sits however the scan chose to visit them. Gaps, uneven spacing
and interleaved orderings all come out of the offsets themselves — nothing is
assumed about the prescription. An interleaved acquisition (0, 2, 4, 1, 3)
hands the reconstruction a shuffled stack under arrival order and an ordered
one under this, which is why it is the default. `SlicePositions[SLC]` holds
under all three sortings, so a reconstruction reading the pair together is
right whichever you pick; only the index a slice is given changes, and so does
nothing else — the slice gap is a spacing between adjacent positions and comes
out the same under every one. Those offsets are read as authored, and
`TransformFOV` scaling rewrites the slice-select gradient without touching
them, so label first and transform second.

`mirror_fourier` reverses the readout, phase and partition directions together
— for a reconstruction that inverse-transforms where this assumes a forward
transform — and is not the same as `reflect=[0, 1, 2]` in the one respect that
matters: it leaves the slice positions and slice-select gradients alone, so
slice ordering is unaffected.

`kSpaceCenterSample` is quoted **after** `REV` has been honoured. A bipolar
train's two polarities put their echo one sample apart — 64 and 63 of 128,
being the same point in k reached from opposite ends — so a single number for
the scan is only true in one frame, and the useful frame is the one the
reconstructed data lives in.

Dimensions the trajectory cannot see are named, not guessed. Which echo of a
train, which frame of a time series and which saturation state all revisit the
same k-space position, so by default they are counted together as `REP`.
`repeat_dims=["REP", "ECO"]` — outermost loop first — says what those visits
were and splits the count between them.

Only the names are needed. How large each dimension is, is written in the
acquisition order and read back from it: a dimension nested inside the k-space
loop brings a position back after a short gap, one outside it only after a
whole pass. Passing `("ECO", 2)` in place of a name pins a size, and it is
then checked against what was read rather than believed.

The reading is narrow on purpose, because a wrong split puts two different
acquisitions in one slot and every label around it still looks ordinary. It
requires the repeats to form a rectangle — every k-space position visited the
same number of times, in the same pattern. Repeats that are ragged, as an
EPI's navigators are, raise instead. A single name never does: it takes the
whole count, which is `REP` under a name that means something.

The other way round works too: label the axes only your design loop knows as
you build, then run one `auto_label()` pass to fill in the geometric ones
around them. Labels it does not derive — `ECO`, `SET`, `AVG`, anything custom
— survive that pass untouched, because the extension chain is rebuilt keeping
every link that is not one of its own. `REP` is the exception, since it *is*
derived by default: pass `skip=["REP"]` to hand it back, or your own
separation of contrasts and frames is overwritten by a bare count of revisits.

## Repetitions

A `.seq` describes one pass. Playing it several times is normally left to the
interpreter, which takes the count from outside the file — on a GE scanner,
the `opnex` knob — and uses the `ONCE` flag to work out what belongs to a
single pass: `1` plays on the first repetition only, `2` on the last only, `0`
on all of them.

`Sequence.expand_repeats(n)` does that arithmetic here instead and writes the
answer down. Afterwards the block table *is* the scan: every repetition is
present in the order it plays, and nothing downstream has to be told how many
times to read it. Call it like `remove_duplicates` — once, on a finished
sequence, before writing.

Only the block table grows. A repetition plays the *same* events, so every
library is untouched and deduplication has nothing left to find: a
100 000-block scan repeated three times is 300 000 rows of six integers and a
duration, and not one extra gradient.

What it buys is that the file stops depending on a number that is not in it. A
sequence written this way plays identically under any interpreter, including
one that has never heard of `ONCE`, and the average index becomes a label a
reconstruction can sort by rather than something the interpreter synthesises
on the way past. `IgnoreAverages 1` goes into `[DEFINITIONS]` to say the
expansion has already happened, so a console-side count cannot multiply it a
second time.

The repetition index is stamped as `AVG`, because the repetition an
interpreter adds is a signal average — the same acquisition, sampled again.
`REP` is the frame counter of a dynamic series and means something else. Only
where it changes: labels are sticky, so one extension per repetition is the
whole cost, and it lands on the first block that repetition actually plays,
which for every repetition after the first is the first block past the
preparation. A sequence that already writes `AVG` is **refused** rather than
overwritten — pass `label=` to choose another counter, or `label=""` for none.

The flags are read per block rather than as sections, so a design whose
preparation is not one contiguous run at the front expands the way it would
have been played. `strip_once=False` keeps them: against Pulserver's own
interpreter that costs nothing at playback — an expanded scan is `1` at the
front, `0` through the middle and `2` at the end, the shape a single-pass
sequence already has — and it preserves the preparation/cooldown split
`pulseg` hands to its TR descriptor.

## Positioning the volume

`TransformFOV` moves, turns and resizes the imaging volume of a sequence that
has already been designed — a port of MATLAB Pulseq's `mr.TransformFOV`, which
upstream PyPulseq has no equivalent of. A translation becomes phase on the RF
and ADC, computed in C++ from absolute k; a rotation is attached as a
`ROTATIONS` extension rather than baked into new waveforms, so one waveform
still serves every orientation; a scale multiplies the amplitude a gradient
row carries and leaves its shape alone. `NOSCL`, `NOPOS` and `NOROT` exempt
the blocks that carry them, and a block range confines the whole thing to one
module. `apply_to_sequence(seq, in_place=True)` transforms a finished sequence
where it stands; without `in_place` it hands back a transformed copy, which is
what MATLAB's `applyToSeq` does.

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
   pulserver.pypulseq.make_adiabatic_pulse
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

## Simulation

`sim_rf` is MATLAB Pulseq's `mr.simRf`, which upstream PyPulseq has no
equivalent of: it rotates the magnetisation through the pulse across a range
of off-resonance, and reports what the pulse does starting from `+z`, `+x` and
`+y`. The three together are what separates a refocusing pulse from one that
merely inverts — `ref_eff` combines the last two. `bloch` is the integrator
underneath, exposed for a caller simulating something the pulse alone does not
describe, such as a two-dimensional profile under its own gradients.

`calc_rf_bandwidth` measures the width of the pulse's main lobe between its
flanks, and follows a pulse that carries a `freq_offset`. It is the low-flip
approximation; `sim_rf` is what to use for a profile at a real flip angle.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.sim_rf
   pulserver.pypulseq.bloch
   pulserver.pypulseq.calc_rf_bandwidth
```

## Base factories

Stated in what a sequence knows rather than in gradient area or dwell time.
`make_crusher` is `make_extended_trapezoid_area` with the area replaced by
cycles of dephasing across a voxel, so it can also ride a gradient that is
already running. `make_phase_encoding` builds the full-amplitude template a
scan loop scales per view — resolution alone fixes it, since field of view and
matrix cancel — and `make_phase_blip` states the step between echoes in k-space
cells, so `steps=R` *is* an acceleration of R.

`calc_adc_timing` returns a dwell on the ADC raster whose
`num_samples * dwell` is a whole number of gradient rasters, which is what lets
the ADC sit exactly over the readout flat top.

`make_slr_pulse` designs its FIR filter in-package, so SigPy is not a
dependency; it is also exported as `make_sigpy_pulse`. `make_spsp_pulse` builds
a spectral-spatial pulse on an alternating gradient, and `make_sms_pulse` turns
any RF event into equispaced bands.

`traj_to_grad` **diverges from upstream by default**: `time_optimal=True` takes
`k` as geometry only and re-parameterises it in time with MRArbGrad, so the
limits hold everywhere by construction and the returned length is the solver's.
Pass `time_optimal=False` for upstream's finite difference of an
already-rastered trajectory.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.calc_adc_timing
   pulserver.pypulseq.make_crusher
   pulserver.pypulseq.make_phase_blip
   pulserver.pypulseq.make_phase_encoding
   pulserver.pypulseq.make_sigpy_pulse
   pulserver.pypulseq.make_slr_pulse
   pulserver.pypulseq.make_sms_pulse
   pulserver.pypulseq.make_spsp_pulse
   pulserver.pypulseq.traj_to_grad
```

## Sampling masks, orderings and angles

Plain arrays: a boolean mask over the phase-encode grid, a list of shots each
holding the view indices it acquires in order, or one angle per spoke. They are
here rather than in {doc}`pulserver.design <design>` because they are data a
scan loop indexes with, not modules; the loop that walks them is the plugin's.

The four mask modes are `make_uniform_mask`, `make_caipirinha_mask`,
`make_random_mask` and `make_poisson_disc_mask`; only the first works on a
single phase-encode axis. Poisson-disc refuses a request it cannot meet and
says which of the two arithmetic reasons applies — the calibration block alone
being denser than the request, or the grid being too coarse for one sample to
land inside `tol`.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.calc_golden_angles
   pulserver.pypulseq.calc_raga_angles
   pulserver.pypulseq.calc_sampled_lines
   pulserver.pypulseq.calc_tiny_golden_angles
   pulserver.pypulseq.calc_traversal_order
   pulserver.pypulseq.calc_uniform_angles
   pulserver.pypulseq.make_caipirinha_mask
   pulserver.pypulseq.make_centric_order
   pulserver.pypulseq.make_linear_order
   pulserver.pypulseq.make_poisson_disc_mask
   pulserver.pypulseq.make_radial_adaptive_order
   pulserver.pypulseq.make_radial_order
   pulserver.pypulseq.make_random_mask
   pulserver.pypulseq.make_shuffling_order
   pulserver.pypulseq.make_uniform_mask
```

## System limits, rasters and RF schedules

The arithmetic a design does against an `Opts` and a count, before any event
exists. `apply_system_derates` leaves a margin under the hardware limits and
caches the originals, so calling it twice does not compound.
`round_to_raster` and `ceil_to_raster` put a time on a raster boundary --
which matters because `add_block` rounds a block up to the block raster, so a
duration computed from `calc_duration` alone is a raster short of where the
block actually ends. `quantize_readout_timing` is `calc_adc_timing` keyed on
bandwidth per pixel rather than on dwell.

The schedules are the per-repetition RF lists: quadratic spoiling, which never
repeats so coherences average away; fixed phase cycling, which deliberately
does repeat so a steady state is preserved; and Alsop's TRAPS sweep for a long
refocusing train.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.apply_system_derates
   pulserver.pypulseq.ceil_to_raster
   pulserver.pypulseq.round_to_raster
   pulserver.pypulseq.quantize_readout_timing
   pulserver.pypulseq.make_phase_cycling_schedule
   pulserver.pypulseq.make_rf_spoiling_schedule
   pulserver.pypulseq.make_traps_schedule
```

## Routines MATLAB Pulseq has and PyPulseq does not

Ports of `mr.*` functions upstream never carried over, under PyPulseq's naming.
`make_hexagon_gradient_area` reaches an area between two given amplitudes
without holding a plateau, which is sometimes shorter than
`make_extended_trapezoid_area` and never longer. `calc_rf_power` gives one
pulse's relative energy, peak power and RMS amplitude — `Sequence.calc_rf_power`
answers the same question of a whole sequence. `rotate3D` turns a block's
gradients by a full 3x3 matrix where upstream's `rotate` turns them about one
axis. `restore_additional_shape_samples` recovers the raster-edge samples of a
gradient stored on a centres raster. `verify_file_signature` re-hashes a `.seq`
and checks it against its own `[SIGNATURE]`.

`mr.addRamps` is **not** ported: it needs a working `calc_ramp`, and
`pypulseq.calc_ramp` raises for any ramp of more than zero intermediate points.
Use `traj_to_grad`, which solves the same problem under the same limits.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.calc_rf_power
   pulserver.pypulseq.get_supported_rf_use
   pulserver.pypulseq.make_hexagon_gradient_area
   pulserver.pypulseq.restore_additional_shape_samples
   pulserver.pypulseq.rotate3D
   pulserver.pypulseq.verify_file_signature
```

## Label constants

The Pulseq label set splits in two. **Counters** say where an acquisition
belongs; a module builds a slot per counter it is asked for and the scan
loop writes the value. **Flags** say how a block is played or classified.
Both are `make_label` events, and both are sticky in the file -- a value
set at one block holds until another block sets it again -- so a flag that
should not outlive its blocks has to be cleared by the loop that set it.

`COUNTER_LABELS`, `FLAG_LABELS` and `STICKY_FLAGS` are that split as module
constants: the ten ISMRMRD `EncodingCounters` fields, everything else, and the
flags that outlive the module which set them.

A third distinction cuts across the second, and it is the one to check before
choosing a label: whether it **maps to data**. Every counter does. Among the
flags, `NAV`, `REV`, `SMS`, `REF`, `IMA`, `NOISE` and `OFF` classify an
acquisition and become fields a reconstruction reads; `NOROT`, `NOPOS`,
`NOSCL`, `PMC`, `ONCE` and `TRID` are instructions to the
interpreter and stop at the scanner. Nothing downstream ever sees the second
group, so a sequence with something to tell its own reconstruction cannot say
it with one of them. {func}`~pulserver.pypulseq.get_supported_labels`
tabulates all three groups with what each label means.

## Analysis results

Every analysis method on `Sequence` takes a keyword-only `compat`. Left alone
it is `True` and the method returns exactly what upstream PyPulseq returns,
including upstream's omissions — that is what lets an unchanged PyPulseq
script keep its unchanged meaning. Passing `compat=False` returns one object
instead, carrying the same information under names plus what upstream's tuple
has nowhere to put.

The flag exists rather than a longer tuple because a tuple return is unpacked
positionally: `a, b, c = seq.waveforms_and_times()` breaks the moment a fourth
element appears, and it breaks at the caller's line. So these objects
deliberately **cannot** be unpacked — they have no `__iter__` and no
`__getitem__`, and fields are added over time without any call site changing.

| method | `compat=True` | `compat=False` |
| --- | --- | --- |
| `waveforms` | list of `(2, n)` arrays | {class}`~pulserver.pypulseq.Waveforms` |
| `waveforms_and_times` | 5-tuple | {class}`~pulserver.pypulseq.WaveformsAndTimes` |
| `rf_times` | 4-tuple | {class}`~pulserver.pypulseq.RfTimes` |
| `adc_times` | 2-tuple | {class}`~pulserver.pypulseq.AdcTimes` |
| `calculate_kspace` | 5-tuple | {class}`~pulserver.pypulseq.KSpace` |
| `calculate_pns` | 4-tuple | {class}`~pulserver.pypulseq.Pns` |
| `calculate_gradient_spectrum` | 4-tuple | {class}`~pulserver.pypulseq.GradientSpectrum` |
| `calc_rf_power` | MATLAB's 4-tuple | {class}`~pulserver.pypulseq.RfPower` |
| `calc_moments_btensor` | MATLAB's 4-tuple | {class}`~pulserver.pypulseq.BTensor` |
| `sim_rf` | MATLAB's 6-tuple | {class}`~pulserver.pypulseq.RfResponse` |

What only `compat=False` can tell you:

- **Every RF use, not two.** Upstream sorts RF into excitation and refocusing
  and silently drops inversion, saturation, preparation and other — an
  inversion pulse does not appear in its answer at all. `RfTimes` is one flat
  table with the tag kept, and `of("inversion")` selects.
- **Per-sample ADC phase and phase modulation.** MATLAB returns `pm_adc` as a
  sixth output; PyPulseq returns nothing. `AdcTimes.phase_modulation` is that
  array, and `sample_phase` is the phase each sample is actually acquired
  with, ppm terms and modulation folded in.
- **Echo centres.** `AdcTimes.echo_center_time` is when each readout reaches
  k-space zero, found by the C core walking the real trajectory rather than by
  MATLAB's `2*t_refocusing - t_excitation` approximation. Computed on first
  read, since it needs that trajectory.
- **The breakpoint-grid trajectory.** `KSpace.k_traj_breakpoints` describes the
  same curve as `k_traj` in five to ten times fewer points.
- **Acoustic resonance lines.** `GradientSpectrum.resonance_lines`, which used
  to be smuggled out as a fifth tuple element.
- **A gradient table a diffusion pipeline can read.** MATLAB's b-tensor is in
  s/m² with the step to a b-value left to the caller. `BTensor.b_tensors` is
  the same tensor in s/mm², which is exactly DIPY's `gradient_table(...,
  btens=)` argument, and `column_stack((b_vectors, b_values))` is MRtrix3's
  `[x y z b]` table. `b_delta` says whether the encoding is linear, planar or
  spherical, from the eigenvalues rather than from the sequence's name.
- **Which window the RF power is for.** `RfPower.duration` and
  `RfPower.window_duration` say whether `mean_power` is an average over the
  whole range or the worst sliding window inside it — MATLAB returns the two
  cases as the same four numbers with nothing to tell them apart.

{class}`~pulserver.pypulseq.SoftDelay` is not a `compat` return: it is what
`get_default_soft_delay_values` puts in its mapping, carrying each delay's
current value together with the range it may be set over. `float()` on it is
the value, so it drops straight back into `apply_soft_delay`.

{class}`~pulserver.pypulseq.DiffusionTable` is not one either. It is the short,
consumer-facing form of `BTensor`: one row per *distinct* encoding rather than
one per shot, plus the name of the MRD counter whose value is the row index.
`Sequence.write_diffusion_definitions(axis="SET")` computes it, checks that the
named counter really does index it, and stores it in `[DEFINITIONS]`;
`mrdserver::add_diffusion_parameters` copies those entries into the MRD header
on the scanner, and `DiffusionTable.from_definitions` reads them back. The same
type on both sides, so a table written by a design script and one recovered
from a scan are comparable.

The b-tensor comes out of that path in **three matrices, not one**, because the
FOV rotation the radiographer prescribes is not in the `.seq`: `NOROT` is how a
block opts out of it, which is what a product diffusion preparation does, while
the imaging gradients of the same shot do not — so `b_fixed`, `b_rotatable` and
`b_cross` are carried separately and `BTensor.compose(R)` puts them together
once the prescription is known.

```{eval-rst}
.. autosummary::
   :toctree: generated

   pulserver.pypulseq.Waveforms
   pulserver.pypulseq.WaveformsAndTimes
   pulserver.pypulseq.RfTimes
   pulserver.pypulseq.AdcTimes
   pulserver.pypulseq.KSpace
   pulserver.pypulseq.Pns
   pulserver.pypulseq.GradientSpectrum
   pulserver.pypulseq.RfPower
   pulserver.pypulseq.RfResponse
   pulserver.pypulseq.SoftDelay
   pulserver.pypulseq.BTensor
   pulserver.pypulseq.DiffusionTable
```
