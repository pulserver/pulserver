# Getting started

This chapter builds a 2D gradient echo, checks it against the limits a scanner
enforces, shows you the repeating unit the checks were made over, and writes
the file. It takes about ten minutes and needs no scanner.

Everything below is a complete snippet: paste it and it runs.

## Step 1: Install

```bash
pip install pulserver
```

Wheels carry the compiled C and C++ extensions, so you need no compiler, no
CMake and no vendor SDK. One install covers sequence design and
reconstruction.

## Step 2: Create the sequence

The sequence zoo ships as `pulserver.app`, so a sequence is a call rather than
a script to copy:

```python
from pulserver.app import gre2D_sequence

seq = gre2D_sequence(n_x=128, n_y=128, n_slices=1)
```

`seq` is a drop-in PyPulseq `Sequence` with a C++ core underneath, so anything
you already do to a `Sequence` works here. Every keyword of
{func}`~pulserver.app.gre2D_sequence` is a sequence parameter — matrix, FOV,
flip angle, TE, TR, bandwidth, acceleration — and the design is re-run when
you change one.

## Step 3: Check it against the hardware

Two checks answer whether the gradients are playable at all:

```python
ok, message = seq.check_hardware_limits()      # amplitude and slew
print("hardware limits :", ok, message)

ok, message = seq.check_gradient_continuity()  # joins across block boundaries
print("gradient joins  :", ok, message)
```

```
hardware limits : True
gradient joins  : True
```

A third asks whether it is playable on a *subject*. Peripheral nerve
stimulation is judged over one repetition played back to back, under the
rheobase/chronaxie model the scanner's own gate applies:

```python
within, total, per_axis, _ = seq.calculate_pns(
    {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333},
    tr="worst_case",
    do_plots=False,
)
print("pns :", within, f"peak {100 * total.max():.1f} % of threshold")
```

```
pns : False peak 133.0 % of threshold
```

**That is a failure, and it is the expected one.** This design asks for the
shortest timing the hardware admits, so its prewinders and spoiler ramp as
fast as they are allowed to. Slow them down and it comes back inside:

```python
import pulserver.pypulseq as pp

gentle = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=80, slew_unit="T/m/s")
seq = gre2D_sequence(n_x=128, n_y=128, n_slices=1, system=gentle)

within, total, _, _ = seq.calculate_pns(
    {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333},
    tr="worst_case",
    do_plots=False,
)
print("pns :", within, f"peak {100 * total.max():.1f} % of threshold")
```

```
pns : True peak 91.0 % of threshold
```

A longer TE or a lower `readout_bandwidth_hz` each work as well. Which one you
choose is a sequence decision; the check only tells you that one is needed.

## Step 4: Look at the repeating unit the checks used

Every check above was made over one repetition, because that is what the
quantities are defined over. You never declared it — it is read back from the
block content:

```python
print("blocks   :", seq.num_blocks)
print("tr_size  :", seq.tr_size)
print("num_trs  :", seq.num_trs)
print("segments :", seq.num_segments)
```

```
blocks   : 864
tr_size  : 6
num_trs  : 144
segments : 2
```

Six blocks repeat 144 times, and the interpreter will prepare two segments to
play them. Draw either one:

```python
seq.plot(tr="worst_case")   # the repetition the safety checks were made over
seq.plot(segment_idx=0)     # one unit the sequencer prepares and replays
```

If `tr_size` is not what you expected, that is worth stopping for: it is the
window every safety verdict is computed on. See
{doc}`../explanations/sequence_model/tr_and_segmentation`.

## Step 5: Write the file

```python
seq.write("gre2d.seq")
```

`write()` re-runs the hardware and continuity checks and records the repeating
unit it found as the `TRSize` definition, so the file carries its own
structure. To write for a scanner instead, use
{func}`~pulserver.design.write_sequence`, which writes the binary form and
leaves the verdict to predownload.

## Requirements

A `.seq` file is playable here if it satisfies the following. Each is checked;
none has to be annotated.

1. **The block structure must be periodic.** The repeating unit is found by
   asking which period holds over the whole block list. A design whose
   structure never repeats is refused rather than guessed at. The usual cause
   is creating gradient events *inside* the loop with varying amplitudes —
   create the event once outside the loop and scale it per shot instead.
2. **Gradients must be within the amplitude and slew limits you declare.**
   Pass the scanner's own limits into `pp.Opts`; the design is held below the
   smaller of those and any ceiling the sequence itself sets.
3. **Gradients must join across block boundaries.** A waveform may cross a
   block boundary, but the value it ends on and the value the next block
   starts on must agree.
4. **Event times must land on the rasters you declare**, and the system and
   sequence rasters must be integer multiples of each other.
5. **One block position may hold several readouts, but not unboundedly many.**
   Two echoes of different length at one position become two prepared
   segments; a scan that digitises differently on every repetition is refused.

Note what is *not* required: you do not label segment boundaries, and you do
not mark the TR. Both are derived from the content, so they cannot disagree
with the sequence as written.

## When a check fails

| What you see | What it usually means |
|---|---|
| `check_hardware_limits` false | A gradient exceeds the amplitude or slew you declared. Lower `max_slew`, lengthen the event, or raise the declared limit if the hardware really allows it. |
| `check_gradient_continuity` false | A waveform ends at a non-zero value and the next block does not continue it. Add the ramp, or let the two blocks share one extended trapezoid. |
| `calculate_pns` false | The design stimulates above threshold. Slow the ramps, lengthen TE, or lower the readout bandwidth. |
| `tr_size` equals the whole block count | No period was found. See requirement 1. |
| `num_segments` much larger than expected | Something varies that you did not intend to vary — most often a delay whose duration changes, or two readouts of different length at one position. |

## Where to go next

- {doc}`../explanations/index` — why the repeating unit is derived rather than
  declared, what each safety check is defined over, and what the performance
  budget is.
- {doc}`../examples/index` — the sequence zoo, one runnable module per
  sequence, and the reconstruction plugins that read them back.
- {doc}`../api/python/index` — the design toolbox, if you are writing a
  sequence rather than running one.
