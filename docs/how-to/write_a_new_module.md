# How to write a new module or loop structure

The shipped factories cover the standard RF, preparation and readout families.
When yours is not among them, you write a `SequenceModule`. When your *loop*
is not among them, you write nothing at all — that is the point of leaving the
loop in the plugin.

## Do you actually need a new module?

Reach for these first:

- **A different waveform in a shipped structure** — pass the prebuilt pulse to
  the factory (`make_bssfp_readout(..., excitation)`,
  `make_fse_readout(..., refocusing)`).
- **A different k-space traversal in a shipped readout** — build the
  `ScanLoop` yourself and pass the indices; the readout does not care
  where they came from.
- **A single-block gradient or delay** — that is a plain Pulseq event, not a
  module. `seq.add_block(event)` already works.

Write a module when you have **several blocks that always travel together and
must be re-rendered per shot**. That is the whole contract.

## Implement the contract

A subclass provides exactly two things: `_set_state`, which replaces the
complete dynamic state, and `_current_blocks`, which returns the block
snapshot for that state. The public `set_state` — which takes the labels out
before delegating the numbers to `_set_state` — plus the collection protocol,
counters, flags, triggers, `duration`, `plot` and `get`, come from the base
class.

Here is a spatial saturation band — a slice-selective pulse plus a spoiler,
positioned anywhere along an axis per shot. It is genuinely not in the shipped
set, and it is small enough to read whole:

```python
import copy

import numpy as np
import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import SequenceModule


class SaturationBand(SequenceModule):
    """A spatially selective saturation band, positioned per shot.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    thickness_m : float
        Band thickness (m).
    axis : str
        Gradient channel the band is selective along.
    flip_rad : float
        Saturation flip angle (rad).
    duration : float
        RF duration (s).
    """

    def __init__(self, system, thickness_m, *, axis="z", flip_rad=np.deg2rad(90.0), duration=2e-3,
                 **declared):
        # **declared carries labels=/flags=/flag_scope=/triggers= straight
        # through to the base, so a caller can declare them at construction.
        super().__init__(system, **declared)
        self.axis = axis
        self.thickness_m = float(thickness_m)

        # Everything expensive happens once, here.
        rf, gz, _ = pp.make_sinc_pulse(
            flip_angle=float(flip_rad),
            duration=float(duration),
            slice_thickness=self.thickness_m,
            apodization=0.5,
            time_bw_product=4,
            system=system,
            return_gz=True,
            use="saturation",
        )
        gz.channel = axis
        self._rf, self._gz = rf, gz
        self._spoiler = design.make_crusher(system, axis, dephasing_cycles=4.0, voxel_size=self.thickness_m)
        # Hz per metre of band offset, read off the selection gradient itself.
        self._hz_per_m = gz.amplitude

        self._position_m = 0.0
        self._phase_offset_rad = 0.0
        self._blocks = None

    def _set_state(self, position_m=0.0, phase_offset_rad=0.0):
        """Place the band at ``position_m`` from isocentre and set its phase."""
        self._position_m = float(position_m)
        self._phase_offset_rad = float(phase_offset_rad)
        self._blocks = None  # invalidate the snapshot

    def _current_blocks(self):
        if self._blocks is None:
            rf = copy.deepcopy(self._rf)
            rf.freq_offset = self._position_m * self._hz_per_m
            rf.phase_offset = self._phase_offset_rad
            self._blocks = ((rf, self._gz), (self._spoiler,))
        return self._blocks


def make_saturation_band(system, thickness_m, **kwargs):
    """Build a :class:`SaturationBand`."""
    return SaturationBand(system, thickness_m, **kwargs)
```

It behaves like everything else in the toolbox from the first line:

```python
band = make_saturation_band(opts, 20e-3, axis="y")

band.set_state(position_m=0.05)
band.num_blocks            # 2
band.duration              # summed from the snapshot, since we publish none
for block in band:
    seq.add_block(*block)
band.plot()                # one-module sequence diagram

# One setter for everything that moves: lowercase names are this module's own
# numbers, uppercase ones are counters and flags.
band.set_state(position_m=0.05, SET=1, NOPOS=1)

# Triggers belong to the design, so they are declared with the module.
gated = make_saturation_band(
    opts, 20e-3, axis="y",
    triggers=(pp.make_digital_output_pulse("osc0", duration=100e-6),),
)
```

## Five rules the contract depends on

1. **Design in `__init__`, render in `_current_blocks`.** A loop over
   thousands of shots must not redesign waveforms. If a shot needs a waveform
   the constructor cannot produce, that is a different module, not a state.
2. **`_set_state` replaces the *complete* state.** No partial updates:
   anything a caller omits must revert to its default, so a stale value from
   three shots ago can never leak into this one. Labels are the one exception
   and the base class owns it — a label event is structure once named, so its
   value persists until something changes it.
3. **Invalidate the cache in `_set_state`.** Setting `self._blocks = None` is
   the whole mechanism. Forget it and every shot silently replays the first
   one's waveforms.
4. **Never hand out an event a caller could mutate into your template.**
   `copy.deepcopy` the events you modify per shot; share the ones you do not
   (`self._gz`, `self._spoiler` above are constant across shots).
5. **Use the shared state vocabulary.** `lin_idx`, `par_idx`,
   `phase_offset_rad`, `freq_offset_hz`, `amplitude_scale`, `rotation` mean
   the same thing everywhere. A new name is a new concept — `position_m`
   above is one, because no shipped module takes a position in metres.

If your module knows its own duration analytically, publish it and callers get
it for free:

```python
self.duration = self._rf_duration_s + pp.calc_duration(self._spoiler)
```

Otherwise the base class sums the rendered blocks.

## Emit your own counters — but only the ones you own

A readout emits `LIN`/`PAR`/`ECO` because it is the only thing that knows
them. Append the label events to the relevant block inside `_current_blocks`:

```python
self._blocks = (
    (rf, self._gz, pp.make_label(type="SET", label="LIN", value=self._lin_idx)),
    (self._spoiler,),
)
```

Do not emit `SLC`, `REP`, `SET`, `PHS` or `AVG`: those belong to the loop, and
`set_state` already merges them into block 0.

**Flags are never hard-coded either.** If your module is always exempt from the
FOV transform, or is always a navigator, declare that with the base class in
`__init__` rather than baking the label events into `_current_blocks` — the
shipped preparations do exactly this:

```python
super().__init__(system, flags={"NOPOS": 1, "NOROT": 1})
```

The base then scopes them for you (set on the first block, reset on the last),
and a caller who sets them later *moves the value* in the event you declared
instead of fighting it. Baked-in label events cannot be overridden at all, and
cannot be recognised by a TR template either.

## New loop structures need no new type

There is deliberately no loop object to subclass. A "new loop structure" is a
`for` statement.

**Quantitative T1 — an inversion-time loop outside the k-space loop.** The
`for` is yours; `make_counter_loop` is what makes the dimension visible to the
reconstruction:

```python
inversions = design.make_counter_loop([0.1, 0.5, 1.2, 2.5], label="SET")

for c in range(len(inversions)):
    for block in inversion.set_state(**inversions.label_state(c)):
        seq.add_block(*block)
    seq.add_block(pp.make_delay(float(inversions[c][0, 0])))
    for shot in sampling:
        for block in excitation.set_state(**inversions.label_state(c)):
            seq.add_block(*block)
        for block in readout.set_state(lin_idx=int(shot[0, 0])):
            seq.add_block(*block)
    seq.add_block(pp.make_delay(recovery_s))
```

**Multi-echo spectroscopy or MRSI — a TE loop with no phase encoding at all:**

```python
for te_s in echo_times_s:
    for block in excitation.set_state():
        seq.add_block(*block)
    seq.add_block(pp.make_delay(te_s / 2))
    for block in refocusing.set_state():
        seq.add_block(*block)
    seq.add_block(pp.make_delay(te_s / 2))
    seq.add_block(adc)
```

**Dynamic imaging with a shared chronology across every level** — one
iterator, consumed innermost, so no counter bookkeeping:

```python
rotations = iter(design.make_noncartesian_2d_sampling(matrix, views=total_views).to_rotations())
phases = iter(design.make_rf_spoiling_schedule(total_views))
frames = design.make_counter_loop(n_frames, label="REP")

for f in range(len(frames)):
    seq.add_block(trigger)
    for s, slice_shot in enumerate(slices.shots):
        for view in range(views_per_frame):
            phase = float(next(phases))
            excitation.set_state(
                freq_offset_hz=float(offsets[slice_shot[0]]),
                phase_offset_rad=phase,
                **frames.label_state(f),
                **slices.label_state(s),
            )
            for block in excitation:
                seq.add_block(*block)
            for block in readout.set_state(lin_idx=view, rotation=next(rotations), phase_offset_rad=phase):
                seq.add_block(*block)
```

**Interleaved contrasts in one scan** — `zip`, `itertools.cycle`, or an index;
nothing special:

```python
import itertools

preparations = itertools.cycle((t2_prep, None, mt_prep, None))
for shot in sampling:
    prep = next(preparations)
    if prep is not None:
        for block in prep.set_state():
            seq.add_block(*block)
    ...
```

The rule of thumb: if you find yourself wanting the toolbox to own your loop,
what you actually want is either a `ScanLoop` that spans the whole chronology
plus `iter()`, or a `make_counter_loop` axis for the dimension you are
hand-rolling — the `for` stays yours, the counter stops being your problem.
`make_counter_loop(n, label=..., order=...)` covers any remaining
one-dimensional loop — partitions, inversion times, diffusion directions —
with the same traversal orders (`sequential`, `interleaved`, `reverse`,
`center_out`, `outside_in`) the slice loop uses.

## Test it the way the shipped modules are tested

Replay into an upstream `pypulseq.Sequence` — Pulserver's own `Sequence` is a
write-only fast builder and cannot be read back — then assert on what came
out:

```python
import pypulseq

seq = pypulseq.Sequence(system=opts)
for block in band.set_state(position_m=0.06):
    seq.add_block(*block)
assert seq.check_timing()[0]
assert len(seq.block_events) == band.num_blocks
```

For anything that encodes, assert that the k-space you *planned* is the
k-space you *played*:

```python
k_adc, *_ = seq.calculate_kspace()
planned = sorted(int(shot[0, 0]) - ny // 2 for shot in sampling)
played = np.round(np.unique(np.round(k_adc[1] * fov_pe_m, 6)), 3)
assert np.array_equal(played, planned)
```

## Next

- [Scan-loop reference](../reference/sampling.md) — the loop factories your
  module will be driven by.
- [Write a sequence plugin with the design toolbox](../tutorials/build_a_sequence_plugin.md)
