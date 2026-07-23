# How to customise a plugin

Recipes for changing one part of a working plugin without touching the rest.
Each assumes the structure of [Write a sequence plugin with the design
toolbox](write_a_plugin_with_modules.md): modules designed once, sampling and
slice objects built once, and a loop you own.

The whole point of the split is that these changes are local. Swapping the
excitation does not touch the readout; changing the sampling does not touch
either; adding a preparation touches only the loop.

## Swap the excitation

Every `make_*_pulse` factory returns the same `SequenceModule`, so an
excitation is replaced by rebinding one name:

```python
excitation = design.make_slice_selective_pulse(np.deg2rad(12), thickness_m, system=opts)

# spectral-spatial instead — water-selective, fat left alone
excitation = design.make_spsp_pulse(np.deg2rad(12), 8e-3, spectral_bandwidth=220.0, system=opts)
```

The loop is unchanged: both accept `freq_offset_hz`, `phase_offset_rad`,
`amplitude_scale` and `rotation`, and both iterate to blocks.

What *does* change is timing, so re-derive the TE budget rather than
hard-coding it — the SPSP pulse above is roughly five times longer than the
sinc:

```python
rf_center_s = pp.calc_rf_center(excitation.rf)[0] + excitation.rf.delay
min_te_s = (excitation.duration - rf_center_s) + readout.t_first_echo_s
```

This is why the TE/TR arithmetic belongs in one helper called from both
`validate_protocol` and `make_sequence`.

Non-selective and multiband work the same way:

```python
excitation = design.make_hard_pulse(np.deg2rad(12), duration=0.5e-3, system=opts)
excitation = design.make_multiband(excitation, num_bands=3, band_offset=band_offset_hz)
```

## Add a preparation

A preparation is a module like any other. Play it at whatever nesting level
its physics requires — that choice is why the loop is not hidden inside a plan
object.

Once per shot, for a segmented inversion-recovery (MPRAGE-style) acquisition:

```python
inversion = design.make_inversion_pulse(system=opts, duration=6e-3)
sampling = design.make_cartesian_sampling(matrix, train_length=16, ordering="centric")

for shot in sampling:
    inversion.set_state().add_to(seq)
    seq.add_block(pp.make_delay(ti_s))
    for ky in shot[:, 0]:
        excitation.set_state().add_to(seq)
        readout.set_state(lin_idx=int(ky)).add_to(seq)
```

Once per TR, for fat saturation:

```python
fatsat = design.make_fat_saturation_pulse(b0=3.0, voxel_size=thickness_m, system=opts)

for shot in sampling:
    fatsat.set_state().add_to(seq)
    excitation.set_state().add_to(seq)
    readout.set_state(lin_idx=int(shot[0, 0])).add_to(seq)
```

Once per frame, for a trigger — the frame loop is a `ScanLoop` like any other,
so it labels itself:

```python
frames = design.make_counter_loop(n_frames, label="REP")

for f in range(len(frames)):
    seq.add_block(pp.make_trigger("physio1", duration=100e-6))
    for s, slice_shot in enumerate(slices.shots):
        for shot in sampling:
            excitation.set_labels(*frames.labels(f), *slices.labels(s))
            ...
```

Every N shots, for a periodic navigator or a magnetisation reset — an
ordinary `if`:

```python
for index, shot in enumerate(sampling):
    if index % 32 == 0:
        navigator.set_state().add_to(seq)
    ...
```

## Change the sampling

The readout does not care how the indices were chosen, so the sampling object
is replaced on its own:

```python
sampling = design.make_cartesian_sampling((64, 64, 32), acceleration=(2, 2))
sampling = design.make_cartesian_sampling(
    (64, 64, 32), acceleration=(2, 2), mask="poisson", calibration=(12, 8)
)
```

The built-in modes cover uniform, CAIPI, random and Poisson-disc support:

```python
sampling = design.make_cartesian_sampling(
    (256, 64, 32), acceleration=(2, 2), mask="caipi", caipi_shift=1,
    train_length=16, ordering="shuffling", seed=7,
)
```

When none of them fits, the mask and the ordering are independent stages, so
you can supply your own boolean support — anything that indexes as
`[ky, kz]` — and let `ScanLoop` order it:

```python
import numpy as np
from pulserver import ScanLoop

mask = np.zeros((64, 32), dtype=bool)
mask[::2, ::2] = True                      # whatever support your scheme wants
mask[28:36, 12:20] = True                  # plus a fully sampled centre
sampling = ScanLoop.from_mask(mask, train_length=16, ordering="shuffling", seed=7)
```

`from_mask` verifies that the ordering covers every sampled location exactly
once, so a hand-written mask cannot silently drop or duplicate lines.

## Go from Cartesian to non-Cartesian

Change the readout and the sampling together; the loop keeps its shape, with
`rotation` replacing the phase-encode index:

```python
readout = design.make_radial_readout(opts, fov_m, nx)
tilt = design.make_noncartesian_2d_sampling(matrix, views=n_views, scheme="golden")

for view, rotation in enumerate(tilt.to_rotations()):
    excitation.set_state().add_to(seq)
    readout.set_state(lin_idx=view, rotation=rotation).add_to(seq)
```

`lin_idx` is still the label — for a non-Cartesian acquisition it is the view
number rather than a ky index. Spiral and rosette differ only in which
`make_*_readout` you call; the sampling object is identical, because it plans
rotations rather than waveform shape.

To make the angular chronology continue across outer loops, ask for the whole
scan's worth of views and consume one iterator; to replay the same set at each
outer position, re-`iter()` inside:

```python
rotations = iter(design.make_noncartesian_2d_sampling(matrix, views=n_frames * n_views).to_rotations())
for frame in range(n_frames):
    for view in range(n_views):
        readout.set_state(lin_idx=view, rotation=next(rotations)).add_to(seq)
```

## Vary the flip angle across a train

`amplitude_scale` scales the RF envelope of an already-designed pulse, so
design the pulse at the *largest* flip in the schedule and scale down:

```python
flips = design.make_traps_schedule(n_echoes, np.deg2rad(120))
peak = float(np.max(flips))
excitation = design.make_slice_selective_pulse(peak, thickness_m, system=opts)

for flip in flips:
    excitation.set_state(amplitude_scale=float(flip) / peak).add_to(seq)
```

Designing at the peak matters: scaling *up* past the designed amplitude can
exceed the B1 limit the pulse was validated against.

## Supply your own prebuilt pulse to a readout

Readouts whose train contains RF take that RF as an argument rather than
selecting it from a string, so a caller-built pulse drops straight in:

```python
custom_exc = design.make_slice_selective_pulse(np.deg2rad(35), thickness_m, duration=1.0e-3, system=opts)
bssfp = design.make_bssfp_readout(opts, fov, matrix, custom_exc)

custom_refoc = design.make_refocusing_pulse(slice_thickness=thickness_m, duration=2.5e-3, system=opts)
fse = design.make_fse_readout(opts, fov, matrix, echo_train_length, custom_refoc)
```

Anything the shipped RF factories cannot express — a measured waveform, an
optimal-control pulse, a pulse designed elsewhere — becomes a module of your
own; see [Write a new module or loop structure](write_a_new_module.md).

## Add a dimension: frames, contrasts, averages

Any counter is a loop axis, so a new dimension is one `make_counter_loop` and
one `for`. A dynamic series:

```python
frames = design.make_counter_loop(n_frames, label="REP")

for f in range(len(frames)):
    for shot in sampling:
        excitation.set_labels(*frames.labels(f))
        ...
```

A contrast dimension carries the values it schedules, and the sequence decides
what they drive — a delay here, an amplitude or a frequency offset elsewhere:

```python
inversions = design.make_counter_loop([0.1, 0.3, 1.0, 2.5], label="SET")

for c in range(len(inversions)):
    inversion.set_state().set_labels(*inversions.labels(c)).add_to(seq)
    seq.add_block(pp.make_delay(float(inversions[c][0, 0])))
    ...
```

`order=` changes the acquisition order without renumbering the data, since the
counter follows the position:

```python
averages = design.make_counter_loop(4, label="AVG", order="interleaved")
```

## Add a counter the loop owns

Modules emit the counters they own: a readout emits `LIN`, `PAR` and `ECO`
because it knows them. Everything an outer loop knows — `SLC`, `REP`, `SET`,
`PHS`, `AVG`, `SEG` — goes through `set_labels`, which merges into the module's
first block. Take them from the loop rather than writing them by hand:

```python
excitation.set_labels(*frames.labels(f), *slices.labels(s))
```

The keyword form spells the same thing when there is no loop object:

```python
excitation.set_labels(SLC=3, REP=frame)
```

`set_labels` replaces the whole counter state, exactly like `set_state`; call
it with no arguments to clear. Emit one for every loop you write — the MRD
`FIRST_IN_*` / `LAST_IN_*` flags are derived downstream from each counter's
observed range, so an unlabelled dimension collapses to a single index and both
its flags fire on every acquisition.

## Flag a module: prescan, dummy TRs, FOV exemption

Flags are the other half of the label set: not *where an acquisition belongs*
but *how these blocks are played or classified*. They go through `set_flags`,
which is independent of `set_labels` — replacing the per-shot counters never
disturbs them.

Play an ADC but discard its data, as a prescan or a dummy readout does:

```python
readout.set_state(lin_idx=0).set_flags(OFF=1).add_to(seq)
```

Mark leading dummy TRs as a preparation section, so the interpreter knows they
are played once rather than repeated:

```python
for _ in range(n_dummies):
    excitation.set_state(phase_offset_rad=phase).set_flags(ONCE=1).add_to(seq)
    readout.set_state(lin_idx=0).set_flags(ONCE=1, OFF=1).add_to(seq)
```

Group consecutive modules under one id so the safety model treats them as a
unit:

```python
fatsat.set_flags(MODULE=1)
excitation.set_flags(MODULE=2)
```

Pulseq labels are sticky, so `set_flags` scopes them by default: the value goes
on the module's first block and `0` on its last, and it cannot leak into
whatever follows. `ONCE`, `MODULE` and `TRID` are exempt — they deliberately
span modules, which is why `ONCE=1` above stays set across both the excitation
and the readout while `OFF` clears at the end of the readout it belongs to.
Override per call with `scope="sticky"` or `scope="module"`.

Preparation modules already carry their own FOV exemptions (`NOPOS`, `NOROT`)
this way, so `make_fat_saturation_pulse` and `make_diffusion_prep` need nothing
from you; calling `set_flags` on one *replaces* that default rather than adding
to it.

## Add a trigger or a digital output

Triggers and digital output pulses are ordinary block events, but which block
they belong on is a property of the module, so `set_triggers` puts them there:

```python
excitation.set_triggers(pp.make_trigger("physio1", duration=100e-6))
readout.set_triggers(pp.make_digital_output_pulse("osc0", duration=100e-6), block=-1)
```

Each call replaces only the block it names, so arming several blocks is several
calls; passing no events clears that block. When the trigger belongs to the
*sequence* rather than to a module — once per frame, before anything is played
— it stays a plain event:

```python
for f in range(len(frames)):
    seq.add_block(pp.make_trigger("physio1", duration=100e-6))
    ...
```

## Next

- [Write a new module or loop structure](write_a_new_module.md)
- [Scan-loop reference](../reference/sampling.md)
