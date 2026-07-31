# Scan loops

Pulserver describes an acquisition with one loop type and **no** object that
owns the loop nesting:

- `ScanLoop` — a table of **positions**, a grouping of them into **shots** (one
  shot being one excitation's worth of the loop), and one `EncodingAxis` per
  position column saying what the numbers mean. Every sampling and scheduling
  factory returns one.
- the loop nesting itself — plain `for` statements in the sequence, over
  frames, contrasts, averages, inversion times, or anything else.

There is deliberately no type that flattens frames × slices × shots into one
iterable. That flattening is what makes it hard to put a trigger at the top of
a frame, an inversion at the top of a shot, or a dummy TR anywhere; keeping the
loop in the plugin costs three lines and buys all of it back.

A `ScanLoop` never accepts a protocol object, appends blocks, or depends on a
sequence module. It carries numbers, and the counter each column is labelled
with.

## A loop is not a k-space object

An encoding position is a set of frequency offsets, gradient scalings and
rotations. Nothing about that restricts it to `ky`, `kz`, `z` or an interleave
angle: a frame of a dynamic series, an inversion time, a b-value, a saturation
offset are the same table under a different axis declaration. So one type
covers every dimension of a scan, and the only thing that distinguishes them is
which counter they emit.

| Loop | `positions` hold | axis `kind` | counter |
|---|---|---|---|
| Cartesian k-space | `(ky[, kz])` integer indices | `index` | `LIN`, `PAR` |
| non-Cartesian | spoke angles or unit directions | `angle`, `direction` | `LIN` |
| slices / SMS | positions along the axis (m) | `position` | `SLC` |
| frames, dynamics | frame number | `index` | `REP` |
| contrasts | TI, b-value, offset, … | `value` | `SET`, `PHS` |
| averages | average number | `index` | `AVG` |

The echo dimension is the exception, and it is not an omission: `ECO` lives
inside the readout, because a multiecho train's echoes are blocks of one shot
rather than iterations of a loop. `num_echoes` belongs to `make_line_readout`,
and the readout emits `SET ECO 0` / `INC ECO 1` itself.

## Counters, and why they are not optional

A loop carries no gradients and no events — it carries the numbers those are
derived from, plus the counter that lets the reconstruction find the data
again:

```python
sampling = design.make_cartesian_sampling((256, 128), acceleration=2)

shot   = sampling[0]        # absolute (ky[, kz]) -> lin_idx / par_idx
scale  = sampling.to_scales()  # [-1, 1) -> pp.scale_grad(gy, ...)
labels = sampling.label_state(0)   # {"LIN": 0} -- the counter value for shot 0
```

`label_state(shot)` returns `{counter: value}`, one entry per axis. Spread it
into `SequenceModule.set_state` for every loop the readout does *not* consume
itself:

```python
excitation.set_state(freq_offset_hz=offsets[s], **frames.label_state(f), **slices.label_state(s))
```

**Emitting them is what makes the MRD `FIRST_IN_*` / `LAST_IN_*` flags
correct.** Those flags are not written by the sequence. The interpreter
computes them by comparing each acquisition's counter against the *observed*
minimum and maximum of that counter over the scan. A frame loop that never
emits `REP` therefore reports `min == max == 0`, and every acquisition comes
back flagged both first *and* last in repetition — which is exactly the failure
that makes a dynamic reconstruction fire once per line instead of once per
frame. `label_limits()` is the loop's own prediction of where those flags will
land:

```python
sampling.label_limits()      # {'LIN': (0, 254)}
frames.label_limits()        # {'REP': (0, 39)}
```

The label value is the position *value* on an `index` axis — its numbers
already are the encoding indices — and the position *index* on every other
kind, because angles, metres and inversion times are not counters but their
order is. That is the rule the shipped modules have always followed; declaring
the axis is what makes it executable rather than a convention you have to
remember.

## Which factory should I call?

| Acquisition | Factory | Important knobs |
|---|---|---|
| 2D/3D GRE, including multiecho | `make_cartesian_sampling` | `train_length=1`, matrix, acceleration |
| 2D/3D FSE | `make_cartesian_sampling` | `train_length=ETL`, FSE ordering |
| 2D/3D MPRAGE or segmented GRE | `make_cartesian_sampling` | segment length, `ordering="centric"` |
| 2D/3D EPI | `make_epi_sampling` | acceleration, segments, CAIPI shift |
| 2D radial/spiral/rosette | `make_noncartesian_2d_sampling` | view count, angle scheme |
| 3D stack of trajectories | `make_noncartesian_2d_sampling` + `calc_traversal_order` | views, partition order |
| 3D projection | `make_noncartesian_projection_sampling` | views, direction scheme |
| the slice loop of any 2D scan | `make_slice_loop` | slice count, spacing, order, SMS |
| frames, contrasts, averages, any other counter | `make_counter_loop` | count or value table, `label`, order |

Every one of them fills in its own axes, so `to_scales()` needs no matrix
argument and `labels()` needs no label argument.

## Cartesian GRE, FSE and MPRAGE

One factory covers these families because their only difference at the
sampling level is how many encoded locations occur inside one excitation.

```python
import pulserver.design as design
import pulserver.pypulseq as pp

gre2d = design.make_cartesian_sampling((128, 128), acceleration=2, calibration=16)
gre3d = design.make_cartesian_sampling((128, 128, 64), acceleration=(2, 2), calibration=(16, 8))
fse2d = design.make_cartesian_sampling((128, 128), train_length=16, ordering="radial")
fse3d = design.make_cartesian_sampling((128, 128, 64), train_length=32, ordering="radial_adaptive")
mprage3d = design.make_cartesian_sampling(
    (256, 192, 160), acceleration=(2, 2), train_length=128, ordering="centric",
)
```

The loop consumes integer coordinates:

```python
for shot in fse3d:
    readout.set_state(lin_idx=shot[:, 0], par_idx=shot[:, 1])
    for block in readout:
        seq.add_block(*block)
```

An FSE shot is a whole train, so `shot[:, 0]` is the per-echo `ky` schedule.
A GRE shot has one entry, so `int(shot[0, 0])` is the line.

`to_scales()` converts the whole table at once, but it is indexed like
`positions` — on an accelerated loop those are only the sampled lines, so index
it with `scales[loop.shots[i]]`, never `scales[ky]`. Converting each shot's own
coordinates with `calc_encoding_scales` avoids the question. The extents come
from the *encoded matrix*, not from the sampled support, so acceleration and
partial Fourier drop views without rescaling the ones that remain.

## The slice loop

For a 2D scan the slice loop is a separate `ScanLoop`, nested outside or
inside the k-space loop as the sequence requires. Its rows select one physical
slice, or `sms_factor` simultaneous bands; it covers every requested position
exactly once. Slice spacing fixes the physical positions, and the
selective-gradient amplitude converts them to RF frequency offsets:

```python
slices = design.make_slice_loop(48, spacing_m=3e-3, order="interleaved", sms_factor=3)
offsets = slices.to_frequencies(excitation.gradients[0].amplitude)

for s, slice_shot in enumerate(slices.shots):
    excitation.set_state(freq_offset_hz=float(offsets[slice_shot[0]]), **slices.label_state(s))
    for shot in sampling:
        ...
```

Iterate `slices` when you want positions, `slices.shots` when you want
indices, and `labels(s)` when you want the `SLC` counter. Pass the whole
`offsets[slice_shot]` row rather than `[0]` to an SMS multiband pulse;
`labels(s)` then names the first band, and `labels(s, position=n)` any other.

## Frames, contrasts and averages

`make_counter_loop` builds the loop over any counter that is neither k-space
nor slices. Two forms, chosen by what you pass:

```python
frames     = design.make_counter_loop(40, label="REP")                  # a bare counter
inversions = design.make_counter_loop([0.1, 0.3, 1.0, 2.5], label="SET")  # values it schedules
averages   = design.make_counter_loop(4, label="AVG")
```

The loop does not know what its values *drive*. Converting an inversion time
into a delay, or a saturation offset into `freq_offset_hz`, is the sequence's
business — which is exactly why one factory covers every contrast dimension
instead of one per physics:

```python
for f in range(len(frames)):
    for c in range(len(inversions)):
        inversion.set_state(**frames.label_state(f), **inversions.label_state(c))
        for block in inversion:
            seq.add_block(*block)
        seq.add_block(pp.make_delay(float(inversions[c][0, 0])))
        for shot in sampling:
            for block in readout.set_state(lin_idx=int(shot[0, 0])):
                seq.add_block(*block)
```

`order=` reorders the *visits* without renumbering the data — the counter
follows the position, so `make_counter_loop(6, label="AVG", order="center_out")`
still labels average 2 as `AVG 2`, it just acquires it first.

`with_axes` retargets an existing loop's counter without rebuilding it, for
when a rotation loop indexes frames rather than views:

```python
from pulserver import EncodingAxis

phases = design.make_noncartesian_2d_sampling(matrix, views=n).with_axes(
    EncodingAxis("PHS", kind="angle")
)
```

## EPI: structural and dynamic

`make_epi_sampling` describes one volume's interleaves. Repeating them is the
frame loop, and nothing in the loop object changes:

```python
epi3d = design.make_epi_sampling((96, 96, 32), acceleration=(2, 2), segments=2, caipi_shift=1)
frames = design.make_counter_loop(n_frames, label="REP")

readouts = [
    design.make_epi_readout(system, fov, matrix, epi3d.n_shots, epi3d.relative(shot))
    for shot in range(epi3d.n_shots)
]

for f in range(len(frames)):
    for shot, readout in enumerate(readouts):
        start = epi3d[shot][0]
        readout.set_state(
            lin_idx=int(start[0]),
            par_idx=int(start[1]),
            labels=epi3d[shot],
            **frames.label_state(f),
        )
        for block in readout:
            seq.add_block(*block)
```

A loop may contain more than one traversal, which is why the readout is
built per shot from `relative(shot)` and only the absolute start varies per
frame.

## Non-Cartesian: continuous or replayed chronology

`views` is the length of the whole angular chronology, not a per-frame count.
Whether golden angles continue across frames and slices or replay at each one
is decided by *where you call `iter()`* — there is no `continuous=` switch:

```python
tilt = design.make_noncartesian_2d_sampling(
    (192, 192), views=n_frames * n_slices * n_views, scheme="tiny_golden", tiny_index=2,
)

rotations = iter(tilt.to_rotations())          # continuous across the whole scan
for f in range(len(frames)):
    for s, slice_shot in enumerate(slices.shots):
        for view in range(n_views):
            readout.set_state(
                lin_idx=view,
                rotation=next(rotations),
                **frames.label_state(f),
                **slices.label_state(s),
            )
            for block in readout:
                seq.add_block(*block)
```

Move `rotations = iter(...)` inside the slice loop and every slice replays the
same angular set.

A stack of trajectories is the same in-plane loop with a partition loop
around it:

```python
rotations = iter(design.make_noncartesian_2d_sampling(matrix, views=nz * n_views).to_rotations())
partitions = design.make_counter_loop(nz, label="PAR", order="center_out")
for par in partitions:
    for view in range(n_views):
        readout.set_state(lin_idx=view, par_idx=int(par[0, 0]), rotation=next(rotations))
```

3D projections need no partition loop:

```python
projection = design.make_noncartesian_projection_sampling(
    (192, 192, 192), views=20_000, scheme="golden_means",
)
for view, rotation in enumerate(projection.to_rotations()):
    readout.set_state(lin_idx=view, rotation=rotation)
```

## When a factory is almost right

Support and ordering are independent stages, and both are reachable without
leaving the public API.

Replace only the **ordering**: every scan-loop factory takes `ordering=`
(`linear`, `centric`, `radial`, `radial_adaptive`, `shuffling`), and
`make_slice_loop` / `make_counter_loop` take `order=` (`sequential`,
`interleaved`, `reverse`, `center_out`, `outside_in`).

Replace only the **support**: `make_cartesian_sampling` covers `mask="uniform"`,
`"caipi"`, `"random"` and `"poisson"`. Beyond those, hand a boolean array to
{meth}`~pulserver.ScanLoop.from_mask` and keep the shipped ordering:

```python
import numpy as np
from pulserver import ScanLoop

mask = np.zeros((128, 64), dtype=bool)
mask[::4, ::2] = True
mask[56:72, 28:36] = True
sampling = ScanLoop.from_mask(mask, train_length=16, ordering="radial")
```

`from_mask` checks that the ordering covers every sampled location exactly
once, so a hand-written mask cannot silently drop or duplicate lines.
{meth}`~pulserver.ScanLoop.from_relative_shifts` does the same for
EPI-style relative traversals, and a `ScanLoop` can always be constructed
directly from positions and shots when neither fits.

Replace the **angles**: `make_noncartesian_2d_sampling` takes `scheme=`
(`linear`, `golden`, `tiny_golden`, `raga`) with `period`, `tiny_index` and
`approximation_order`, plus `segment_length` for prepared trains;
`make_noncartesian_projection_sampling` takes `scheme=` (`golden_means`,
`phyllotaxis`) and `interleaves`.

A loop built by hand from bare positions infers its axes: integer columns
become `LIN`/`PAR` encoding indices, three float columns a `direction` axis,
anything else a neutral `value` axis that labels by index and leaves every
converter available. Declare them explicitly when you want a different counter
or a matrix extent the positions cannot imply:

```python
from pulserver import EncodingAxis, ScanLoop

loop = ScanLoop(positions, shots, axes=(EncodingAxis("LIN", "index", size=192),))
```

## Relative EPI traversals

`ScanLoop.from_relative_shifts` is the escape hatch for EPTI, zigzag or
another custom blipped train — a shot is a start point plus per-echo
increments:

```python
sampling = ScanLoop.from_relative_shifts(
    starts=[[8, 2], [9, 2]],
    shifts=[[[0, 0], [2, 1], [4, 2]],
            [[0, 0], [2, -1], [4, -2]]],
    shape=(32, 8),
)
sampling.increments(0)   # blip areas to play between echoes
sampling[0]              # absolute coordinates to label
```

## References

- Stirnberg et al., segmented skipped-CAIPI, DOI `10.1002/mrm.28486`.
- Dong et al., EPTI encoding, DOI `10.1002/mrm.28295`.
- Scholand et al., RAGA, DOI `10.1002/mrm.30254`, arXiv `2401.02892`.
- Piccini et al., spiral phyllotaxis, DOI `10.1002/mrm.22898`.
- SigPy `sigpy.mri.samp.poisson`, BSD 3-Clause License.
