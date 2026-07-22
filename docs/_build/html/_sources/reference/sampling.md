# Sampling plans

Pulserver exposes two deliberately separate levels:

- `AcquisitionPlan` is the normal sequence-author API. Its factories accept
  plain UI-facing numeric values: image matrix, acceleration, ETL or segment
  length, frames, slice geometry, SMS and non-Cartesian view count. Iterating
  it yields one `Acquisition` per sequence iteration.
- `SamplingPattern` is the lower-level support/order representation. It joins
  a custom boolean mask to an ordering, or describes raw angle/direction
  support. Mask generators and ordering functions can therefore be replaced
  independently.

Neither level accepts a protocol object, appends blocks or depends on a
sequence module. An
`Acquisition` merely carries numeric state: `lin_idx`, `par_idx`, absolute
`labels`, a relative EPI traversal, slice positions/frequencies and, for a
non-Cartesian scan, a rotation matrix.

## Which high-level factory should I call?

| Acquisition | Factory | Important knobs |
|---|---|---|
| 2D/3D GRE, including multiecho | `make_cartesian_sampling` | `train_length=1`, matrix, acceleration, frames/slices |
| 2D/3D FSE | `make_cartesian_sampling` | `train_length=ETL`, FSE ordering |
| 2D/3D MPRAGE or segmented GRE | `make_cartesian_sampling` | segment length, `ordering="centric"` |
| 2D/3D EPI, structural or fMRI | `make_epi_sampling` | acceleration, segments, CAIPI shift, frames |
| 2D radial/spiral/rosette | `make_noncartesian_2d_sampling` | views per frame, slices, frames, angle scheme |
| 3D stack of trajectories | `make_noncartesian_stack_sampling` | views per partition, partition order, frames |
| 3D projection | `make_noncartesian_projection_sampling` | views per frame, direction scheme, frames |

The readout's echo count is not a sampling dimension. GRE and multiecho GRE
therefore share the same plan; `num_echoes` belongs to
`make_line_readout`.

## Cartesian GRE, FSE and MPRAGE

The same factory covers these families because their difference at the plan
level is how many encoded locations occur inside one excitation.

```python
import pulserver.pypulseq as pp

gre2d = pp.make_cartesian_sampling(
    (128, 128), acceleration=2, calibration=16,
    num_slices=32, slice_spacing_m=5e-3,
)
gre3d = pp.make_cartesian_sampling(
    (128, 128, 64), acceleration=(2, 2), calibration=(16, 8),
)
fse2d = pp.make_cartesian_sampling(
    (128, 128), train_length=16, ordering="radial",
    num_slices=32, slice_spacing_m=5e-3,
)
fse3d = pp.make_cartesian_sampling(
    (128, 128, 64), train_length=32, ordering="radial_adaptive",
)
mprage2d = pp.make_cartesian_sampling(
    (256, 192), train_length=96, ordering="centric",
    num_slices=24, slice_spacing_m=5e-3,
)
mprage3d = pp.make_cartesian_sampling(
    (256, 192, 160), acceleration=(2, 2),
    train_length=128, ordering="centric",
)
```

The loop consumes only numeric entry properties:

```python
for acquisition in fse3d:
    readout.set_state(
        lin_idx=acquisition.lin_idx,
        par_idx=acquisition.par_idx,
    )
    for block in readout:
        seq.add_block(*block)
```

For 2D plans, the slice/SMS schedule is deliberately separate from the
Cartesian ``(ky, kz)`` mask and its echo-train ordering. Its rows select one
physical slice for conventional imaging or ``sms_factor`` simultaneous bands;
it covers every requested position once. Slice spacing fixes the physical
positions, and the selective-gradient amplitude later converts them to RF
frequency offsets:

```python
plan = pp.make_cartesian_sampling(
    (128, 128), num_slices=48, slice_spacing_m=3e-3, sms_factor=3,
)
frequency_table_hz = plan.slice_frequency_table_hz(gz.amplitude)

for acquisition in plan:
    offsets_hz = acquisition.frequency_offsets_hz(gz.amplitude)
```

## EPI: structural and multiple-volume

`frames=1` is structural; larger values produce the repeated volume loop used
by fMRI. In 2D, slice groups sit inside each frame. In 3D, each frame directly
covers `(ky, kz)`.

```python
epi2d = pp.make_epi_sampling(
    (96, 96), acceleration=2, segments=1,
    frames=200, num_slices=48, slice_spacing_m=3e-3,
)
epi3d = pp.make_epi_sampling(
    (96, 96, 32), acceleration=(2, 2), segments=2,
    caipi_shift=1, frames=100,
)
```

An EPI plan may contain more than one relative traversal. Build one reusable
readout per plan shot, then use the absolute entry state for each volume:

```python
readouts = [
    pp.make_epi_readout(
        system, fov, matrix, epi3d.sampling.n_shots,
        epi3d.sampling.relative(shot),
    )
    for shot in range(epi3d.sampling.n_shots)
]

for acquisition in epi3d:
    readout = readouts[acquisition.shot]
    readout.set_state(
        lin_idx=acquisition.lin_idx,
        par_idx=acquisition.par_idx,
        labels=acquisition.labels,
    )
    for block in readout:
        seq.add_block(*block)
```

## Non-Cartesian structural and dynamic plans

The 2D factory applies to any rotated in-plane base trajectory, not only
radial. Golden/RAGA angles continue chronologically across slice and frame by
default, so different slices and dynamic windows see different tilts. Set
`continuous=False` when every outer location must replay an identical set.

```python
inplane = pp.make_noncartesian_2d_sampling(
    (192, 192), views_per_frame=300, frames=20,
    num_slices=24, slice_spacing_m=5e-3,
    scheme="tiny_golden", tiny_index=2,
)

for acquisition in inplane:
    readout.set_state(
        lin_idx=acquisition.view,
        rotation=acquisition.rotation,
    )
```

Stacks add a Cartesian partition index while allowing the in-plane angle to
continue across both partition and frame:

```python
stack = pp.make_noncartesian_stack_sampling(
    (192, 192, 64), views_per_partition=200, frames=10,
    partition_order="center_out", scheme="golden",
)

for acquisition in stack:
    readout.set_state(
        lin_idx=acquisition.view,
        par_idx=acquisition.par_idx,
        rotation=acquisition.rotation,
    )
```

Projection plans provide full 3D rotations whose directions vary in both
azimuth and polar angle:

```python
projection = pp.make_noncartesian_projection_sampling(
    (192, 192, 192), views_per_frame=20_000,
    frames=10, scheme="golden_means",
)

for acquisition in projection:
    readout.set_state(
        lin_idx=acquisition.view,
        rotation=acquisition.rotation,
    )
```

## Lower-level mask and ordering composition

Use this layer when a built-in high-level sampling factory is almost right but
one stage must be replaced. Build the `SamplingPattern` directly, then nest it
under whatever outer loops your sequence needs — plain `for` loops over the
non-imaging dimensions (frames, contrasts, averages) are the natural way to
write this, with no helper required to flatten them.

```python
import numpy as np

mask = pp.make_poisson_disc_mask(
    (128, 64), 4.0, calib=(16, 8), seed=0,
)
sampling = pp.SamplingPattern.from_mask(
    mask, train_length=16, ordering="radial",
)
for frame in range(3):
    for contrast in range(2):
        for shot in sampling:
            pass  # set the readout state from `shot`, `frame`, `contrast`
```

The stages remain independently available:

- Ordering functions (`make_linear_order`, `make_centric_order`,
  `make_radial_order`, `make_radial_adaptive_order`, `make_shuffling_order`)
  accept coordinates or a boolean mask and return support indices grouped by
  train.
- Mask functions (`make_random_mask`, `make_poisson_disc_mask`,
  `make_caipirinha_mask`) return boolean
  masks only.
- Individual tilt generators (`calc_uniform_angles`, `calc_golden_angles`,
  `calc_tiny_golden_angles`, `calc_raga_angles`) return flat angle arrays.
- Low-level EPI ordering (`make_skipped_caipi_order`) and tilt generators
  (`make_radial_tilt`, `make_golden_means_3d_tilt`,
  `make_spiral_phyllotaxis_tilt`) expose support and ordering without adding
  frames or slices.

## Relative EPI traversals

`SamplingPattern.from_relative_shifts` is the escape hatch for EPTI, zigzag
or another custom blipped train:

```python
sampling = pp.SamplingPattern.from_relative_shifts(
    starts=[[8, 2], [9, 2]],
    shifts=[[[0, 0], [2, 1], [4, 2]],
            [[0, 0], [2, -1], [4, -2]]],
    shape=(32, 8),
)
for frame in range(20):
    for shot in sampling:
        pass  # configure one EPI readout from `frame` and `shot`
```

## Slice grouping

`make_slice_sampling` remains useful when building a custom loop. Its
selection mask is over physical slices, not phase encodes, and its ordering
therefore never changes a Cartesian echo-train order:

```python
slice_sampling = pp.make_slice_sampling(
    48, spacing_m=3e-3, order="interleaved", sms_factor=3,
)
```

## References

- Stirnberg et al., segmented skipped-CAIPI, DOI `10.1002/mrm.28486`.
- Dong et al., EPTI encoding, DOI `10.1002/mrm.28295`.
- Scholand et al., RAGA, DOI `10.1002/mrm.30254`, arXiv `2401.02892`.
- Piccini et al., spiral phyllotaxis, DOI `10.1002/mrm.22898`.
- SigPy `sigpy.mri.samp.poisson`, BSD 3-Clause License.
