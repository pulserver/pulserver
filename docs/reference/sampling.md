# Sampling plans

Pulserver's flat sampling API treats sampled locations (or trajectory tilts)
and their acquisition order as equally important. A `SamplingPattern` stores
the sampled `support`, a tuple of support-index arrays in `order`, and an
optional Cartesian `mask`. Indexing a pattern returns one shot in acquisition
order; variable train lengths are supported.

Sampling stays numeric and sequence-independent. Readout objects consume the
coordinates or rotation matrices, apply gradients, and emit Pulseq labels.
Outer-loop labels such as slice group and frame remain with the sequence loop.

## Cartesian and FSE

```python
import numpy as np
from pulserver import SamplingPattern
from pulserver.pypulseq import make_random_sampling

mask = make_random_sampling((128, 64), 4, calib=(16, 8), seed=0)
fse = SamplingPattern.from_mask(mask, train_length=16, ordering="radial")

# Absolute (LIN, PAR) coordinates for echo train 3.
coordinates = fse[3]
```

GRE and other acquisitions without an inner train use `train_length=1`.
Poisson-disc masks use the bundled deterministic C++ implementation and do
not require Numba.

## EPI and other relative-shift trains

```python
from pulserver import SamplingPattern

plan = SamplingPattern.from_relative_shifts(
    starts=[[8, 2], [9, 2]],
    shifts=[[[0, 0], [2, 1], [4, 2]],
            [[0, 0], [2, -1], [4, -2]]],
    shape=(32, 8),
)

absolute_labels = plan[0]
relative_gradient_positions = plan.relative(0)
consecutive_blips = plan.increments(0)
```

Configure an EPI readout with `relative_gradient_positions`, then pass
`absolute_labels` to `set_state(labels=...)`. `make_skipped_caipi_sampling` constructs the
segmented blipped-CAIPI lattice. `SamplingPattern.from_relative_shifts` is also the extension
point for reconstruction-specific EPTI and zigzag schedules; the package does
not embed a reconstruction optimizer.

## Non-Cartesian tilts and segmentation

```python
from pulserver.pypulseq import (
    make_golden_means_3d_sampling,
    make_radial_sampling,
    make_spiral_phyllotaxis_sampling,
)

radial = make_radial_sampling(1000, scheme="tiny_golden", tiny_index=2)
raga = make_radial_sampling(
    1000, scheme="raga", tiny_index=1, approximation_order=13
)
sphere = make_golden_means_3d_sampling(2000, segment_length=32)
phyllotaxis = make_spiral_phyllotaxis_sampling(2584, 34)

rotations = sphere.to_rotations()
```

The order tuple is the convenient continuous-gradient segment boundary. RAGA
keeps its finite equidistant angular support separate from the golden-like
temporal index order.

## Slice order, SMS, and dynamic outer dimensions

```python
from pulserver.pypulseq import make_outer_product, make_slice_groups

groups = make_slice_groups(
    48, spacing_m=3e-3, order="interleaved", sms_factor=3
)
offsets_hz = groups[0].frequency_offsets_hz(gradient_hz_per_m=42_000)

outer = make_outer_product(frame=range(20), slice_group=groups)
```

Each SMS `SliceGroup` retains its logical group index, physical slice indices,
positions, and per-band frequency offsets. The first `make_outer_product` dimension
changes slowest and the last changes fastest.

## References

- Stirnberg et al., segmented skipped-CAIPI, DOI `10.1002/mrm.28486`.
- Dong et al., EPTI encoding, DOI `10.1002/mrm.28295`.
- Dong et al., single-shot echo-planar time-resolved imaging (2024 reference
  PDF under the project `refcode/` directory).
- Scholand et al., RAGA, DOI `10.1002/mrm.30254`, arXiv `2401.02892`.
- Piccini et al., spiral phyllotaxis, DOI `10.1002/mrm.22898`.
- SigPy `sigpy.mri.samp.poisson`, BSD 3-Clause License.
