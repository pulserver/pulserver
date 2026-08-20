# pulseq (C++)

The standalone sequence library: a `.seq` read, built, analysed and written,
with nothing here knowing about Python and nothing here knowing about
`pulseg`. Together with `src/c/pulseq` underneath it, these are the two halves
of a Pulseq implementation that can be lifted out on their own.

```cpp
#include "pulseq.hpp"

pulseq::Raster raster;
raster.rf_us    = 1.0f;
raster.grad_us  = 4.0f;
raster.block_us = 10.0f;

pulseq::File file = pulseq::File::read("scan.seq", raster);
```

````{only} not doxygen
```{note}
The reference below is generated from the headers by Doxygen, which is not
installed in this build. Everything else on this page is unaffected;
`apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## Sequence

The model PyPulseq keeps as a `Sequence`, held the way a large scan needs it
held. The contents are the same — every library here is one the file format
defines — but each is a dense array of fixed-width rows rather than a
dictionary of tuples, because a three-dimensional protocol has millions of
blocks and an object per row is the whole cost of building one.

Gradients are the visible consequence: a trapezoid is five numbers and an
arbitrary waveform is a shape reference and five more, and the format numbers
both in one sequence. They are stored in two fixed-width tables with the
shared id mapped onto whichever row is real, so a sequence written out is
numbered exactly as PyPulseq would have numbered it and the split never
reaches the file.

Registering an event returns what it was registered as and whether an
identical one was already there, which is where deduplication happens: a scan
that registers a waveform per shot keeps far fewer than it registers, so a
shape is appended as it stands and `ShapeLibrary::compress` runs once at the
end rather than per registration.

````{only} doxygen
```{doxygenclass} pulseq::Sequence
:members:
```
````

````{only} doxygen
```{doxygenstruct} pulseq::Block
:members:
```

```{doxygenclass} pulseq::Definition
:members:
```

```{doxygenstruct} pulseq::SoftDelay
:members:
```

```{doxygenclass} pulseq::ShapeLibrary
:members:
```

```{doxygenclass} pulseq::RaggedTable
:members:
```

```{doxygenclass} pulseq::BasicTable
:members:
```
````

## Events

What a block is made of. PyPulseq hands back a `SimpleNamespace` and reading
one costs a dictionary lookup per field — fine once, not fine two million
times — so these are structs with the fields in slots.

````{only} doxygen
```{doxygenstruct} pulseq::Event
:members:
```

```{doxygenstruct} pulseq::RfEvent
:members:
```

```{doxygenstruct} pulseq::TrapEvent
:members:
```

```{doxygenstruct} pulseq::GradEvent
:members:
```

```{doxygenstruct} pulseq::AdcEvent
:members:
```

```{doxygenstruct} pulseq::DelayEvent
:members:
```

```{doxygenstruct} pulseq::LabelEvent
:members:
```

```{doxygenstruct} pulseq::TriggerEvent
:members:
```

```{doxygenstruct} pulseq::RotationEvent
:members:
```

```{doxygenstruct} pulseq::SoftDelayEvent
:members:
```

```{doxygenstruct} pulseq::Registration
:members:
```
````

## Files

`File` and `FileSet` are RAII over the C reader — one file, and a chain
followed to its end. `SequenceFile` is the lazy view beside them: a consumer
that needs one section should not pay for the whole file, and the dominant
case is the reconstruction side reading `[DEFINITIONS]` alone.

````{only} doxygen
```{doxygenclass} pulseq::File
:members:
```

```{doxygenclass} pulseq::FileSet
:members:
```

```{doxygenclass} pulseq::SequenceFile
:members:
```
````

````{only} doxygen
```{doxygenfunction} pulseq::read_file
```

```{doxygenfunction} pulseq::write_text
```

```{doxygenfunction} pulseq::write_binary
```

```{doxygenfunction} pulseq::required_revision
```

```{doxygenfunction} pulseq::path_dirname
```

```{doxygenfunction} pulseq::path_join
```
````

Both writers take the sequence by non-const reference, because both record
`TotalDuration` in `[DEFINITIONS]` before they start.

## Shapes

Pulseq's codec: run-length encoding of the derivative, which is chosen for
what MR waveforms actually look like — a constant run and a linear ramp are
each a handful of numbers.

````{only} doxygen
```{doxygenfunction} pulseq::compress_shape
```

```{doxygenfunction} pulseq::decompress_shape(const double*, int, int)
```

```{doxygenfunction} pulseq::decompress_shape(const pulseq_shape&, float)
```

```{doxygenstruct} pulseq::Shape
:members:
```
````

## k-space

Where every ADC sample sits. The arithmetic is a native pass over the
`Sequence`: each distinct gradient is integrated once and the blocks are
walked over the result, so the cost follows the number of distinct gradients
rather than the length of the scan.

````{only} doxygen
```{doxygenfunction} pulseq::calculate_kspace
```

```{doxygenstruct} pulseq::KSpace
:members:
```

```{doxygenstruct} pulseq::KSpaceOptions
:members:
```

```{doxygenstruct} pulseq::Readout
:members:
```

```{doxygenstruct} pulseq::RfEventTiming
:members:
```

```{doxygenfunction} pulseq::block_k_origins
```
````

### Base trajectories

The k-space a block's ADC window traverses with the amplitude left outside it,
so one stored curve serves every instance that plays the same shape at a
different scale.

````{only} doxygen
```{doxygenstruct} pulseq::AxisTrajectory
:members:
```

```{doxygenstruct} pulseq::BaseTrajectory
:members:
```

```{doxygenfunction} pulseq::base_trajectory
```

```{doxygenfunction} pulseq::absolute_trajectory
```

```{doxygenfunction} pulseq::attach_base_trajectory
```

```{doxygenfunction} pulseq::has_base_trajectory
```

```{doxygenfunction} pulseq::read_base_trajectory
```

```{doxygenfunction} pulseq::active_axes
```

```{doxygenfunction} pulseq::interleave_readout
```
````

### Field of view

````{only} doxygen
```{doxygenfunction} pulseq::apply_fov_scale
```

```{doxygenfunction} pulseq::apply_fov_shift
```

```{doxygenenum} pulseq::FovShiftScope
```
````

## Moments and the b-tensor

Twenty-seven numbers a shot rather than a waveform: the diffusion b-tensor and
the gradient moments, one entry per excitation.

````{only} doxygen
```{doxygenfunction} pulseq::calc_moments
```

```{doxygenfunction} pulseq::compose_btensor
```

```{doxygenfunction} pulseq::gradient_moment
```

```{doxygenfunction} pulseq::identity3
```

```{doxygenstruct} pulseq::Moments
:members:
```

```{doxygenstruct} pulseq::MomentsOptions
:members:
```

```{doxygenstruct} pulseq::BTensorParts
:members:
```
````

## Labels

A `.seq` this project did not author usually carries no `LABELSET` extensions,
so nothing downstream knows which line, partition or slice an acquisition
belongs to. These recover the counters from the sequence's own trajectory,
which is the only source that cannot disagree with what was played.

````{only} doxygen
```{doxygenfunction} pulseq::detect_labels
```

```{doxygenfunction} pulseq::auto_label
```

```{doxygenfunction} pulseq::apply_labels
```

```{doxygenfunction} pulseq::label_gate_runs
```

```{doxygenfunction} pulseq::label_id_for_name
```

```{doxygenfunction} pulseq::hint_id_for_name
```
````

````{only} doxygen
```{doxygenstruct} pulseq::AutoLabelOptions
:members:
```

```{doxygenstruct} pulseq::AutoLabelResult
:members:
```

```{doxygenstruct} pulseq::AutoLabels
:members:
```

```{doxygenstruct} pulseq::AutoLabelAux
:members:
```

```{doxygenstruct} pulseq::RepeatDim
:members:
```

```{doxygenenum} pulseq::SliceSorting
```
````

## Expansion

A Pulseq file describes one pass, and playing it N times is left to the
interpreter — which means the playing order is written nowhere in the file.
This does the arithmetic and writes the answer down: after `expand_repeats`
the block table *is* the scan.

````{only} doxygen
```{doxygenfunction} pulseq::expand_repeats
```

```{doxygenstruct} pulseq::ExpandOptions
:members:
```

```{doxygenstruct} pulseq::ExpandResult
:members:
```
````

## Types and errors

````{only} doxygen
```{doxygenstruct} pulseq::Raster
:members:
```

```{doxygenenum} pulseq::Axis
```

```{doxygenenum} pulseq::EventKind
```

```{doxygenenum} pulseq::RfUse
```

```{doxygenenum} pulseq::GradientType
```

```{doxygenenum} pulseq::GradKind
```

```{doxygenenum} pulseq::ExtensionType
```

```{doxygenenum} pulseq::LabelId
```

```{doxygenenum} pulseq::HintId
```

```{doxygenenum} pulseq::TriggerType
```

```{doxygenenum} pulseq::TriggerChannelInput
```

```{doxygenenum} pulseq::TriggerChannelOutput
```
````

````{only} doxygen
```{doxygenclass} pulseq::Error
:members:
```

```{doxygenfunction} pulseq::check
```

```{doxygenfunction} pulseq::error_message
```
````

## Double precision

`src/c/pulseq` is `float` end to end and deliberately so: it is the raw model
of what a file stores. An analysis that needs more compiles the same sources a
second time at double precision, scoped inside `pulseq::raw64`, so both
instantiations can be linked into one program.

## See also

{doc}`../c/pulseq` is the reader underneath. {doc}`../python/pypulseq` is the
Python event layer over the same format, and is where a design script lives.
