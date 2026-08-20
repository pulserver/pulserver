# pulseg (C++)

The scanner-side library with its handles owned and its error codes thrown.
One include, one namespace:

```cpp
#include "pulseg.hpp"
using namespace pulseg;

Opts opts;
opts.gamma_hz_per_t          = 42.576e6f;
opts.b0_t                    = 3.0f;
opts.max_grad_hz_per_m       = 42.576e6f * 0.040f;
opts.max_slew_hz_per_m_per_s = 42.576e6f * 150.0f;
opts.grad_raster_us          = 4.0f;
opts.rf_raster_us            = 1.0f;
opts.adc_raster_us           = 2.0f;
opts.block_raster_us         = 10.0f;

const char *buffer = data.data();
int         length = static_cast<int>(data.size());
Collection  collection(&buffer, &length, 1, opts);

auto waveforms = collection.get_tr_gradient_waveforms();
```

Everything `Collection` returns is a value: a `std::vector`, a struct of them,
or a `std::string`. Nothing has to be freed, and nothing can be freed twice —
which is most of what this layer is for.

````{only} not doxygen
```{note}
The reference below is generated from the headers by Doxygen, which is not
installed in this build. Everything else on this page is unaffected;
`apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## Collection

The whole surface hangs off this one class: reading, structure, events,
waveforms, the cursor, the safety gate. Every method forwards to the C entry
point of the same name, checks its code, and throws {cpp:class}`pulseg::Error`
on failure.

````{only} doxygen
```{doxygenclass} pulseg::Collection
:members:
```
````

## System limits

````{only} doxygen
```{doxygenstruct} pulseg::Opts
:members:
```

```{doxygenstruct} pulseg::ForbiddenBand
:members:
```
````

## What the analyses return

Value types over the C structs, so a caller keeps them, copies them and lets
them go out of scope.

````{only} doxygen
```{doxygenstruct} pulseg::ScanTimeInfo
:members:
```

```{doxygenstruct} pulseg::RfStats
:members:
```

```{doxygenstruct} pulseg::LabelLimit
:members:
```

```{doxygenstruct} pulseg::LabelLimits
:members:
```

```{doxygenstruct} pulseg::BlockInstance
:members:
```

```{doxygenstruct} pulseg::SegmentLayout
:members:
```
````

### Waveforms

````{only} doxygen
```{doxygenstruct} pulseg::GradAxisWaveform
:members:
```

```{doxygenstruct} pulseg::TrGradientWaveforms
:members:
```

```{doxygenstruct} pulseg::ChannelWaveform
:members:
```

```{doxygenstruct} pulseg::AdcEvent
:members:
```

```{doxygenstruct} pulseg::TrBlockDescriptor
:members:
```

```{doxygenstruct} pulseg::TrWaveforms
:members:
```
````

### Safety

````{only} doxygen
```{doxygenstruct} pulseg::MechResonancesSpectra
:members:
```

```{doxygenstruct} pulseg::PnsResult
:members:
```
````

The nerve model is still injected, as in C: `Collection::calc_pns` and
`Collection::check_safety` take a `pulseg_pns_model`, and the two published
forms are initialised through {doc}`../c/safety`'s `pulseg_pns_*_init`.

### Sequence description

````{only} doxygen
```{doxygenstruct} pulseg::SequenceDescription
:members:
```

```{doxygenstruct} pulseg::RfDefinitionShapes
:members:
```
````

## Errors

````{only} doxygen
```{doxygenclass} pulseg::Error
:members:
```

```{doxygenfunction} pulseg::check(int)
```

```{doxygenfunction} pulseg::check(int, const pulseg_diagnostic&)
```
````

## See also

{doc}`../c/pulseg` is the library underneath, and the one to read for what a
call actually does; this page is the ownership and error handling over it.
{doc}`../../examples/cpp/safety_only` is the smallest integration that
matters: a `.seq`, an `Opts`, a verdict.
