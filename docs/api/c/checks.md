# Checks

What a collection has to pass before the scanner plays it: event times on the
rasters the hardware can address, gradient amplitude and slew inside the
amplifiers' limits, no step across a block boundary the amplifiers cannot
follow, no harmonic in an acoustic band the magnet resonates at, and a nerve
stimulation response under threshold.

```c
#include "pulseg/pulseg.h"
#include "pulseg/pulseg_pns_models.h"

pulseg_pns_irnich context;
pulseg_pns_model model;
pulseg_pns_irnich_init(&model, &context, 360.0f, 20.0f, 1.0f);

pulseg_forbidden_band_list bands = PULSEG_FORBIDDEN_BAND_LIST_INIT;
bands.count = num_bands;
bands.bands = band_array;

int code = pulseg_check_safety(coll, &diag, NULL, &opts, &bands, &model, 80.0f);
```

## What is checked, and what is not

**RF safety is deliberately absent.** SAR limits are vendor-proprietary, so
this library reports the per-pulse summary through `pulseg_get_rf_stats` (see
{doc}`generation`) and leaves the limit to whoever owns it.

**The PNS model is injected.** `pulseg_pns_model` is an interface; the two
published forms ship with it and a vendor's own can be passed the same way.
Only the per-scanner coefficients are proprietary, and none are distributed
here — both shipped models are initialised from numbers the caller supplies.

**These verdicts are estimates that run before the scanner's own.** They never
replace a predownload gate or a hardware monitor. What they buy is finding the
violation at design time rather than at the console.

## All of them, or one

`pulseg_check_safety` runs every check and returns at the first violation with
a diagnostic naming it.

Each check is also a public entry point, because a platform may enforce some of
them in hardware and want only the rest — a scanner with an acoustic monitor
but no nerve model, or the reverse. None of them assumes another has run, and
each is exactly the check `pulseg_check_safety` runs, not an approximation of
it.

````{only} doxygen
```{doxygenfunction} pulseg_check_safety
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_raster_alignment
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_max_grad
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_max_slew
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_grad_continuity
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_pns
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_mech_resonances
:project: pulserver_c
```
````

Amplitude and slew are compared as vector magnitudes of the unrotated
waveform, because that magnitude is what bounds every physical axis under an
arbitrary rotation. Slew and continuity are neighbours but not the same
question: slew bounds what one event demands, continuity bounds the step
*between* two adjacent events — including a subsequence that ends without
ramping to zero.

## Sharing the preprocessing

The PNS and mechanical-resonance checks both answer from uniform-raster
gradient waveforms extracted over a window of the canonical TR, and both need
the repetitions grouped by the set of shapes they play. That is the expensive
part, and a `pulseg_check_plan` holds it so it is not repeated.

Passing `NULL` instead is always allowed: the check then builds a plan for its
own use and destroys it, which costs one small allocation and needs no
lifecycle from the caller.

Where a plan pays is asking the same thing twice — a verdict followed by the
spectra behind it, or a check re-run against a different band table or PNS
threshold, neither of which changes the waveforms. It is worth being precise
about where it does *not*: the windows the PNS and mechanical-resonance checks
evaluate are different windows, so running those two back to back reuses the
shape grouping but not the extraction.

`cache_budget_kb` caps what a plan retains, which matters on an embedded
target: past the cap it drops what it has not used recently and re-extracts if
asked again. The verdict never depends on it.

````{only} doxygen
```{doxygenfunction} pulseg_check_plan_create
:project: pulserver_c
```

```{doxygenfunction} pulseg_check_plan_destroy
:project: pulserver_c
```

```{doxygenstruct} pulseg_check_plan_config
:project: pulserver_c
:members:
```
````

## The data behind a verdict

A refusal that only says *no* is hard to act on. These two return the spectra
and the slew-rate waveforms themselves, which is what a plot draws and what
makes a borderline sequence explainable instead of merely rejected. Pass the
same plan the check used and the extraction is not repeated.

````{only} doxygen
```{doxygenfunction} pulseg_calc_mech_resonances
:project: pulserver_c
```

```{doxygenfunction} pulseg_mech_resonances_spectra_free
:project: pulserver_c
```

```{doxygenstruct} pulseg_mech_resonances_request
:project: pulserver_c
:members:
```

```{doxygenfunction} pulseg_calc_pns
:project: pulserver_c
```

```{doxygenfunction} pulseg_pns_result_free
:project: pulserver_c
```

```{doxygenstruct} pulseg_mech_resonances_spectra
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_pns_result
:project: pulserver_c
:members:
```
````

## Nerve models

````{only} doxygen
```{doxygenstruct} pulseg_pns_model
:project: pulserver_c
:members:
```
````

**Irnich / den Boer** is the rheobase-chronaxie `c/(c+tau)^2` kernel. It is
linear and time-invariant, so it publishes its impulse response and the safety
core may take its memoized per-shape route — on a long canonical TR, one to
two orders of magnitude cheaper than convolving the whole window.

````{only} doxygen
```{doxygenfunction} pulseg_pns_irnich_init
:project: pulserver_c
```

```{doxygenstruct} pulseg_pns_irnich
:project: pulserver_c
:members:
```
````

**SAFE** is Hebrank and Gebhardt's three-branch filter, in the implementation
Witzel and Szczepankiewicz published. Its branches rectify on both sides of
their lowpass, so it is not a convolution and deliberately publishes no
kernel.

````{only} doxygen
```{doxygenfunction} pulseg_pns_safe_init
:project: pulserver_c
```

```{doxygenstruct} pulseg_pns_safe
:project: pulserver_c
:members:
```

```{doxygenstruct} pulseg_pns_safe_axis
:project: pulserver_c
:members:
```
````

Both `*_init` functions leave the model borrowing its context struct, which
has to outlive every call made through it.

A vendor's own model is caller-side code, so what a convolution model needs to
do its work is published too: `pulseg_conv_fft_plan` transforms a kernel once
and applies it to each gradient axis, instead of the model bringing its own
FFT.

````{only} doxygen
```{doxygenfunction} pulseg_conv_fft_plan_create
:project: pulserver_c
```

```{doxygenfunction} pulseg_conv_fft_plan_apply
:project: pulserver_c
```

```{doxygenfunction} pulseg_conv_fft_plan_free
:project: pulserver_c
```
````

## See also

{doc}`../../explanations/index` holds the physics these checks implement —
what the canonical TR is, why a compressed train is evaluated the way it is,
and where each model comes from. {doc}`types` defines the band list and the
limits every check is judged against.
