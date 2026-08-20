# Safety

The gate a collection passes before the scanner plays it: gradient amplitude
and slew, continuity across every block boundary, acoustic resonance, and
peripheral nerve stimulation.

```c
#include "pulseg/pulseg.h"
#include "pulseg/pulseg_pns_models.h"

pulseg_pns_irnich context;
pulseg_pns_model model;
pulseg_pns_irnich_init(&model, &context, 360.0f, 20.0f, 1.0f);

int code = pulseg_check_safety(
    coll, &diag, &opts, num_bands, bands, &model, 80.0f);
```

````{only} not doxygen
```{note}
The reference below is generated from the headers by Doxygen, which is not
installed in this build. Everything else on this page is unaffected;
`apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## What is checked, and what is not

**RF safety is deliberately absent.** SAR limits are vendor-proprietary, so
this library reports the per-pulse summary through `pulseg_get_rf_stats` and
leaves the limit to whoever owns it.

**The PNS model is injected.** `pulseg_pns_model` is an interface; the two
published forms ship with it and a vendor's own can be passed in the same way.
Only the per-scanner coefficients are proprietary, and none are distributed
here — both shipped models are initialised from numbers the caller supplies.

**These verdicts are estimates that run before the scanner's own.** They never
replace a predownload gate or a hardware monitor. What they buy is finding the
violation at design time rather than at the console.

## The one call

`pulseg_check_safety` runs everything and returns at the first violation with
a diagnostic naming it. The TR waveforms are extracted once and shared between
the acoustic and PNS checks, which is most of why one call is cheaper than
four.

````{only} doxygen
```{doxygenfunction} pulseg_check_safety
```
````

## Asked one at a time

Continuity is the one question a design tool can ask offline, with no PNS
model and no band table: a step larger than `max_slew * grad_raster` between
adjacent raster points, per axis in the physical frame, or a subsequence
ending without ramping to zero.

````{only} doxygen
```{doxygenfunction} pulseg_check_grad_continuity
```
````

The other two return the spectra and the slew-rate waveforms themselves,
rather than a verdict — which is what a plot needs, and what makes a
borderline sequence explainable instead of merely rejected.

````{only} doxygen
```{doxygenfunction} pulseg_calc_mech_resonances
```

```{doxygenfunction} pulseg_mech_resonances_spectra_free
```

```{doxygenfunction} pulseg_calc_pns
```

```{doxygenfunction} pulseg_pns_result_free
```
````

````{only} doxygen
```{doxygenstruct} pulseg_forbidden_band
:members:
```

```{doxygenstruct} pulseg_mech_resonances_spectra
:members:
```

```{doxygenstruct} pulseg_pns_result
:members:
```
````

## Nerve models

````{only} doxygen
```{doxygenstruct} pulseg_pns_model
:members:
```
````

**Irnich / den Boer** is the rheobase-chronaxie `c/(c+tau)^2` kernel. It is
linear and time-invariant, so it publishes its impulse response and the safety
core may take its memoized per-shape route — on a long canonical TR, one to
two orders of magnitude cheaper than convolving the whole window.

````{only} doxygen
```{doxygenfunction} pulseg_pns_irnich_init
```

```{doxygenstruct} pulseg_pns_irnich
:members:
```
````

**SAFE** is Hebrank and Gebhardt's three-branch filter, in the implementation
Witzel and Szczepankiewicz published. Its branches rectify on both sides of
their lowpass, so it is not a convolution and deliberately publishes no
kernel.

````{only} doxygen
```{doxygenfunction} pulseg_pns_safe_init
```

```{doxygenstruct} pulseg_pns_safe
:members:
```

```{doxygenstruct} pulseg_pns_safe_axis
:members:
```
````

Both `*_init` functions leave the model borrowing its context struct, which
has to outlive every call made through it.

## See also

{doc}`../../explanations/index` holds the physics these checks implement —
what the canonical TR is, why a compressed train is evaluated the way it is,
and where each model comes from. {doc}`../cpp/pulseg` is the same gate behind
RAII types.
