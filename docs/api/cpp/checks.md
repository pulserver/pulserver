# Checks

`check_safety` runs every check. `check_max_grad`, `check_max_slew`,
`check_raster_alignment`, `check_pns` and `check_mech_resonances` are the same
checks individually.

`check_grad_continuity` returns the diagnostic message instead of throwing, so
a caller can ask every question rather than stop at the first. Empty means
continuous.

`CheckPlan` is the shared preprocessing behind the PNS and mechanical-resonance
checks. It is optional everywhere it appears; omitting it keeps the work
private to the call. {doc}`../c/checks` states what it does and does not save.

`calc_mech_resonances` and `calc_pns` return the spectra and slew-rate
waveforms behind a verdict. Both take the same optional plan.

The C counterpart is {doc}`../c/checks`.

## Reference

````{only} doxygen
```{doxygenfunction} pulseg::Collection::check_safety
```

```{doxygenfunction} pulseg::Collection::check_max_grad
```

```{doxygenfunction} pulseg::Collection::check_max_slew
```

```{doxygenfunction} pulseg::Collection::check_raster_alignment
```

```{doxygenfunction} pulseg::Collection::check_grad_continuity
```

```{doxygenfunction} pulseg::Collection::check_pns
```

```{doxygenfunction} pulseg::Collection::check_mech_resonances
```

```{doxygenfunction} pulseg::Collection::calc_pns
```

```{doxygenfunction} pulseg::Collection::calc_mech_resonances
```

```{doxygenclass} pulseg::CheckPlan
:members:
```

```{doxygenstruct} pulseg::MechResonancesSpectra
:members:
```

```{doxygenstruct} pulseg::PnsResult
:members:
```
````
