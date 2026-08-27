# Cache

A binary sidecar beside the `.seq`, holding what is expensive to recompute —
the deduplication tables, the segmentation, the execution stream. Its
extension is vendor-selectable through `pulseg_opts.cache_ext`: `.pseg` by
default, `.pge` on GE.

The reconstruction side reads the `.seq` directly and never touches this.

## Writing

`pulseg_save_cache` derives the cache path from the sequence path and the
extension in `opts`, and reads the integrity size off the `.seq` itself — so
it pairs with the loaders below, which locate the cache the same way. A caller
keeping its cache elsewhere supplies the path and the size itself.

````{only} doxygen
```{doxygenfunction} pulseg_save_cache
:project: pulserver_c
```

```{doxygenfunction} pulseg_save_cache_to_path
:project: pulserver_c
```

```{doxygenfunction} pulseg_clear_cache
:project: pulserver_c
```
````

## Reading, by section

The per-section loaders exist so a consumer pays only for what it reads. The
sections that scale with the length of the scan are the ones worth not
reading, and the waveform-generation pass does not need them: everything it
resolves per position is frozen into the segment definitions at parse time.

| Loader | Sections | Used by |
|---|---|---|
| `pulseg_load_geninstructions_cache` | `COMMON`, `SHAPES` | waveform generation ({doc}`generation`) |
| `pulseg_load_scanloop_cache` | `COMMON`, `INSTANCES`, `ROTATIONS`, `SHAPES`, `SCANLOOP` | playout ({doc}`playout`) |
| `pulseg_load_cache` | all, at an explicit path | a caller managing its own cache |

````{only} doxygen
```{doxygenfunction} pulseg_load_geninstructions_cache
:project: pulserver_c
```

```{doxygenfunction} pulseg_load_scanloop_cache
:project: pulserver_c
```

```{doxygenfunction} pulseg_load_cache
:project: pulserver_c
```
````

## A vendor's own section

An extension point: an opaque blob written after the base sections and read
back verbatim, for whatever a platform needs to carry alongside the structure
and this library has no opinion about. Leaving the callback unset writes no
section, which is not an error.

````{only} doxygen
```{doxygenfunction} pulseg_write_vendor_cache_section
:project: pulserver_c
```

```{doxygenfunction} pulseg_read_vendor_cache_section
:project: pulserver_c
```
````

## See also

{doc}`file` produces the collection this stores; {doc}`generation` and
{doc}`playout` are the two stages that read it back.
