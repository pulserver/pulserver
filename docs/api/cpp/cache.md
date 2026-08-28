# Cache

`Collection::save_cache` derives the cache path from the sequence path and
`Opts::cache_ext`. `save_cache_to_path` takes an explicit path and an
integrity size.

`Collection::from_geninstructions_cache` and `from_scanloop_cache` are static
factories: they allocate the collection, so they cannot be members of an
existing one. Which sections each reads is on {doc}`../c/cache`.

`clear_cache` deletes the cache beside a sequence.

The C counterpart is {doc}`../c/cache`.

## Reference

````{only} doxygen
```{doxygenfunction} pulseg::clear_cache
```

```{doxygenfunction} pulseg::Collection::from_geninstructions_cache
```

```{doxygenfunction} pulseg::Collection::from_scanloop_cache
```

```{doxygenfunction} pulseg::Collection::save_cache
```

```{doxygenfunction} pulseg::Collection::save_cache_to_path
```

```{doxygenfunction} pulseg::Collection::load_cache
```
````
