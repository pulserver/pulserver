# pulseg Public C API Reference

This document describes the public surface of the pulseg C library
that vendor integrations and downstream consumers may rely on, the
optional / opt-in modules controlled by build flags, and the documented
extension points for vendor-specific behaviour.

The accompanying explanatory documents live under
[../explanations/](../explanations/).

## 1. Public headers

The complete public API is contained in the headers under
[`csrc/include/pulseg/`](../../csrc/include/pulseg/). Application code
should include the umbrella header, `pulseg.h`, which pulls in all of
the below.

A consumer that only needs to read raw Pulseq `.seq` files, with no IR
and no vendor concepts, can depend on
[`csrc/include/pulseq/`](../../csrc/include/pulseq/) alone: it is a
standalone module that includes zero pulseg headers, and pulseg depends
on it and never the reverse.

| Header | Purpose |
| --- | --- |
| `pulseg_errors.h` | `PULSEG_ERR_*` return codes and the `PULSEG_SUCCEEDED`/`PULSEG_FAILED` predicates |
| `pulseg_config.h` | Vendor-id constants (`PULSEG_VENDOR_*`), allocator macros, build-time toggles |
| `pulseg_types.h` | Public structs and `*_INIT` initialisers (collection, opts, rf_stats, pns_params, mech-resonance spectra, …) |
| `pulseg_protocol.h` | UI / parameter protocol IDs (mirrors `python/UIParam`) |
| `pulseg_io.h` | Pulseq `.seq` reader entry points and scan-time peek |
| `pulseg_convert.h` | Sequence-description (SEQDESC) derivation entry points |
| `pulseg_collection.h` | Collection load/free, diagnostics, getters, and block cursor |
| `pulseg_safety.h` | RF/gradient safety checks, mechanical-resonance spectra, and PNS |
| `pulseg_trajectory.h` | TR gradient/waveform extraction and k-space trajectory computation |
| `pulseg_cache.h` | Binary (`.pge`) cache read/write |
| `pulseg_freqmod.h` | Frequency-modulation collection build/update/cache |
| `pulseg_bridge.h` | Minimal C++ bridge for `_pulseg_wrapper.cpp` |

The split is structural, not a convention: everything under
`csrc/include/` is public and everything under `csrc/src/` is private.
The private header `csrc/src/pulseg_internal.h` (internal structs,
cross-file `pulseg__` helpers) is reachable only by a target that opts
into `csrc/src` as an include directory, which the library itself, the C
tests and the in-tree vendor plug-in `src_gelib/` do. Its contents may
change without notice.

Symbol prefixes encode the same three tiers:

| Prefix | Meaning |
| --- | --- |
| `pulseg_` / `pulseq_` | public, declared under `csrc/include/` |
| `pulseg__` (double underscore) | private, shared across `.c` files, declared in `pulseg_internal.h` |
| unprefixed lowercase | file-static, never leaves its `.c` file |

## 2. Build

All modules are compiled unconditionally (see
[`csrc/CMakeLists.txt`](../../csrc/CMakeLists.txt)) — there are no
opt-out build flags. `pulseg_cache.c` (binary cache I/O for sequence
descriptors and trajectories) and `pulseg_seqdesc.c` /
`pulseg_cache_seqdesc.c` (sequence-descriptor metadata used by the
analysis and recon layers) are always part of the `pulseg` static
library.

## 3. Recommended workflow for vendor integrations

A third-party vendor wrapping pulseg (e.g. inside a Pulseq
ExternalSequence-style interpreter) does not need any vendor-specific
entrypoints. The standard path is:

1. **Build a collection** from the `.seq` file(s):
   `pulseg_read(...)` (filesystem) or
   `pulseg_read_from_buffers(...)` (in-memory bytes). Pass the
   vendor's `pulseg_opts` (gradient/slew limits, rasters,
   `opts.vendor`).

2. **Run the safety check** on the collection:
   ```c
   pulseg_check_safety(coll, &diag, &opts,
                          num_forbidden_bands, forbidden_bands,
                          &pns_params, pns_threshold_percent);
   ```
   This validates max gradient amplitude, gradient continuity, max slew
   rate, forbidden mechanical-resonance bands (structural / analytical
   harmonic analysis of the canonical TR), and PNS against the vendor's
   model (selected by `pns_params.vendor`). It does **not** perform an
   RF / SAR safety check — RF safety is vendor-proprietary; downstream
   consumers retrieve the per-pulse summary via
   `pulseg_get_rf_stats()` / `pulseg_get_rf_array()` and apply
   their own scanner-specific limits (e.g. on the GE PSD side).

3. **Discard the collection**: `pulseg_collection_free(coll)`.

The cache (`pulseg_cache.c`) is purely an acceleration aid — pass
`cache_binary=0` if the integration is one-shot and the on-disk
artefact is unwanted.

For wrapper-side plotting (i.e. when a UI needs the sample-level
waveforms / spectra rather than just a pass/fail verdict), use the
collection-based getters:

- `pulseg_calc_pns()` — returns per-axis slew waveforms.
- `pulseg_calc_mech_resonances()` — returns the structural
  (analytical TR-harmonic) candidate set used by the safety check, plus
  an auxiliary full-TR FFT magnitude spectrum for display only. The
  full-TR spectrum is **not** consulted by `pulseg_check_safety`;
  the verdict comes from the structural candidates.

The structural-candidate numerics returned by
`pulseg_calc_mech_resonances` are bit-identical to those used inside
`pulseg_check_safety`.

## 4. Extension points

Three seams let a vendor add proprietary behaviour without touching the
core: the PNS model, the RF-statistics hook, and the opaque VENDOR cache
section (`pulseg_opts.vendor_section_write_fn` /
`pulseg_read_vendor_cache_section()`).

### 4a. PNS model injection

PNS is not implemented in this library. `pulseg_check_safety()` takes a
`const pulseg_pns_model *`, a two-function-pointer interface the caller
supplies:

```c
typedef struct pulseg_pns_model
{
    void *ctx;
    int (*required_padding)(void *ctx, float dt_us);
    int (*evaluate)(void *ctx,
                    const float *dgdt_x, const float *dgdt_y, const float *dgdt_z,
                    int n, float dt_us,
                    float *out_x, float *out_y, float *out_z);
} pulseg_pns_model;
```

The core extracts uniform-raster dG/dt for the canonical TR, asks
`required_padding()` how many circularly-wrapped samples the model needs
to warm its filter, then calls `evaluate()` once per canonical TR with
arrays of that padded length. `evaluate()` returns per-axis values as a
percentage of the model's own threshold; the core compares them against
`pns_threshold_percent`. Passing `NULL` skips PNS entirely.

The GE Irnich rheobase-chronaxie model lives outside this library, in
[`src_gelib/pulserver_ge_pns.c`](../../../../src_gelib/pulserver_ge_pns.c) —
that file is the worked example for adding another vendor's model.

### 4b. RF-statistics hook

Flip angle, peak `|gamma*B1|`, `|B1|` integral and bandwidth are
vendor-neutral and always computed. Anything beyond that is not:
`pulseg_opts.vendor_rf_stats_fn` is an optional callback invoked at dedup
time with a read-only `pulseg_rf_view` of the pulse envelope, filling the
four opaque slots in `pulseg_rf_stats.vendor_stat[]`.

```c
int (*vendor_rf_stats_fn)(void *ctx, const pulseg_rf_view *rf, float out_stat[4]);
```

Leave it `NULL` and the slots stay zero; the core never interprets them.
The GE implementation (abs width, effective width, duty cycle, max pulse
width) is in
[`src_gelib/pulserver_ge_rf_stats.c`](../../../../src_gelib/pulserver_ge_rf_stats.c).

The cache stamps its format version (`PULSEG_CACHE_VERSION_MAJOR` /
`_MINOR`), so cache files are regenerated rather than misread on upgrade.

## 5. Compatibility invariant

Modifications to the C library must keep the following three consumers
byte / behaviour-identical:

- the Python wrapper (`python/pulserver`);
- the PSD amalgamation (`pulserver-interpreter/src_psd/pulserver_amalg.c`,
  fed by the `src_psd/pulseglib-src/` symlink farm);
- the C++ recon reader (`cxx/recon/trajectory_cache_reader.{h,cpp}`),
  which only reads cache files.

Verify by running, in order:

```bash
# 1. C unit tests + C89 conformance
bash scripts/check_c89_compliance.sh
bash scripts/build_and_run_ctests.sh

# 2. C++ tests (incl. the .pge truth fixtures) and Python
bash scripts/build_and_run_cpp_tests.sh
python -m pytest tests/python -q

# 3. PSD + full release build
cd ../.. && bash scripts/build.sh --pre-release && bash scripts/run_tests.sh
```
