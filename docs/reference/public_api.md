# pulserverlib Public C API Reference

This document describes the public surface of the pulserverlib C library
that vendor integrations and downstream consumers may rely on, the
optional / opt-in modules controlled by build flags, and the documented
extension points for vendor-specific behaviour.

The accompanying explanatory documents live under
[../explanations/](../explanations/).

## 1. Public headers

The complete public API is contained in five headers under
[`csrc/`](../../csrc/):

| Header | Purpose |
| --- | --- |
| `pulseqlib_config.h` | Vendor-id constants (`PULSEQLIB_VENDOR_*`), allocator macros, build-time toggles |
| `pulseqlib_types.h` | Public structs and `*_INIT` initialisers (collection, opts, rf_stats, pns_params, mech-resonance spectra, …) |
| `pulseqlib_protocol.h` | UI / parameter protocol IDs (mirrors `python/UIParam`) |
| `pulseqlib_methods.h` | All public function declarations |
| `pulseqlib_bridge.h` | Minimal C++ bridge for `_pulseqlib_wrapper.cpp` |

`pulseqlib_internal.h` is **not** part of the public API. It contains
build-internal macros, in-progress structs, and helpers that may change
without notice.

## 2. Build-time opt-out flags

Both flags are defined in [`csrc/CMakeLists.txt`](../../csrc/CMakeLists.txt):

| CMake option | Default | What it controls |
| --- | --- | --- |
| `PULSEQLIB_BUILD_CACHE` | `ON` | Compiles `pulseqlib_cache.c` (binary cache I/O for sequence descriptors and trajectories). |
| `PULSEQLIB_BUILD_SEQDESC` | `OFF` | Compiles `pulseqlib_seqdesc.c` and `pulseqlib_cache_seqdesc.c` (sequence-descriptor metadata used by pge / Python; the PSD amalgamation does not need it). |

The pge Python wheel turns `PULSEQLIB_BUILD_SEQDESC` `ON` because it
exposes `pulseqlib_get_sequence_description` / `_parameters` to Python.
The PSD amalgamation (`pulserver-interpreter/psd/pulserverlib_common.c`)
does not include those translation units.

## 3. Recommended workflow for vendor integrations

A third-party vendor wrapping pulserverlib (e.g. inside a Pulseq
ExternalSequence-style interpreter) does not need any vendor-specific
entrypoints. The standard path is:

1. **Build a collection** from the `.seq` file(s):
   `pulseqlib_read(...)` (filesystem) or
   `pulseqlib_read_from_buffers(...)` (in-memory bytes). Pass the
   vendor's `pulseqlib_opts` (gradient/slew limits, rasters,
   `opts.vendor`).

2. **Run the safety check** on the collection:
   ```c
   pulseqlib_check_safety(coll, &diag, &opts,
                          num_forbidden_bands, forbidden_bands,
                          &pns_params, pns_threshold_percent);
   ```
   This validates max gradient amplitude, gradient continuity, max slew
   rate, forbidden mechanical-resonance bands (structural / analytical
   harmonic analysis of the canonical TR), and PNS against the vendor's
   model (selected by `pns_params.vendor`). It does **not** perform an
   RF / SAR safety check — RF safety is vendor-proprietary; downstream
   consumers retrieve the per-pulse summary via
   `pulseqlib_get_rf_stats()` / `pulseqlib_get_rf_array()` and apply
   their own scanner-specific limits (e.g. on the GE PSD side).

3. **Discard the collection**: `pulseqlib_collection_free(coll)`.

The cache (`pulseqlib_cache.c`) is purely an acceleration aid — disable
it via `cache_binary=0` or the `PULSEQLIB_BUILD_CACHE=OFF` build flag if
the integration is one-shot and the on-disk artefact is unwanted.

### 3a. Iterator-based safety entrypoint (no collection required)

Vendors that don't want to manage a `pulseqlib_collection*` lifetime
can use the one-shot facade `pulseqlib_check_safety_from_file()`. It
takes a `.seq` file path and the same `opts` / forbidden-bands /
`pns_params` triple as `pulseqlib_check_safety()`, and internally:

1. calls `pulseqlib_read()` (no cache, no signature verification, no
   label parsing);
2. runs `pulseqlib_check_safety()` with the caller-supplied limits;
3. frees the collection before returning.

```c
pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
int rc = pulseqlib_check_safety_from_file(
    &diag, "scan.seq", &opts,
    num_forbidden_bands, forbidden_bands,
    &pns_params, pns_threshold_percent);
```

The performed checks are exactly those of `pulseqlib_check_safety()`:
gradient continuity, max gradient amplitude, max slew rate, structural
mechanical-resonance forbidden bands, and PNS thresholding under the
vendor model selected by `pns_params.vendor`.

A worked example ships under
[`examples/cexamples/safety_with_external_sequence.c`](../../examples/cexamples/safety_with_external_sequence.c).

For wrapper-side plotting (i.e. when a UI needs the sample-level
waveforms / spectra rather than just a pass/fail verdict), use the
collection-based getters:

- `pulseqlib_calc_pns()` — returns per-axis slew waveforms.
- `pulseqlib_calc_mech_resonances()` — returns the structural
  (analytical TR-harmonic) candidate set used by the safety check, plus
  an auxiliary full-TR FFT magnitude spectrum for display only. The
  full-TR spectrum is **not** consulted by `pulseqlib_check_safety`;
  the verdict comes from the structural candidates.

The structural-candidate numerics returned by
`pulseqlib_calc_mech_resonances` are bit-identical to those used inside
`pulseqlib_check_safety`.

## 4. Extension points

Two areas are designed to be extended for non-GE vendors *without*
breaking existing GE callers:

### 4a. PNS model dispatch

`pulseqlib_pns_params` carries an `int vendor` field. The GE Healthcare
exponential model (chronaxie / rheobase / alpha) is the only model
implemented in this library today. `calc_pns_from_uniform()` dispatches
on the `vendor` field:

| `vendor` value | Behaviour |
| --- | --- |
| `0` (unspecified) | Treated as GEHC (back-compat for callers that don't set the field) |
| `PULSEQLIB_VENDOR_GEHC` | Existing exponential PNS calculation |
| Any other value | Returns `PULSEQLIB_ERR_NOT_IMPLEMENTED` |

To add a new model (e.g. Siemens SAFE), add a new branch in
`calc_pns_from_uniform()` in `pulseqlib_safety.c` that selects an
alternative kernel / threshold metric. The output struct
(`pulseqlib_pns_result`) stays vendor-neutral (per-axis slew arrays).
If a future model requires additional parameters that don't fit the
existing struct, a sibling parameter struct can be added and selected
via the same `vendor` tag.

### 4b. RF-statistics dispatch

`pulseqlib_rf_stats` also carries an `int vendor` field. The current
fields (`abs_width`, `eff_width`, `duty_cycle`, `max_pulse_width`,
`bandwidth_hz`, `base_amplitude_hz`, `total_b1sq_power`, …) match the
GE Healthcare safety check inputs (concept analogous to sigpy's
`ge_rf_params`). The concept of an RF-pulse safety summary
generalises to other vendors (Siemens REFGRAD/MINSLICE/MAXSLICE,
Philips `am_c_*`), but the field semantics differ.

Today `compute_rf_stats()` in `pulseqlib_dedup.c` is invoked only when
`seq->opts.vendor == PULSEQLIB_VENDOR_GEHC`; the resulting struct is
stamped with `stats.vendor = seq->opts.vendor`. For non-GE vendors the
struct is left at `PULSEQLIB_RF_STATS_INIT` (vendor=0, all fields zero).

To add vendor-specific RF-stat computation:

1. Provide a sibling `compute_rf_stats_<vendor>()` populating the same
   `pulseqlib_rf_stats` struct (re-using physically meaningful fields)
   *or* a parallel struct selected via the `vendor` tag.
2. Dispatch on `seq->opts.vendor` in
   [`pulseqlib_dedup.c`](../../csrc/pulseqlib_dedup.c).
3. Downstream consumers (pge, PSD safety check) read `stats.vendor`
   to decide which interpretation applies — no recompilation of
   pulserverlib core needed for read-side consumers, since the struct
   shape is shared.

The cache binary format stamps the version (`PULSEQLIB_CACHE_VERSION`)
including the vendor field, so cache files are regenerated on upgrade.

## 5. Compatibility invariant

Modifications to the C library must keep the following three consumers
byte / behaviour-identical:

- the pge Python wrapper (`python/pge`);
- the PSD amalgamation (`pulserver-interpreter/psd/pulserverlib_common.c`);
- the C++ mrdserver (`mrdserver/csrc/`), which only reads cache files
  via `trajectory_cache_reader`.

Verify by running, in order:

```bash
# 1. pge
cd pulserverlib && conda run -n octave2 pip install -e . -q
conda run -n octave2 python -m pytest tests/python/ -q

# 2. ctests
cd pulserverlib/tests/ctests/build && cmake --build . -j && ./bin/run_tests

# 3. PSD
cd pulserver-interpreter && bash scripts/build_psd.sh --ese=MR30.1_R04 --clean
```
