# AGENT.md — Domain Knowledge for AI Assistants

This file captures critical domain knowledge about the **pulserver**
project so that AI coding agents do not re-learn (or get wrong) the
same things.

---

## 1. Project Overview

**pulserver** is a cross-platform toolkit for working with Pulseq MRI
sequences (`.seq` files). Its core is a C89 library (`pulseqlib`) that
parses `.seq` files and performs **Segmentation** — identifying the
Repetition Time (TR) and splitting TRs into Segments. The library also
provides safety checks, acoustic analysis, PNS (Peripheral Nerve
Stimulation) evaluation, frequency modulation support, RF/gradient
statistics, and a block cursor/iterator for efficient playback.

High-level wrappers expose the C library to **MATLAB** (via MEX) and
**Python** (via pybind11 over a C++ RAII layer). Both wrappers surface
a `SequenceCollection` class with methods: `report()`, `check()`,
`plot()`, `pns()`, `grad_spectrum()`, `validate()`, etc.

Target vendor: **GE HealthCare** (compile-time `PULSEQLIB_VENDOR`).

---

## 2. Sequence Collection

A Pulseq "sequence" on disk may be a **chain** of `.seq` files linked by
a `next` definition key. The library reads the entire chain and
represents it as a **collection** of **subsequences**
(`pulseqlib_collection` containing N `pulseqlib_sequence_descriptor`s).

Each subsequence is independently parsed, deduped, and segmented. The
collection sums durations, offsets ADC/segment/block indices, and
exposes global iteration via the block cursor.

Serialisation helpers (`serialize` / `deserialize` in both MATLAB and
Python) write and read these linked `.seq` chains.

---

## 3. Processing Pipeline

For each subsequence the library executes (in order):

1. **Event deduplication** — RF, gradient, and ADC events are deduped
   by timing-only rows (see §4). Block definitions are deduped the same
   way. Gradient shot indices and statistics are computed.
2. **Prep / cooldown detection** — ONCE flags are parsed to determine
   prep block count and cooldown block count (see §5).
3. **Multipass folding** — If ONCE flags appear mid-sequence, the block
   table is folded to a single period (see §5).
4. **TR identification** — `first_repeating_segment()` finds the
   shortest repeating period in the **imaging region** block-ID pattern.
   Pattern detection **always** runs on block definition IDs. It is NOT
   triggered by ONCE flags (see §6).
5. **Scan table construction** — Maps block indices through prep ×
   main × cooldown regions, expanded for `num_averages`.
6. **Segmentation** — A state machine splits TRs into segments based
   on RF/ADC boundaries and gradient continuity (see §7).
7. **Segment timing** — RF and ADC anchors (isocenters, k-zero
   crossings) are computed per segment.
8. **Freq mod flags** — Blocks with (RF or ADC) + gradient are flagged
   for later frequency modulation building (see §9).
9. **Label table** — (GEHC only) A dry-run walk builds a per-ADC label
   table from LABELSET/LABELINC extension chains.
10. **Consistency + safety checks** — See §8.

---

## 4. Block Definition Model

### 4.1 Deduplication rows

| Event type | Dedup columns (timing only) | Per-instance scalars |
|---|---|---|
| **RF** | mag_shape_id, phase_shape_id, time_shape_id, delay | amplitude, freq_offset, phase_offset |
| **Gradient** | type, rise/flat/fall (trap) or num_samples+time_shape_id (arb), delay | amplitude |
| **ADC** | num_samples, dwell_time, delay | freq_offset, phase_offset |
| **Block** | `[block_duration, rf_def_id, gx_def_id, gy_def_id, gz_def_id]` | — |

**ADC and extensions (labels, rotations, flags) are NOT part of the
block definition ID.**  Two blocks with the same RF/gradient timing but
different ADC events, labels, or rotation matrices get the **same**
definition ID.

### 4.2 What does NOT affect block definition IDs

These are per-instance overrides applied at playback time:

- `rf.freqOffset` / `rf.phaseOffset` — slice-select frequency, RF spoiling, bSSFP phase alternation
- `adc.freqOffset` / `adc.phaseOffset`
- `scaleGrad` — phase-encode amplitude
- Gradient amplitude (only the normalized shape matters for the definition)
- All labels and flags (ONCE, NAV, NOROT, NOPOS, PMC, SLC, LIN, etc.)
- Rotation matrices / triggers / digital-out

### 4.3 Implication for testing

Multi-slice loops where only `rf.freqOffset` changes per slice produce
**identical block definition patterns** across slices, making them ideal
for multipass/ONCE-flag testing.

---

## 5. Labels, Flags, and the ONCE Flag

### 5.1 All labels and flags are sticky

**Every** label and flag in the extension chain is latching:

```c
norot_flag = (ext.flag.norot >= 0) ? ext.flag.norot : norot_flag;
nopos_flag = (ext.flag.nopos >= 0) ? ext.flag.nopos : nopos_flag;
pmc_flag   = (ext.flag.pmc   >= 0) ? ext.flag.pmc   : pmc_flag;
nav_flag   = (ext.flag.nav   >= 0) ? ext.flag.nav   : nav_flag;
once_flag  = (ext.flag.once  >= 0) ? ext.flag.once  : once_flag;
```

A value of `-1` means "not set in this block's extension" — the
previous value carries forward. This applies to ALL flags, not just ONCE.

LABELSET/LABELINC (SLC, LIN, REP, AVG, SEG, SET, ECO, PAR, PHS, ACQ)
are similarly accumulated through the block walk.

### 5.2 ONCE flag values

| Value | Name        | Playback rule                              |
|-------|-------------|--------------------------------------------|
| 1     | Preparation | Played only on the **first** average       |
| 0     | Main        | Played on **every** average                |
| 2     | Cooldown    | Played only on the **last** average        |

When `num_averages == 1`, all blocks (prep + main + cooldown) are played.

### 5.3 Prep/cooldown counting (raw walk)

The C library counts prep blocks by walking **forward from block 0**
using the **raw** `ext.flag.once` value (not the sticky-resolved value):

- Blocks with explicit `SET ONCE 1` → `ext.flag.once = 1` → counted
- Blocks with no extension at all → `ext.flag.once = -1` → counted
  (since `-1 ≠ 0`)
- Block with `SET ONCE 0` → `ext.flag.once = 0` → **stops the walk**

The `SET ONCE 0` (transition) block is the **first main block**, not
part of prep.

Cooldown counting walks **backward from the last block**, incrementing
until `ext.flag.once == 2` is encountered (that block is included).

### 5.4 Multipass (inner-loop) folding

When ONCE flags appear mid-sequence (`once_counter` exceeds the
expected 0–2 for a simple prep+cooldown), the library attempts folding:

1. Count trailing `once_flag == 2` blocks → `trailing`.
2. `effective = num_blocks - trailing`.
3. Find shortest period `pl` where `effective % pl == 0` and the
   `(block_id, once_flag)` pattern tiles identically.
4. If not found with trailing separated, try the entire sequence.
5. Fold: keep first period + trailing cooldown.
6. `num_passes = effective / period` (must be ≥ 2).
7. Re-count prep/cooldown from the folded table.

**Valid multipass**: all passes have identical `(block_id, once_flag)`.
**Invalid**: passes differ → `PULSEQLIB_ERR_INVALID_ONCE_FLAGS`.

---

## 6. TR Identification

TR identification runs on the **imaging region** (blocks between prep
and cooldown) using block definition IDs:

1. Build `seq_pat[n]` from block def IDs (pure delays get special
   negative values).
2. `first_repeating_segment()` finds the shortest repeating period `l`.
3. Verify the entire imaging region tiles with period `l`.
4. If no period found and total active duration ≤ 15 s → single-TR
   fallback (entire sequence is one TR).
5. **Prep check**: if `num_prep_blocks % l == 0` and all prep "TRs"
   match the imaging pattern → `degenerate_prep = 1` (prep is just
   extra copies of main TR, not structurally different).
   Otherwise → `degenerate_prep = 0` (true non-degenerate prep region,
   with total duration limited to `PREP_COOLDOWN_THRESHOLD_US = 100000`).
6. Same logic for cooldown.

**Key**: pattern detection **always** happens. It is NOT triggered by
or dependent on ONCE flags. ONCE only determines which blocks are
prep/cooldown; TR detection operates purely on block definition ID
patterns.

---

## 7. Segmentation

The segmentation state machine splits a TR into segments. It runs on
each section (prep+main, main, main+cooldown) independently.

### 7.1 Segment boundary rules

A boundary candidate exists between consecutive blocks if all 3 gradient
axes have physical first/last values within the max-slew-per-raster
threshold (i.e., gradients are at or near zero at the boundary).

States:
- `SEEKING_FIRST_ADC` — before any ADC is seen. If an RF appears, save
  the pre-RF candidate. When an ADC appears, split at the pre-RF
  candidate (separating "excitation" from "readout").
- `SEEKING_BOUNDARY` — after first ADC. Look for an RF after
  a valid candidate → split there. If RF appears without a gradient-
  safe candidate → enter OPTIMIZED_MODE (single segment for the TR).
- `OPTIMIZED_MODE` — no further splitting.

### 7.2 Post-processing

1. **Strip pure delays** — single-block delay-only segments are split
   off from segment cores.
2. **NAV-aware split/merge** (when PMC enabled) — segments with mixed
   NAV/non-NAV blocks are split; adjacent NAV segments are merged.
3. **Deduplication** — segments with identical `unique_block_indices`
   arrays are merged into one unique segment definition.
4. **Per-block flags** — `has_digitalout`, `has_rotation`, `norot_flag`,
   `nopos_flag`, trigger classification (INPUT vs OUTPUT).

### 7.3 Scan-table segmentation fallback

If block-table segmentation fails due to non-zero gradient boundaries
(at TR edges), the library falls back to scan-table-based segmentation,
which resolves blocks through scan table indirection.

---

## 8. Safety Checks

`pulseqlib_check_safety()` runs four checks in order:

1. **Max gradient amplitude** — GSOS (geometric sum of squares) of
   gradient amplitudes across all blocks vs `max_grad_hz_per_m`.
2. **Gradient continuity** — cursor-based dry-run over the full scan
   table. Checks that the step from last gradient sample of block N to
   first sample of block N+1 (in physical coordinates after rotation)
   does not exceed `max_slew * grad_raster_s`. Checks subsequence
   boundaries and trailing edge → zero.
3. **Max slew rate** — per unique gradient definition, `slew_rate[shot]
   * max_amplitude[shot]` vs `max_slew / sqrt(3)`.
4. **Per-subsequence acoustic + PNS** — for each unique shot-index TR
   variant (including prep/cooldown TRs if non-degenerate):
   - Acoustic: gradient spectrogram with FFT, check against forbidden
     frequency bands.
   - PNS: per-axis slew rate convolved with nerve stimulation kernel,
     check combined magnitude vs threshold percentage.

`pulseqlib_check_consistency()` is a separate lighter check run at load
time (file-format validation, raster compatibility).

---

## 9. Frequency Modulation

For off-isocenter imaging, gradients create time-varying frequency
offsets. The freq mod subsystem:

1. **Flags blocks** that have (RF or ADC) + at least one gradient axis.
2. **Builds freq mod library** — per-block normalized gradient waveforms
   within the active event window (RF start→end or ADC start→end).
   Three-channel (Gx, Gy, Gz) peak-normalized shapes.
3. **Deduplication** — entries with identical 3-channel shapes are
   shared.
4. **Plan instances** — per-subsequence plan waveforms incorporating
   rotation matrices and the `NOROT` flag. When NOROT is set, gradient
   waveforms are NOT rotated (used for e.g. multi-shot spiral where the
   readout trajectory should not be frequency-modulated even though it
   is rotated for imaging).
5. **k-space zero crossings** — identified from cumulative gradient
   integration. Used to set ADC `kzero_index` anchors per segment.
   Refocusing pulses (auto-detected from ~180° flip angle) negate k
   at their isocenter.
6. **PMC support** — `pulseqlib_update_freq_mod_collection()` can
   recompute a single subsequence when the PMC navigator updates the
   rotation matrices.
7. **Caching** — freq mod collections can be written/read as binary
   cache files.

---

## 10. Caching

The library supports binary caching of parsed results:

- `pulseqlib_save_cache()` / `pulseqlib_load_cache()` write/read a
  `.bin` file alongside the `.seq` file.
- On `pulseqlib_read()`: if `cache_binary` is true, the library first
  tries to load the cache. If successful, it skips the full parse and
  only recomputes derived data (segment timing, TR-start flags).
- MD5 signature verification (`verify_signature`) can check file
  integrity.
- Freq mod collections have their own separate cache mechanism.

---

## 11. Wrappers

### 11.1 C++ interface (`extensions/pulseqlib/`)

Header-only C++11 RAII layer: `pulseqlib::Collection` (movable,
non-copyable) wraps all C getters as methods. Value types: `Opts`,
`ScanTimeInfo`, `RfStats`, `TrWaveforms`, `AcousticSpectra`,
`PnsResult`, etc.

### 11.2 Python (`python/pulserver/`)

- **pybind11 extension**: `_pulseqlib_wrapper.cpp` binds to
  `pulseqlib::Collection`. Creates `_PulseqCollection` from in-memory
  byte buffers.
- **`SequenceCollection`** class (extends `pypulseq.Sequence`) with
  methods: `report()`, `check()`, `plot()`, `pns()`, `grad_spectrum()`,
  `validate()`, `num_blocks()`, `get_block()`, `num_segments()`,
  `segment_size()`, `get_segment()`, `tr_size()`.
- Helper modules: `_waveforms.py` (unit conversion Hz/m→mT/m),
  `_acoustics.py`, `_pns.py`, `_plot.py`, `_validate.py`,
  `_cache.py`, `_iostream.py`.
- Build: CMakeLists.txt creates pybind11 target `_pulseqlib_wrapper`
  from all `csrc/*.c` + the wrapper `.cpp`.

### 11.3 MATLAB (`matlab/+pulserver/`)

- **MEX gateway**: `pulseqlib_mex.cpp` (R2018a+ C++ MEX API). Single
  entry point dispatching on string commands (`"load"`, `"free"`,
  `"find_tr"`, `"find_segments"`, `"get_tr_waveforms"`, `"check"`,
  `"report"`, `"get_block"`, `"pns"`, `"grad_spectrum"`, etc.).
  Manages a persistent global collection store with 1-based handles.
- **`pulserver.SequenceCollection`** class (handle-based) mirrors the
  Python API. Accepts `mr.Sequence` objects or `.seq` file paths.
- Build: `scripts/setup_mex.m` compiles all `csrc/*.c` + MEX gateway.

### 11.4 Key differences

| Aspect | Python | MATLAB |
|---|---|---|
| Binding layer | pybind11 → C++ Collection | Raw C MEX, command dispatch |
| Input serialization | In-memory bytes via `write_to_stream()` | Temp file → `fread` → delete |
| Memory management | Python GC + RAII unique_ptr | Global store, explicit `"free"` + destructor |
| Indexing | 0-based | 1-based (converted at MEX boundary) |
| Base class | Extends `pypulseq.Sequence` | Standalone `handle` class wrapping `mr.Sequence` |

---

## 12. Project File Structure

| Path | Contents |
|---|---|
| `csrc/` | C89 library source (parse, dedup, structure, safety, freqmod, cache, core, waveforms, getters) |
| `extensions/pulseqlib/` | C++ header-only RAII wrapper |
| `python/pulserver/` | Python package: `SequenceCollection`, pybind11 `.cpp`, helpers |
| `matlab/+pulserver/` | MATLAB package: `SequenceCollection.m`, MEX gateway, helpers |
| `tests/generators/` | MATLAB scripts that generate `.seq` test files |
| `tests/data/` | Generated `.seq` files + ground-truth CSV/TXT files |
| `tests/pytests/` | Python test suite |
| `tests/ctests/` | C test suite (built with CMake) |
| `examples/` | C and Python example programs |
| `docs/` | Sphinx documentation source |
| `scripts/` | Build scripts (MEX setup, CTest runners) |

---

## 13. Test Generator Design Rules

1. All events created **outside** the acquisition loop; only `scaleGrad`
   / scalar property changes (`rf.phaseOffset`, `rf.freqOffset`, etc.)
   inside loops.
2. Preparation scans marked `ONCE=1`; cooldown blocks `ONCE=2`.
3. Raster times chosen for GE + Siemens compatibility.
4. Ground truth exported per-sequence as CSV + metadata text files.
5. `check_and_write()` handles timing check, definitions, file write,
   and ground-truth export. It auto-prepends `dataDir`.
6. Generators must write output to `tests/data/` using
   `dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data')`.
7. Ground-truth files: `_blocks.csv`, `_meta.txt`, `_segments.txt`,
   `_scan_table.csv`, `_tr*_*.csv`.

---

## 14. Common Pitfalls (for AI agents)

- **Do NOT assume** `freqOffset`, `phaseOffset`, or `scaleGrad` changes
  create new block definitions. They don't —only RF/gradient timing and
  block duration matter for the definition ID.
- **Do NOT assume** ADC or extension/label differences create new block
  definitions. They don't.
- **Do NOT assume** ONCE flags trigger pattern detection. Pattern
  detection ALWAYS runs on block definition IDs in the imaging region.
  ONCE only determines prep/cooldown boundaries.
- **ALL labels and flags are sticky** — not just ONCE. Any unset flag
  (`ext.flag.X == -1`) inherits its value from the previous block.
- Multi-slice loops are ideal for multipass testing precisely because
  scalar changes are invisible to block deduplication.
- When adding new generator functions, remember `dataDir` path handling.
- The `export_ground_truth` function derives its output path from the
  `fname` passed to `check_and_write`, so the directory must be embedded.
- Gradient continuity is checked in **physical** coordinates (after
  rotation), not logical. A sequence that is safe in logical coordinates
  may fail after rotation.
- The segmentation state machine can fall back to scan-table-based
  segmentation if block-table boundaries have non-zero gradients.
