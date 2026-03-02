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

The target vendor is selected at **runtime** via the `opts->vendor`
field (a `PULSEQLIB_VENDOR_*` constant from `pulseqlib_config.h`).
`pulseqlib_opts_init()` sets `vendor` to the compile-time default
(`PULSEQLIB_VENDOR`, which defaults to `PULSEQLIB_VENDOR_GEHC` unless
overridden with `-DPULSEQLIB_VENDOR=N`).  Callers may override
`opts->vendor` after calling `opts_init`.  The vendor propagates from
`seq->opts.vendor` → `desc->vendor` (copied in `get_unique_blocks`).

All vendor-specific behaviour uses **runtime** `if (vendor == …)`
checks — there are **no** `#if PULSEQLIB_VENDOR` guards in the C
source.  Vendor-gated features include: RF statistics/bandwidth
(GEHC), FFT convolution for PNS (GEHC), label table columns (GEHC),
RF shape amplitude scaling (GEHC), and freq-mod active region
calculation (GEHC uses RF-stats duration/isodelay, others use block
duration).

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
5. **Scan table construction** — The scan table is the fully-expanded
   play order.  The outer loop is over **passes** (`num_passes`),
   the inner loop is over **averages** (`num_averages`).  Within each
   average, the block table is walked as
   `prep + main + cooldown` with ONCE semantics (prep only on the
   first average, cooldown only on the last, main on every average).
   Total scan table length =
   `num_passes × (prep×1 + main×num_averages + cooldown×1)`.
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
expected 0–2 for a simple prep+cooldown), the library attempts folding
by **whole-pass comparison** (no period-finding — that belongs to
`get_tr_in_sequence`):

1. Record `first_once = block_table[0].once_flag`.
2. Walk the block table; every transition **to** `first_once` from a
   different once_flag marks a new **pass boundary**.
3. Derive `pass_len` from the first pass (blocks between boundary 0
   and boundary 1).
4. If the last pass is longer than `pass_len` and the extra tail
   blocks are all `once==2`, treat the tail as **trailing cooldown**.
   If the last pass is shorter and all `once==2`, treat the entire
   last segment as trailing cooldown and decrement the pass count.
5. Verify every pass has the same length and identical
   `(block_id, once_flag)` at each position.
6. Fold: keep first pass + trailing cooldown.
   `num_passes ≥ 2` required.
7. Re-count prep/cooldown from the folded table.

**Valid multipass**: all passes are identical block-for-block.
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

Block-table segmentation requires that the first and last blocks of
each TR have near-zero gradient first/last values (within
`max_slew × grad_raster_s`).  When this condition is violated —
typically in **bSSFP** sequences where gradients are intentionally
non-zero at TR edges — segmentation returns
`PULSEQLIB_ERR_SEG_NONZERO_START_GRAD` or
`PULSEQLIB_ERR_SEG_NONZERO_END_GRAD`.

The library then falls back to **scan-table-based segmentation**
(`find_segments_on_scan_table`).  This variant:

1. Builds a block-def-ID pattern from the scan table.
2. Finds the repeating period via `first_repeating_segment()`.
3. Runs the same segment state machine but resolves blocks through
   scan-table indirection (positions are scan-table indices, not
   block-table indices).
4. The first/last block gradient-zero checks now apply to the
   **entire scan-table period** edges, not individual TR edges.
   For bSSFP, this typically means the whole pass is one segment
   (because non-zero gradient edges suppress all internal boundary
   candidates).

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

### 8.1 TR waveform amplitude modes

When extracting gradient waveforms for safety checks or k-space
analysis, the library supports three **amplitude modes**
(parameter `amplitude_mode` in `get_gradient_waveforms_range`):

| Mode | Name | Description |
|------|------|-------------|
| 0 | Actual | Uses the per-instance amplitude from the block table entry (one shot index). |
| 1 | Position-max | For each block **position** within the TR, computes the worst-case (maximum \|amplitude\|) across **all TR instances** that share the same shot-index group. Used for **safety checks** — gives the worst-case gradient waveform at every position. |
| 2 | Definition-min | For each gradient definition, uses `gd->min_amplitude[shot]` (the minimum \|amplitude\| observed across all table entries for that definition and shot index). Used for **k-space zero-crossing detection** — gives the best-case (smallest) gradient amplitude, which is more robust for identifying crossings. |

The position-max computation (`compute_position_max_amplitudes_filtered`)
groups TR instances by shot-index fingerprint
(`find_unique_shot_trs`), then for each position in the TR template
takes `max(|amplitude|)` across all instances in the matching group.

> **Note**: `min_amplitude` is currently the minimum **absolute**
> amplitude across all table entries for a grad definition + shot.  A
> more robust alternative would be **min positional amplitude** (the
> minimum at each TR position across instances, analogous to how
> position-max works) rather than global min across all entries.

### 8.2 Max-energy segment instance

After segment deduplication, each unique segment may have many
instances in the expanded segment table.  The library tracks which
instance has the **highest total gradient energy**
(`inst_energy = Σ energy[shot] × amplitude²` across all 3 gradient
axes and all blocks in the instance).  The winning instance's
`start_block` is stored in
`segment_definitions[unique_idx].max_energy_start_block` and is used
as the representative instance for gradient initial-state definition
(connect waveforms, etc.).

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
  segmentation if block-table boundaries have non-zero gradients
  (e.g., bSSFP). See §7.3 for details.
- **Safety TR selection** uses position-max amplitude (mode 1) to get
  worst-case gradients across all TR instances. **k-space crossing
  detection** uses definition-min amplitude (mode 2). See §8.1.
- The max-energy segment instance (§8.2) determines which instance's
  gradients define the initial state for waveform connection.
