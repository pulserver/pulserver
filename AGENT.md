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
3. **Multipass verification** — If ONCE flags appear mid-sequence, the
   block table is verified per-section across passes but NOT folded.
   The full unfolded block table is preserved. `pass_len` is set to
   the number of blocks per pass (see §5).
4. **TR identification** — `first_repeating_segment()` finds the
   shortest repeating period in the **imaging region** of the first
   pass (`[0, pass_len)`).  Pattern detection runs on block definition
   IDs augmented with RF amplitude/shim info. It is NOT triggered by
   ONCE flags (see §6).
5. **Scan table construction** — The scan table is the fully-expanded
   play order.  The outer loop is over **passes** (`num_passes`),
   the inner loop is over **averages** (`num_averages`).  Within each
   average, the per-pass block table slice is walked as
   `prep + main + cooldown` with ONCE semantics (prep only on the
   first average, cooldown only on the last, main on every average).
   Each pass uses its own block-table offset (`base = pass × pass_len`)
   so that `scan_table_block_idx` values point to the correct per-pass
   block-table entries (preserving per-instance RF/ADC data).
   Total scan table length =
   `num_passes × (prep×1 + main×num_averages + cooldown×1)`.
6. **Segmentation** — A state machine splits the first pass of the
   scan table into segments (prep/main/cooldown) based on RF/ADC
   boundaries and gradient continuity (see §7).
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
| **Gradient** | type, rise/flat/fall (trap) or num_samples+time_shape_id (arb), delay | amplitude, shot_index |
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

### 5.4 Multipass verification (no folding)

When ONCE flags appear mid-sequence (`once_counter` exceeds the
expected 0–2 for a simple prep+cooldown), the library detects passes
and verifies per-section structural identity without folding:

1. Record `first_once = block_table[0].once_flag`.
2. Walk the block table; every transition **to** `first_once` from a
   different once_flag marks a new **pass boundary**.
3. Derive `pass_len` from the first pass (blocks between boundary 0
   and boundary 1).
4. Reject uneven passes: verify `num_blocks == num_passes × pass_len`.
   No trailing-cooldown special-casing: the last pass's cooldown runs
   to EOF and is naturally the same length as every other pass's
   cooldown.
5. Count section sizes within first pass: leading `once==1` blocks
   → `num_prep_in_pass`, trailing `once==2` → `num_cool_in_pass`,
   remainder → `num_main_in_pass`.
6. Verify all passes per section: for each pass 1..N-1, compare
   `(block_id, once_flag)` at each position in prep, main, and
   cooldown sections against the first pass.
7. Set `desc->num_passes`, `desc->pass_len`, `desc->num_prep_blocks`,
   `desc->num_cooldown_blocks`. The full block table is preserved
   (no folding) so that per-instance RF/ADC freq/phase data is retained.

**Valid multipass**: all passes are structurally identical per-section.
**Invalid**: any section differs → `PULSEQLIB_ERR_INVALID_ONCE_FLAGS`.

For single-pass sequences, `pass_len = num_blocks`.

---

## 6. TR Identification

TR identification runs on the **imaging region** (blocks between prep
and cooldown) using block definition IDs:

1. Build `seq_pat[n]` from block def IDs (pure delays get special
   negative values).
2. **RF-aware pattern augmentation** — before period detection, `seq_pat`
   is augmented so that blocks sharing the same structural definition
   but differing in RF amplitude or RF shim ID receive distinct pattern
   values.  A copy of the pre-augmentation pattern (`base_pat`) is
   saved for VFA rejection (step 4b).

   Implementation: quantise the RF amplitude to an integer
   (`amplitude × 1e6`, rounded), form 3-column integer rows
   `(base_pat, quantised_amp, shim_id)`, run them through the existing
   hash-based dedup (`pulseqlib__deduplicate_int_rows`), then remap
   `seq_pat` entries to `max_existing_pat + 1 + label`.  This
   makes MRF-style flip-angle schedules that repeat every N TRs
   produce the correct RF-period-aware TR size.
3. `first_repeating_segment()` finds the shortest repeating period `l`
   in the RF-augmented `seq_pat`.
4. Verify the entire imaging region tiles with period `l`.
   - **4a. If period found**: validate that every block in the imaging
     region matches the pattern. Any mismatch → falls into the
     "not found" path.
   - **4b. If no period found — VFA rejection**: compare against the
     saved `base_pat` (structural pattern without RF).  If the base
     pattern **does** have a valid repeating period, the sequence has
     non-periodic RF over a repeating structure (e.g. VFA SPGR).  Such
     sequences must be designed as separate subsequences, so the library
     rejects with `PULSEQLIB_ERR_TR_PATTERN_MISMATCH` instead of
     falling through to single-TR.
   - **4c. If no period found and base also non-periodic**: if total
     active duration ≤ 15 s → single-TR fallback (entire sequence is
     one TR).  Otherwise → `PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN`.
5. **Prep check**: if `num_prep_blocks % l == 0` and all prep "TRs"
   match the imaging pattern → `degenerate_prep = 1` (prep is just
   extra copies of main TR, not structurally different).
   Otherwise → `degenerate_prep = 0` (true non-degenerate prep region,
   with total duration limited to `PREP_COOLDOWN_THRESHOLD_US = 100000`).
6. Same logic for cooldown.

**Key**: pattern detection **always** happens. It is NOT triggered by
or dependent on ONCE flags. ONCE only determines which blocks are
prep/cooldown; TR detection operates purely on block definition ID
patterns (augmented with RF info).

---

## 7. Segmentation

Segmentation operates exclusively on the **scan table** — there is no
block-table segmentation path.  The entry point is
`pulseqlib__get_scan_table_segments()` in `pulseqlib_structure.c`,
called from `pulseqlib_core.c` during sequence loading.

### 7.1 Segment boundary rules

A boundary candidate exists between consecutive scan-table positions if
all 3 gradient axes have physical first/last values within the
max-slew-per-raster threshold (i.e., gradients are at or near zero at
the boundary).  Block-table indices are resolved through the scan table
(`scan_table_block_idx[pos]`).

Gradient first/last values are resolved using the **per-instance
shot index** (`desc->grad_table[gid].shot_index`), not shot 0.
This ensures that deduped gradient definitions with multiple shots
(e.g. phase-encode tables) use the correct amplitude for the specific
scan-table entry being checked.

States (in `find_segments_on_scan_table`):
- `SEEKING_FIRST_ADC` — before any ADC is seen. If an RF appears, save
  the pre-RF candidate. When an ADC appears, split at the pre-RF
  candidate (separating "excitation" from "readout").
- `SEEKING_BOUNDARY` — after first ADC. Look for an RF after
  a valid candidate → split there. If RF appears without a gradient-
  safe candidate → enter OPTIMIZED_MODE (single segment for the TR).
- `OPTIMIZED_MODE` — no further splitting.

### 7.2 Three-section retry on the first pass

Segmentation operates on the **first pass** of the scan table
(`pass_size = scan_table_len / num_passes`).  The first pass is divided
into three sections based on the TR descriptor:

- **Prep**: `[0, num_prep_blocks + k×tr_size)` for k=1,2,…
  (skipped if `degenerate_prep` or `num_prep_blocks == 0`)
- **Main**: `[num_prep_blocks, num_prep_blocks + k×tr_size)` for k=1,2,…
- **Cooldown**: `[pass_size - num_cooldown_blocks - k×tr_size, pass_size)`
  for k=1,2,…
  (skipped if `degenerate_cooldown` or `num_cooldown_blocks == 0`)

Each section retries with increasing multiples of `tr_size`:
1. A fast `scan_boundary_gradients_ok()` pre-check tests whether
   the first/last scan-table positions of the candidate region have
   near-zero gradient values (skips the full state machine if not).
2. `find_segments_on_scan_table()` runs the segment state machine.
3. If it fails with `SEG_NONZERO_START_GRAD` or `SEG_NONZERO_END_GRAD`,
   the section expands to the next multiple.
4. If any section covers the entire first pass, remaining sections
   are skipped.
5. When the main section needed `mult > 1` TRs to succeed, the
   `tr_descriptor` is updated with the expanded TR size and duration.

**Fallback**: if all three sections produce zero segments (e.g. when
all boundary pre-checks skip, or when cooldown has 0 blocks and is
never entered), a single `find_segments_on_scan_table` call over
`[0, pass_size)` without boundary pre-check runs as a last resort.
This either succeeds or propagates the actual gradient error code.

### 7.3 Post-processing

1. **Strip pure delays** (`strip_pure_delays_scan`) — single-block
   delay-only segments are split off from segment cores.  Applied
   per section (prep, main, cooldown independently).
2. **NAV-aware split/merge** (`nav_split_merge`, when PMC enabled) —
   segments with mixed NAV/non-NAV blocks are split; adjacent NAV
   segments are merged.  Applied per section.
3. **Deduplication** — segments with identical `unique_block_indices`
   arrays are merged into one unique segment definition.  Pure-delay
   segments share a single definition.  Dedup is across all sections.
4. **Per-block flags** — `has_digitalout`, `has_rotation`, `norot_flag`,
   `nopos_flag`, trigger classification (INPUT vs OUTPUT).
5. **Segment tables** — three separate tables (`prep_segment_table`,
   `main_segment_table`, `cooldown_segment_table`) map expanded
   segments to unique segment IDs.

### 7.4 Scan-table seg_id tiling

The seg_id pattern from the first pass is tiled across all passes
in `scan_table_seg_id[n] = pattern[n % pass_size]`.

### 7.5 Scan-table consistency validation

`check_scan_table_segments` validates the expanded scan table against
segment definitions.  It tracks `pos_in_seg` (position within the
current segment) and resets it to 0 when the segment ID changes **or**
at a TR boundary (`scan_table_tr_start[n]` is set).  Without the TR
boundary reset, the same segment spanning consecutive TRs would
accumulate `pos_in_seg` past the segment's `num_blocks`, causing a
spurious `PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH`.

### 7.6 Cross-pass RF/shim consistency

`check_cross_pass_rf_consistency` (in `check_consistency`) compares
RF amplitude and shim ID patterns across passes.  Pass 0 is the
reference; passes 1..N-1 are compared position-by-position via
`scan_table_block_idx`, which now points to each pass's own
block-table entries (no folding).  Mismatches produce
`PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC` or
`PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC`.

Note: `freq_offset` and `phase_offset` are NOT compared — they
legitimately differ across passes (e.g. multi-slice selection).

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
analysis, the library supports three **amplitude modes**.  The same
`PULSEQLIB_AMP_*` defines are used consistently in both the public API
(`pulseqlib_get_tr_waveforms`) and the internal helper
(`pulseqlib__get_gradient_waveforms_range`):

| Define | Value | Name | Description |
|--------|-------|------|-------------|
| `PULSEQLIB_AMP_MAX_POS` | 0 | Position-max | For each block **position** within the TR, computes the worst-case (maximum \|amplitude\|) across **all TR instances** that share the same shot-index group. Used for **safety checks**. |
| `PULSEQLIB_AMP_ZERO_VAR` | 1 | Zero-variable | For each block position, zeros out gradient axes that **vary** across TR instances (detected via `variable_grad_flags`), while keeping constant-amplitude gradients at their actual value. Uses the position-max arrays with variable positions zeroed. Used for **k-space analysis**. |
| `PULSEQLIB_AMP_ACTUAL` | 2 | Actual | Uses the per-instance amplitude from the block table entry (one shot index). |

The position-max computation (`compute_position_max_amplitudes_filtered`)
groups TR instances by shot-index fingerprint
(`find_unique_shot_trs`), then for each position in the TR template
takes `max(|amplitude|)` across all instances in the matching group.

> **Note**: `variable_grad_flags` is an `int*` array of size `tr_size * 3`
> stored in `pulseqlib_sequence_descriptor`.  Layout: `flags[pos * 3 + axis]`
> where axis 0=gx, 1=gy, 2=gz.  Value 1 = variable, 0 = constant.
> Computed once by `pulseqlib__compute_variable_grad_flags()` during
> sequence analysis (called from `pulseqlib_core.c`).

### 8.2 K-zero refinement (per-ADC)

For each ADC event in a segment, the kzero sample index is determined
by finding the minimum of `krss` (k-space RSS magnitude) within the
ADC's time window.  This per-ADC approach replaces the previous global
zero-crossing search and is more robust for sequences with multiple
readout types or non-Cartesian trajectories.  The N/2 fallback is
used when no k-space trajectory is available.

### 8.3 Max-energy segment instance

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
- Segmentation operates exclusively on the scan table (no block-table
  segmentation path). The three-section retry (prep/main/cooldown)
  expands regions by TR-size multiples when gradient boundaries are
  non-zero.  A final fallback over the entire first pass runs without
  boundary pre-check to propagate the actual error code. See §7.
- **Safety TR selection** uses position-max amplitude (mode 1) to get
  worst-case gradients across all TR instances. **k-space crossing
  detection** uses definition-min amplitude (mode 2). See §8.1.
- The max-energy segment instance (§8.3) determines which instance's
  gradients define the initial state for waveform connection.
- **TR detection is RF-aware** — blocks identical in structure but
  differing in RF amplitude or shim ID get distinct pattern values.
  MRF-style sequences (variable flip angle, periodic RF schedule) are
  correctly handled. VFA SPGR-style concatenations (non-periodic RF
  over a repeating structure) are rejected at load time with
  `PULSEQLIB_ERR_TR_PATTERN_MISMATCH`. See §6 step 4b.
- **Segmentation uses per-instance shot indices** for gradient
  first/last value lookups, not shot 0. Deduped gradient definitions
  with multiple shots resolve to the correct amplitude for each block
  table entry.
- **Three-section segmentation retry** (§7.2): prep/main/cooldown
  each retry with increasing TR-size multiples on the first pass of
  the scan table. If all sections produce nothing, a single un-gated
  attempt over the full first pass runs as a last resort.
- **Cross-pass RF/shim check** (§7.6): after segmentation,
  `check_cross_pass_rf_consistency` verifies RF amplitude and shim ID
  patterns are identical across passes.  This is a real check since
  the unfolded block table preserves per-pass entries.
- **`pass_len` vs `num_blocks`**: after the unfolding change,
  `num_blocks = num_passes × pass_len` (total across all passes).
  `pass_len` is the per-pass count.  Code that operates on a single
  pass (TR detection, label table, etc.) must use `pass_len`, not
  `num_blocks`.  Getter bounds checks use `num_blocks` since they
  accept indices into the full block table.
- **Scan-table consistency** resets `pos_in_seg` at TR boundaries
  (§7.5), not just on segment-ID changes. Without this, the same
  segment spanning consecutive TRs triggers a false SEG_MISMATCH.
