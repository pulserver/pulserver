# Adding New Test Sequences

This guide shows how to add a new sequence type with ground-truth data for the
C test suite.

## Overview

Each test sequence needs two things:

1. **MATLAB generator** — builds the Pulseq sequence and uses `TruthBuilder`
   to export ground-truth binary files into `tests/generators/expected/`.
2. **C test cases** — entries in `tests/ctests/test_sequences.c` that load the
   `.seq` file and compare the library output against the exported truth.

### generators/ layout

```
tests/generators/
  +testutils/          MATLAB package: TruthBuilder + truth_* inspection utilities
  scripts/             generate_*.m scripts (run from here or from MATLAB path)
  expected/            ground-truth artifacts: .seq + binary truth files
  README.md
```

## Step 1: Write the MATLAB generator

Create a new `.m` file (or add a function to an existing one) that:

1. Builds an `mr.Sequence` object.
2. Hands it to `TruthBuilder` with a few hints.
3. Calls `export()`.

### Minimal template

```matlab
function seq = write_my_sequence(write, ...)
    base = 'my_seq_variant';
    sys  = mr.opts( ...
        'MaxGrad',   28,   'GradUnit', 'mT/m', ...
        'MaxSlew',   150,  'SlewUnit', 'T/m/s', ...
        'rfRasterTime',         2e-6, ...
        'gradRasterTime',      20e-6, ...
        'adcRasterTime',        2e-6, ...
        'blockDurationRaster', 20e-6);
    seq = mr.Sequence(sys);

    % --- build your sequence blocks here ---
    % Use mr.makeLabel('SET','ONCE',1) on the first block of a prep/dummy
    % region, and mr.makeLabel('SET','ONCE',0) on the first block of the
    % main imaging region.  The builder uses these labels to mark scan-table
    % rows as ONCE (non-repeating) segments.

    if ~write, return; end

    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');

    tb = TruthBuilder(seq, sys);
    tb.setBlocksPerTR(4);                % blocks per TR
    tb.setSegments([4]);                 % unique segment definitions
    tb.setSegmentOrder([1]);             % segment instances per TR
    tb.setNumAverages(1);                % number of averages
    % Optional: one anchor fraction per unique ADC definition.
    % [] means all ADC anchors default to 0.5 (middle of readout).
    % tb.anchorPoints.adc = [0.5, 0.0];  % cartesian ADC, spiral ADC
    % tb.setBaseRotation(R);             % optional rotation matrix
    tb.export(out_dir, base);
end
```

### TruthBuilder API

| Method | Purpose |
|--------|---------|
| `TruthBuilder(seq, sys)` | Constructor — accepts the built sequence and system opts. |
| `setBlocksPerTR(n)` | Number of sequence blocks that make up one TR. |
| `setSegments(sizes)` | Unique segment definitions by block count. E.g. `setSegments([2, 1, 4])` for prep, delay, SPGR-shot. |
| `setSegmentOrder(order)` | Segment instances within one TR (1-based segment IDs). E.g. `setSegmentOrder([1, 2, repmat(3,1,Ny), 2])` for prep -> delay -> Ny shots -> delay. |
| `setNumAverages(n)` | Number of averages (default 1). |
| `anchorPoints.adc = [...]` | Optional per-unique-ADC-definition anchor fractions in `[0, 1]`. Empty means all ADC anchors default to `0.5`. |
| `setBaseRotation(R)` | 3×3 rotation matrix (default `eye(3)`). |
| `export(out_dir, base_name)` | Run all computation phases and write the truth output files. |

Notes:

- `setSegments` defines unique segment shapes; `setSegmentOrder` controls reuse/interleaving.
- Repeated or interleaved delays should be represented by a single delay definition reused in `setSegmentOrder`.
- `anchorPoints.adc` is indexed by ADC definition, not ADC block instance. For example, if your case has two unique ADC definitions and the second one is a spiral navigator, use `tb.anchorPoints.adc = [0.5, 0.0]`.
- RF anchors do not need manual configuration here; TruthBuilder continues to use the RF center from Pulseq.

### Exported files

`export()` writes the following into `out_dir`, all prefixed by `base_name`:

| Suffix | Format | Contents |
|--------|--------|----------|
| `.seq` | Pulseq text | The sequence file itself. |
| `_meta.txt` | Key-value text | ADC defs, ADC anchor fractions, max-B1 index, TR duration, segment metadata. |
| `_tr_waveform.bin` | Binary float32 | Canonical TR waveform knot arrays (time, gx, gy, gz). |
| `_segment_def.bin` | Binary float32 | Per-segment block-level gradient data (time, gx, gy, gz per block). |
| `_freqmod_def.bin` | Binary float32 | Frequency-modulation block definitions (RF and ADC). |
| `_freqmod_plan.bin` | Binary float32 | Supplemental plan-level projected freq-mod truth (x/y/z/oblique probes). |
| `_label_state.bin` | Binary int32 | Supplemental sticky-label state snapshots for all scan and ADC rows. |
| `_scan_table.bin` | Binary int32 | Scan table: block indices, once flags, norot flags, rotation matrices. |

## Step 2: Add C test cases

Open `tests/ctests/test_sequences.c` and follow the existing pattern.

### 2a. Define a case entry (in the unified `seq_case` struct)

All test sequences use the same `seq_case` struct:

```c
typedef struct {
    const char* name;
    const char* seq_file;
    const char* base;
    int num_averages;
} seq_case;
```

Add your sequence to an appropriate static array:

```c
static const seq_case kMySeqCases[] = {
    {"my_seq_v1", "my_seq_v1.seq", "my_seq_v1", 1},
    {"my_seq_v2", "my_seq_v2.seq", "my_seq_v2", 3},
};
```

For a new sequence family, create a new array (e.g. `kMySeqCases`).
For variants of an existing type (e.g. more GRE cases), append to the existing array.
For variants of an existing type with a different distinguishing characteristic
(e.g. noncartesian MPRAGE vs cartesian MPRAGE), create a separate array
(e.g. `kMprageNoncartCases`), which allows for future expansion with other variants.

### 2b. Add MU_TEST wrappers and suite registration

For each case index in your array(s) and each phase (check, uieval, geninstructions, freqmod,
scantable), add a one-liner wrapper and register it. All test functions accept a `seq_case*`:

```c
/* Wrappers */
MU_TEST(test_check_my_seq_v1) { run_check_case(&kMySeqCases[0]); }
MU_TEST(test_check_my_seq_v2) { run_check_case(&kMySeqCases[1]); }
/* ... repeat for run_sequences_uieval_case, run_sequences_geninstructions_case, 
       run_freq_mod_definitions_case, run_scan_table_case */

/* Suite */
MU_TEST_SUITE(suite_my_seq_check)
{
    MU_RUN_TEST(test_check_my_seq_v1);
    MU_RUN_TEST(test_check_my_seq_v2);
}
```

Then add each phase suite to the `test_sequences_main()` runner.

### 2c. The five test phases

All runner functions work with the unified `seq_case` struct:

| Runner | What it tests |
|--------|---------------|
| `run_check_case` | ADC definitions, max-B1 subsequence, nominal TR. |
| `run_sequences_uieval_case` | Segment definitions + canonical TR waveform. |
| `run_sequences_geninstructions_case` | Per-segment gradient instruction data. |
| `run_freq_mod_definitions_case` | Frequency-modulation block extraction. |
| `run_scan_table_case` | Full scan table (block indices, once/norot flags, rotations). |

## Step 3: Build and run

```bash
# Regenerate ground-truth (MATLAB — run from tests/generators/)
cd tests/generators
matlab -batch "run('scripts/generate_test_sequences.m')"

# Build and run C tests (from repo root)
cd tests/ctests
cmake -S . -B build && cmake --build build
./build/bin/run_tests
```

## Checklist

- [ ] MATLAB generator builds valid sequence (`seq.checkTiming` passes)
- [ ] ONCE labels placed on first dummy block and first imaging block
- [ ] `TruthBuilder` hints match sequence structure (blocks/TR, segment defs, segment order, averages, ADC anchors when needed)
- [ ] `.seq` + truth artifact files appear in `tests/generators/expected/`
- [ ] Case struct entry added to `test_sequences.c`
- [ ] 5 MU_TEST wrappers + suite registrations added
- [ ] All tests pass (`run_tests` exits 0)

## MATLAB Truth Validation Utilities

The `+testutils` package in `tests/generators/` provides functions to parse,
report, and plot the ground-truth artifacts exported by `TruthBuilder`.  All
functions accept either a base-name string (e.g. `'gre_2d_1sl_1avg'`) or a
pre-parsed truth struct, so you can parse once and reuse.

To use these utilities interactively, add `tests/generators/` to the MATLAB
path (`addpath`) or `cd` into it before calling them:

### Parse all artifacts for one case

```matlab
truth = testutils.truth_parse_case('gre_2d_1sl_1avg');
```

Parses the 5 binary/text truth artifacts (`_meta.txt`, `_tr_waveform.bin`,
`_segment_def.bin`, `_freqmod_def.bin`, `_scan_table.bin`) into a single
struct.  Also runs cross-file consistency checks (stored in
`truth.validation`).

### Print a terminal report

```matlab
report = testutils.truth_report_case('gre_2d_1sl_1avg');
```

Prints a summary to the command window:

- Metadata: ADC definitions, ADC anchor fractions, TR duration, segment count.
- Canonical TR waveform stats (samples, duration, peak gradients).
- Frequency-modulation definitions (type, samples, raster, ref integral).
- Scan-table coverage (RF / ADC / freq-mod / trigger / digitalout row counts).
- Cross-file consistency warnings.

### Plot canonical TR waveforms

```matlab
testutils.truth_plot_tr_waveforms('gre_2d_1sl_1avg');
testutils.truth_plot_tr_waveforms(truth, 'overlay', false);
testutils.truth_plot_tr_waveforms(truth, 'show_slew', true);
```

Three subplots (Gx / Gy / Gz) showing all canonical TR waveforms overlaid.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `'overlay'` | `true` | Overlay all TRs on the same axes with a legend. |
| `'show_slew'` | `false` | Plot slew rate (gradient derivative) instead of amplitude. |

### Plot segment block waveforms

```matlab
testutils.truth_plot_segments('gre_2d_1sl_1avg');
testutils.truth_plot_segments(truth, 'segment_idx', [1 2]);
```

Five subplots (Gx / Gy / Gz / RF |B1| / ADC) on a shared time axis in ms.
Each block is colour-coded with a global legend at the bottom.

- Gradient time axes use the per-channel `grad_time_s` knot arrays (correct
  rise/flat/fall for trapezoids, uniform raster for arbitrary waveforms).
- RF time axis uses the per-block `rf_raster_us` field.
- Units: gradients in mT/m, RF in µT, ADC as a binary mask.
- Block start offsets are computed from actual event endpoints with a small
  visual gap between blocks.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `'segment_idx'` | all | 1-based segment indices to plot (one figure per segment). |

### Plot frequency-modulation definitions

```matlab
testutils.truth_plot_freqmod_defs('gre_2d_1sl_1avg');
testutils.truth_plot_freqmod_defs(truth, 'def_idx', [1 2]);
testutils.truth_plot_freqmod_defs(truth, 'segment_idx', 2);
```

3×2 grid (rows = Gx / Gy / Gz; col 1 = freq mod in Hz/m, col 2 = cumulative
phase in rad/m).  Each definition is shown as a zero-padded boxcar on the full
segment timeline: zero before the active window (RF or ADC delay), the stored
waveform during the window, and zero after.  A circle marker on the phase
column indicates the stored `ref_integral` value at `ref_time_us` as a sanity
check of the running integral.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `'def_idx'` | all | Which frequency-modulation definitions to plot (1-based). |
| `'segment_idx'` | `1` | Which segment provides the timing context. |

### One-shot workflow

```matlab
testutils.truth_plot_case('gre_2d_1sl_1avg');
testutils.truth_plot_case('gre_2d_1sl_1avg', 'show_report', true);
testutils.truth_plot_case('gre_2d_1sl_1avg', 'plot_freqmod', false);
testutils.truth_plot_case('fse_2d_1sl_1avg', 'scan_shift', [0.01 0 0]);
```

Convenience entrypoint that parses once, then runs report + all plotters and
prints the terminal inspection tables.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `'show_report'` | `true` | Print the terminal report. |
| `'plot_tr'` | `true` | Plot canonical TR waveforms. |
| `'plot_segments'` | `true` | Plot segment block waveforms. |
| `'plot_freqmod'` | `true` | Plot frequency-modulation definitions (skipped if none exist). |
| `'print_scan_table'` | `true` | Print the full scan-table terminal listing. |
| `'print_label_table'` | `true` | Print the ADC label-sequence terminal listing. |
| `'plot_kspace_traj'` | `true` | Plot non-Cartesian k-space trajectories (no-op if all are straight lines). |
| `'scan_shift'` | `[0 0 0]` | Isocenter shift in metres forwarded to `truth_print_scan_table`. |

### Print the scan table

```matlab
testutils.truth_print_scan_table(truth);
testutils.truth_print_scan_table(truth, 'adc_only', true);
testutils.truth_print_scan_table(truth, 'shift', [0.01 0 0]);
```

Prints a formatted scan-table listing to the terminal, one line per scan row.
Each row shows:

- RF frequency (Hz), phase (rad), and amplitude (normalised).
- Gradient amplitudes Gx / Gy / Gz (Hz/m).
- Shot index (value of the `SEG` sticky label at that row).
- ADC frequency (Hz) and ADC phase (rad).
- Rotation-matrix ID (1-based first-appearance ordering).
- Freq-mod plan ID (0 = no freq-mod).
- When freq-mod is active: per-probe projected phase totals (x / y / z /
  oblique, in rad).
- When freq-mod is active and `shift` is provided: per-axis shift (mm),
  reference gradient scale factors (rad/m), and resulting projected phase
  contribution (rad).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `'shift'` | `[0 0 0]` | Isocenter shift in metres used for phase-product columns. |
| `'adc_only'` | `false` | When `true`, only rows with an active ADC are listed. |

### Print the ADC label sequence

```matlab
testutils.truth_print_label_table(truth);
```

Prints a compact listing of every ADC readout row to the terminal:

```
ADC#  |  ave  |  rot  |  traj  |  n_samp  |  center
```

- `ave` — value of the `AVE` (or `REP`) sticky label at this ADC row.
- `rot` — rotation-matrix ID (1-based, first-appearance ordering) from the
  corresponding scan-table entry.
- `traj` — value of the `SEG` (or `SHT`) sticky label (shot/trajectory index).
- `n_samp` — number of ADC samples for this readout (from `_meta.txt`).
- `center` — 1-based sample index of the k-space centre (`adc_kzero_us /
  dwell_us`, rounded to nearest integer, plus 1).

Prints a "no ADC rows" notice if the sequence has no ADC events.

### Plot k-space trajectories

```matlab
testutils.truth_plot_kspace_traj(truth);
```

Reconstructs the k-space trajectory for each unique (ADC-definition,
gradient-amplitude) block in the segment definitions by integrating the
piecewise-linear gradient waveforms onto the ADC dwell grid.  K-space values
are in cycles/m (Hz·s/m convention) with the k-space centre at sample index 0.

Straight-line trajectories — where every sample lies within a relative
tolerance of the chord between the first and last k-space points — are
automatically detected and excluded.  This covers standard Cartesian readout
lines, radial spokes, and any other constant-gradient readout.

If every unique trajectory is a straight line, no figure is created and an
informational message is printed.  Otherwise a figure with three stacked
subplots (kx / ky / kz vs sample index) is produced, one curve per
non-trivial trajectory.

No parameters.  Requires MATLAB R2017b+ (`vecnorm`) and R2018b+ (`sgtitle`).

### Label-state export note (future MRD converters)

`TruthBuilder` also exports `_label_state.bin` to capture dynamic sticky labels
for downstream metadata validation. This artifact is designed for future MRD
converter checks where label state must be validated at ADC rows.

- Label vocabulary is discovered dynamically from sequence block labels.
- Per-block updates use sticky semantics with `SET` applied before `INC`.
- Two tables are exported:
    - per-scan-row label states (aligned to `_scan_table.bin` row order), and
    - per-ADC-row label states (subset of scan rows with ADC active).
- Per-label ADC observed min/max values are exported for encoding-limit checks
    against int32 metadata storage.
