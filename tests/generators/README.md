# Adding New Test Sequences

This guide shows how to add a new sequence type with ground-truth data for the
C test suite.

## Overview

Each test sequence needs two things:

1. **MATLAB generator** — builds the Pulseq sequence and uses `TruthBuilder`
   to export ground-truth binary files into `tests/data/`.
2. **C test cases** — entries in `tests/ctests/test_sequences.c` that load the
   `.seq` file and compare the library output against the exported truth.

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
| `setBaseRotation(R)` | 3×3 rotation matrix (default `eye(3)`). |
| `export(out_dir, base_name)` | Run all computation phases and write the 6 output files. |

Notes:

- `setSegments` defines unique segment shapes; `setSegmentOrder` controls reuse/interleaving.
- Repeated or interleaved delays should be represented by a single delay definition reused in `setSegmentOrder`.

### Exported files

`export()` writes the following into `out_dir`, all prefixed by `base_name`:

| Suffix | Format | Contents |
|--------|--------|----------|
| `.seq` | Pulseq text | The sequence file itself. |
| `_meta.txt` | Key-value text | ADC defs, max-B1 index, TR duration. |
| `_tr_waveform.bin` | Binary float32 | Canonical TR waveform knot arrays (time, gx, gy, gz). |
| `_segment_def.bin` | Binary float32 | Per-segment block-level gradient data (time, gx, gy, gz per block). |
| `_freqmod_def.bin` | Binary float32 | Frequency-modulation block definitions (RF and ADC). |
| `_scan_table.bin` | Binary int32 | Scan table: block indices, once flags, norot flags, rotation matrices. |

## Step 2: Add C test cases

Open `tests/ctests/test_sequences.c` and follow the existing pattern.

### 2a. Define a case struct (if adding a new sequence *type*)

For a brand-new sequence family, add a new `typedef struct` and static array:

```c
typedef struct {
    const char* name;
    const char* seq_file;
    const char* base;
    int num_averages;
} my_seq_case;

static const my_seq_case kMySeqCases[] = {
    {"my_seq_v1", "my_seq_v1.seq", "my_seq_v1", 1},
    {"my_seq_v2", "my_seq_v2.seq", "my_seq_v2", 3},
};
```

If adding variants of an existing type (e.g. more GRE cases), just append to
the existing `kGreCases[]` array.

### 2b. Add MU_TEST wrappers and suite registration

For each case index and each phase (check, uieval, geninstructions, freqmod,
scantable), add a one-liner wrapper and register it:

```c
/* Wrappers */
MU_TEST(test_check_my_seq_v1) { run_check_case(&kMySeqCases[0]); }
MU_TEST(test_check_my_seq_v2) { run_check_case(&kMySeqCases[1]); }
/* ... repeat for run_sequences_uieval_case, etc. */

/* Suite */
MU_TEST_SUITE(suite_my_seq_check)
{
    MU_RUN_TEST(test_check_my_seq_v1);
    MU_RUN_TEST(test_check_my_seq_v2);
}
```

Then add the suite to the `main()` runner:

```c
MU_RUN_SUITE(suite_my_seq_check);
```

### 2c. The five test phases

The existing runner functions work for any case struct with the same fields:

| Runner | What it tests |
|--------|---------------|
| `run_check_case` | ADC definitions, max-B1 subsequence, nominal TR. |
| `run_sequences_uieval_case` | Segment definitions + canonical TR waveform. |
| `run_sequences_geninstructions_case` | Per-segment gradient instruction data. |
| `run_freq_mod_definitions_case` | Frequency-modulation block extraction. |
| `run_scan_table_case` | Full scan table (block indices, once/norot flags, rotations). |

## Step 3: Build and run

```bash
# Regenerate ground-truth (MATLAB, from tests/generators/)
matlab -batch "generate_test_sequences"

# Build and run C tests
cd tests/ctests
cmake -S . -B build && cmake --build build
./build/bin/run_tests
```

## Checklist

- [ ] MATLAB generator builds valid sequence (`seq.checkTiming` passes)
- [ ] ONCE labels placed on first dummy block and first imaging block
- [ ] `TruthBuilder` hints match sequence structure (blocks/TR, segment defs, segment order, averages)
- [ ] `.seq` + 5 truth files appear in `tests/data/`
- [ ] Case struct entry added to `test_sequences.c`
- [ ] 5 MU_TEST wrappers + suite registrations added
- [ ] All tests pass (`run_tests` exits 0)
