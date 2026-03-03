## Plan: Phase 2 — Segmentation Tests + Generator Revisions

Validate the C library's segmentation pipeline against MATLAB-generated ground truth. Implement in two waves (existing data vs. post-regen), with three parallel workstreams: (A) C test suite, (B) once-flag generator fix, (C) segmentation generator scan-table fix. Also includes two C-side algorithmic changes: per-ADC kzero refinement and MIN_ABS → MIN_POS mode replacement.

**Steps**

### A. Once-Flag Generator Revision

1. **Revise** `tests/generators/generate_once_flag_test_cases.m`:
   - **Fix tests 01–03, 08–09**: merge standalone `mr.makeLabel('SET','ONCE', 0)` block into the first actual main block (add `lblOnce0` to the `rf + gx_flat1` addBlock call). This matches the pattern used in the segmentation generators (bSSFP, SPGR, etc.).
   - **Fix test 10**: merge both standalone ONCE=0 blocks into their respective next blocks.
   - **Remove tests 05, 06** (prep-only, cooldown-only — ramp blocks break TR periodicity; tests 01–04 cover these cases sufficiently).
   - **Remove test 07** (ambiguous "nonvalid" intent — would succeed after fix).
   - **Remove test 14** (trailing cooldown not supported — `num_blocks != num_passes × pass_len` fails at Phase B).
   - **Renumber** remaining 13 tests sequentially 01–13. New mapping:
     - 01: single TR valid (was 01)
     - 02: dual TR valid (was 02)
     - 03: triple TR valid (was 03)
     - 04: degenerate prep/cooldown (was 04)
     - 05: prep too long (was 08)
     - 06: cooldown too long (was 09)
     - 07: ONCE in middle, invalid (was 10)
     - 08: multipass valid [P,M,M,C]×3 (was 11)
     - 09: multipass valid prep only (was 12)
     - 10: multipass valid cooldown only (was 13)
     - 11: multipass multi-TR (was 15)
     - 12: multipass fail diff main (was 16)
     - 13: multipass fail diff length (was 17)

2. **Regenerate** all 13 `.seq` files on MATLAB machine.

### B. Segmentation Generator Scan-Table Fix

3. **Fix `export_scan_table`** in `tests/generators/generate_segmentation_test_sequences.m` (lines 338–360):
   - Current bug: each pass replays all `num_blocks` entries with `block_idx 0..num_blocks-1`
   - Fix: compute `pass_len = N / num_passes`, then each pass walks `pass_len` blocks with base offset `pass * pass_len`
   - Per-pass scan order: `prep_idx = base..(base+num_prep-1)`, `main_idx = base+num_prep..base+pass_len-num_cool-1`, `cool_idx = base+pass_len-num_cool..base+pass_len-1`
   - Prep on first average, cooldown on last average (existing logic), main on every average
   - Total rows = `num_passes × per_pass_scan_size` where `per_pass_scan_size = num_prep + num_averages × num_main + num_cool`

4. **Regenerate** all 25 segmentation ground truth datasets.

### C. C Test Suite Implementation

5. **Add ground-truth parser helpers** in new `tests/ctests/test_seg_helpers.h`:
   - `parse_meta(path)` → struct with `num_blocks`, `num_averages`, `num_adcs`, `num_prep_blocks`, `num_cool_blocks`, `degenerate_prep`, `degenerate_cool`, `tr_size`, `num_segments`, `num_passes`, `total_duration_us`
   - `parse_segments(path, seg_idx, out_ids, out_count)` — reads Nth line of `_segments.txt`
   - `parse_scan_table(path, out_positions, out_block_indices, out_count)` — reads `_scan_table.csv`
   - `parse_waveform_csv(path, out_time_us, out_amp, out_count)` — reads `_tr_*_{gx,gy,gz}.csv`
   - `parse_anchors(path)` → struct with `rf_isocenter_us`, `rf_refocus_isocenter_us`, `adc_kzero_us`

6. **Add `load_seq_with_averages(basename, opts, num_averages)`** in `tests/ctests/test_helpers.h` — identical to `load_seq` but accepting `num_averages` parameter.

7. **Create** `tests/ctests/test_segmentation.c` with test suites:

   **Suite A — Once-flag tests** (13 tests): For each revised once-flag sequence, load with `default_opts_init` and verify the expected outcome:
   - Tests 01–04, 08–11: expect `PULSEQLIB_SUCCESS`; validate `num_prep_blocks`, `num_cooldown_blocks`, `num_passes`, `pass_len`, `tr_size`, `degenerate_prep`, `degenerate_cooldown`
   - Tests 05–06: expect `ERR_TR_PREP_TOO_LONG` / `ERR_TR_COOLDOWN_TOO_LONG`
   - Tests 07, 12, 13: expect `ERR_INVALID_ONCE_FLAGS`

   **Suite B — Metadata tests** (all 25 segmentation sequences): Load each `.seq` with `gre_opts_init` + matching `num_averages`, call `pulseqlib_get_subseq_info`, compare against `_meta.txt` fields.

   **Suite C — Segment definition tests** (6–8 representative `_1sl_1avg` variants): Call `pulseqlib_get_segment_block_def_indices` for each segment, compare against `_segments.txt`.

   **Suite D — Waveform tests** (6 types × 2 modes × 3 axes = 36 comparisons): For each `_1sl_1avg` variant, call `pulseqlib_get_tr_waveforms` with `amplitude_mode=0` (MAX_POS) and the new MIN_POS mode. Compare corner-point arrays against `_tr_max_*` and `_tr_min_*` CSVs (tolerance: time ± 1 raster, amplitude ± 0.1 Hz/m).

   **Suite E — Anchor tests** (6 types × 2 modes): Access internal `desc->segment_definitions[0].rf_anchors[0].isocenter_us` and `.adc_anchors[0].kzero_us`, compare against `_tr_*_anchors.txt` (tolerance ± 20 µs).

   **Suite F — Scan table tests**: For single-pass sequences (`_1sl_1avg`, `_1sl_4avg`), compare `desc->scan_table_block_idx` against `_scan_table.csv`. Multi-pass scan table tests deferred to Wave 2 (after generator regen).

8. **Wire up** `test_segmentation_main()` in `tests/ctests/test_runner.c` and add forward declaration to `tests/ctests/test_helpers.h`. Build is automatic (CMakeLists globs `*.c`).

### D. C-Side Algorithmic Changes

9. **Replace MIN_ABS with MIN_POS** in `csrc/pulseqlib_waveforms.c`:
   - `amplitude_mode=1` changes from "minimum absolute amplitude, discard sign" to "minimum absolute amplitude, **preserve sign**" (return the signed value of the shot whose `|amplitude|` is smallest at each raster position)
   - Remove any code path that stripped the sign for mode 1
   - Update the mode enum/comments in `csrc/pulseqlib_types.h` or wherever it's documented
   - Update AGENT.md §3 waveform mode descriptions

10. **Per-ADC kzero refinement** in `csrc/pulseqlib_safety.c` (lines 458–493):
    - Replace "nearest global zero crossing" with **local krss minimum within each ADC's time window**
    - For each ADC event: compute start/end raster samples from ADC onset/duration, scan `krss[start..end]` for minimum, convert to ADC-local sample index
    - Keep `find_kspace_zero_crossings` and `num_kzero_crossings` for the public `pulseqlib_segment_info` API
    - Keep `N/2` fallback for blocks outside the TR or when trajectory computation fails

### E. MATLAB Ground Truth Regeneration

11. After steps 1–4 are done, **regenerate** on MATLAB machine:
    - All 13 once-flag `.seq` files
    - All 25 segmentation ground truth datasets (`.seq` + `_meta.txt` + `_segments.txt` + `_scan_table.csv` + waveform CSVs + anchor files)
    - The `_tr_min_*` CSVs will need to match the new MIN_POS semantics

### F. Wave 2 — Post-Regen Completion

12. After regenerated data is pulled:
    - Enable multipass scan table tests in Suite F
    - Validate all waveform/anchor tests against new ground truth
    - Run full test suite: existing 26 + new ~80+ tests

**Verification**

- Build: `cd tests/ctests/build && cmake .. && cmake --build .`
- Run: `./bin/run_tests`
- Wave 1 target: ~50+ new tests passing (suites A–E for single-pass, suite F single-pass only)
- Wave 2 target: ~80+ total new tests (all suites including multipass scan table)

**Decisions**
- Once-flag generator: merge ONCE=0 into first main block (matches segmentation generators)
- Remove tests 05, 06 (ramp blocks break periodicity), 07 (ambiguous), 14 (trailing cooldown unsupported)
- Remove MIN_ABS mode entirely; replace with MIN_POS (preserve sign)
- Per-ADC kzero: local krss minimum within ADC window (physically correct for EPI/FSE/non-Cartesian)
- Two-wave: implement Wave 1 with existing data, complete Wave 2 after MATLAB regen
- Waveform tests: `_1sl_1avg` only (waveforms type-dependent, not slice/avg-dependent)
- Anchor tolerance: ±20 µs (1 gradient raster)
