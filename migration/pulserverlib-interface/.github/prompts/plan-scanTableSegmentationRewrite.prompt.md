# Plan: Scan-Table-Only Segmentation Rewrite

## Summary

Remove the block-table segmentation path entirely. Rewrite `get_scan_table_segments` to operate on the **first pass** only, with three-section (prep/main/cooldown) retry logic based on gradient boundary checks. Fill segment tables with prep/main/cooldown sections (matching the existing `segment_table_result` structure). Tile `scan_table_seg_id` across all passes. Add cross-pass RF/shim consistency checking in `check_consistency`. Remove ~1050 lines of dead code.

---

## Step 1 — Rewrite `get_scan_table_segments`

**File**: `csrc/pulseqlib_structure.c` (lines 2149–2579)

- Compute `pass_size = scan_len / num_passes` (where `num_passes = desc->num_passes >= 1`). All segmentation operates on `[0, pass_size)` only.
- Compute section boundaries from known TR descriptor values: `num_prep = desc->tr_descriptor.num_prep_blocks`, `tr_size = desc->tr_descriptor.tr_size`, `num_cool = desc->tr_descriptor.num_cooldown_blocks`. No `scan_table_tr_id` usage.
- Compute `max_allowed = opts->max_slew_hz_per_m_per_s * desc->grad_raster_us * 1e-6f`.
- Compute `max_mult` as the number of TRs × num_averages, bounded by pass geometry: roughly `(pass_size - num_prep - num_cool) / tr_size`.

### Prep section (if `num_prep > 0` and `!degenerate_prep`)

- For k=1,2,...,max_mult: region `[0, num_prep + k*tr_size)`.
- Before each call, check boundary gradients via new `scan_boundary_gradients_ok()`: first samples of scan position 0 and last samples of scan position `num_prep + k*tr_size - 1` resolved via `scan_table_block_idx`. Skip retry if not near-zero.
- Call `find_segments_on_scan_table(desc, raw_segs, 0, diag, opts, scan_table_block_idx, 0, region_size)`.
- If success: record `n_prep_raw`; if `region_size >= pass_size`, set `all_covered=1` and skip main+cooldown.
- If failure with `SEG_NONZERO_START/END_GRAD`: retry with next k.
- Other error: fail.

### Main section (if `!all_covered`)

- For k=1,2,...,max_mult: region `[num_prep, num_prep + k*tr_size)`.
- Same boundary + retry logic.
- If success: record `n_main_raw`; if `num_prep + k*tr_size >= pass_size`, set `all_covered=1`.
- When `k > 1` succeeded: update `desc->tr_descriptor.tr_size` and `num_trs` accordingly.

### Cooldown section (if `!all_covered` and `num_cool > 0` and `!degenerate_cooldown`)

- For k=1,2,...,max_mult: region `[pass_size - num_cool - k*tr_size, pass_size)`.
- Same boundary + retry logic.
- **Final fallback**: if all retries fail, try `[0, pass_size)` (entire first pass).

### Post-processing (per-section)

- **Populate `unique_block_indices`** for each raw segment using `scan_pat[raw_segs[n].start_block + i]` (block def IDs via scan-table indirection).
- **Strip pure delays** per section using `strip_pure_delays_scan`, tracking `n_prep`, `n_main`, `n_cool` counts separately.
- **NAV split/merge** per section (when PMC enabled) using `nav_split_merge` with `scan_bi = desc->scan_table_block_idx`, maintaining per-section counts.

### Dedup and segment tables

- **Build segment tables** with prep/main/cooldown split:
  - `desc->segment_table.num_prep_segments = n_prep`, `.prep_segment_table`
  - `desc->segment_table.num_main_segments = n_main`, `.main_segment_table`
  - `desc->segment_table.num_cooldown_segments = n_cool`, `.cooldown_segment_table`
- **Dedup** across all sections (existing logic: pure-delay collapse + `array_equal` matching). Each expanded segment maps to its unique ID in the corresponding section table.
- **Transfer segments** to `desc->segment_definitions`. Convert `start_block` from scan-table position to block-table index via `desc->scan_table_block_idx`.
- **Per-block flags** (digitalout, rotation, norot, nopos, trigger) — walk expanded segments using scan-table indirection.
- **Max energy** — same scan-table-based walk as current.
- **NAV tagging** — same as current.

### Fill `scan_table_seg_id`

- Build a `pattern_seg_id[pass_size]` from all three sections (prep positions, main positions, cool positions).
- Tile across the full scan table: `scan_table_seg_id[n] = pattern_seg_id[n % pass_size]`.

---

## Step 2 — Add scan-table boundary pre-check

**File**: `csrc/pulseqlib_structure.c`

New static function:

```c
static int scan_boundary_gradients_ok(
    const pulseqlib_sequence_descriptor* desc,
    const int* scan_block_idx,
    int first_scan_pos, int last_scan_pos,
    float max_allowed);
```

Resolves block-table indices via `scan_block_idx[pos]` and checks gradient first/last values on the per-instance shot index. Returns 1 if OK, 0 if any axis violates. Replaces `boundary_gradients_ok` which operates on block-table indices directly.

---

## Step 3 — Simplify core pipeline

**File**: `csrc/pulseqlib_core.c` (lines 528–547)

- Remove block-table segmentation call (`get_segments_in_tr`) and its fallback logic.
- Remove `fill_scan_seg_id_from_blocktable` call.
- Single call: `result = pulseqlib__get_scan_table_segments(&desc, diag, &raw->sequences[i].opts)`.
- The `const pulseqlib__seq_file* seq` parameter is no longer needed by the scan-table path (it only uses `opts`).

**Before** (current):
```c
/* Try blockTable-based segmentation first */
result = pulseqlib__get_segments_in_tr(&desc, diag, &raw->sequences[i]);
if (PULSEQLIB_FAILED(diag->code)) {
    if (diag->code == PULSEQLIB_ERR_SEG_NONZERO_START_GRAD ||
        diag->code == PULSEQLIB_ERR_SEG_NONZERO_END_GRAD) {
        pulseqlib_diagnostic_init(diag);
        result = pulseqlib__get_scan_table_segments(&desc, diag, &raw->sequences[i].opts);
        if (PULSEQLIB_FAILED(diag->code)) goto fail;
    } else {
        goto fail;
    }
} else {
    result = pulseqlib__fill_scan_seg_id_from_blocktable(&desc);
    if (PULSEQLIB_FAILED(result)) { diag->code = result; goto fail; }
}
```

**After**:
```c
result = pulseqlib__get_scan_table_segments(&desc, diag, &raw->sequences[i].opts);
if (PULSEQLIB_FAILED(diag->code)) goto fail;
```

---

## Step 4 — Add cross-pass RF/shim consistency check

**File**: `csrc/pulseqlib_core.c` (in `check_consistency`, after existing RF periodicity checks)

When `desc->num_passes > 1`:

- Compute `pass_size = desc->scan_table_len / desc->num_passes`.
- For each scan-table position `n` in `[pass_size, scan_table_len)`:
  - Compare RF **amplitude** against position `n % pass_size` (the corresponding position in pass 0).
  - Compare **shim ID** against position `n % pass_size`.
  - RF amplitude: resolve via `scan_table_block_idx[n] → block_table[idx].rf_id → rf_table[rf_id].amplitude`.
  - Shim ID: resolve via `scan_table_block_idx[n] → block_table[idx].rf_shim_id`.
  - On mismatch: return `PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC` or `PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC`.

**Note**: Do NOT compare `freq_offset` or `phase_offset` — these legitimately differ across passes (multi-slice selection). Only amplitude (flip-angle schedule) and shim ID need to match.

> **Known limitation**: multipass folding discards `block_table` entries for passes 2..N, so `scan_table_block_idx` for pass-2+ maps back to pass-1 entries. This means the cross-pass check currently compares pass-1 against itself (always passes). The check becomes meaningful only after folding is fixed to preserve per-pass block_table entries (separate future task). We add the check now so that the infrastructure is in place.

---

## Step 5 — Remove dead code

**File**: `csrc/pulseqlib_structure.c`

Remove the following static functions (only used by the block-table segmentation path):

| Function | Lines | Description |
|---|---|---|
| `find_segments_internal` | 583–757 | Block-table segment state machine |
| `strip_pure_delays` (non-scan) | 759–823 | Block-table pure delay stripping |
| `boundary_gradients_ok` | 985–1022 | Block-table boundary pre-check |
| `pulseqlib__get_segments_in_tr` | 1024–1510 | Block-table three-section entry point |
| `pulseqlib__fill_scan_seg_id_from_blocktable` | 1512–1626 | Backfill scan_table_seg_id from block-table results |

**Total removal**: ~1044 lines

---

## Step 6 — Remove dead declarations

**File**: `csrc/pulseqlib_internal.h` (lines 982–983)

Remove:
```c
int   pulseqlib__get_segments_in_tr(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag, const pulseqlib__seq_file* seq);
int   pulseqlib__fill_scan_seg_id_from_blocktable(pulseqlib_sequence_descriptor* desc);
```

---

## Step 7 — Update AGENT.md

- **§3 step 6**: update to say segmentation operates on the scan table directly (not block table)
- **§7.1**: clarify state machine runs on scan-table positions (via `find_segments_on_scan_table`)
- **§7.2**: rewrite to describe scan-table three-section retry (prep/main/cool within first pass, expanding by tr_size multiples, with cooldown final fallback to entire pass)
- **§7.3**: update to note per-section strip/NAV/dedup all use scan-table indirection
- **§7.4**: rewrite entirely — no longer a "fallback", this IS the primary (and only) path. Describe first-pass-only operation + tiling to all passes.
- **§7.5**: update consistency check description to include cross-pass RF/shim validation
- **§14**: remove "segmentation can fall back" bullet, add note about first-pass segmentation + pass tiling

---

## Verification

1. Build the C test suite: `cd tests/ctests/build && cmake .. && cmake --build .`
2. Run all 26 existing tests: `ctest --output-on-failure`
3. All tests should pass unchanged (the scan-table path now produces the same segment tables as the old block-table path, including prep/main/cooldown split)
4. Manual sanity check: verify that a multi-pass sequence with `num_passes > 1` has consistent `scan_table_seg_id` across passes

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Segmentation scope | First pass only (`pass_size = scan_len / num_passes`), then tile `seg_id` to all passes | Passes are structurally identical (verified by folding) |
| Section boundaries | `num_prep_blocks`, `tr_size`, `num_cooldown_blocks` from `tr_descriptor` | No `scan_table_tr_id` needed |
| Cross-pass RF/shim check | In `check_consistency` (post-load), not in segmentation | Matches existing check architecture |
| Cooldown final fallback | `[0, pass_size)` if all k-retries fail | Covers edge cases where no clean boundary exists |
| Prep minimum region | `num_prep + 1*tr_size` (k starts at 1) | Prep always includes at least 1 TR of main |
| Folding data loss | Deferred to separate future task | Pre-existing issue; segmentation only needs structural info |
| Compare freq/phase across passes | No — only amplitude and shim ID | freq/phase legitimately differ (multi-slice) |
