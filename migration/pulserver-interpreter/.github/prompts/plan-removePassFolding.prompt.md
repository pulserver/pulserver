# Plan: Remove Pass Folding from `get_unique_blocks`

## Summary

Remove the pass-folding step from `pulseqlib__get_unique_blocks` so that the full (unfolded) block table is preserved across all passes. This retains per-instance RF/ADC freq/phase data that folding currently discards. Add a `pass_len` field to `sequence_descriptor`. Replace whole-pass identity verification with per-section (prep/main/cooldown) verification. Restrict TR pattern search to the first pass. Rewrite `build_scan_table` to emit per-pass block-table indices. Activate the cross-pass RF/shim consistency check (currently a structural no-op due to folding).

---

## Step 1 — Add `pass_len` to `sequence_descriptor`

**File**: `csrc/pulseqlib_internal.h` (L423–505)

- Add `int pass_len;` after `num_passes` (L433). This is the number of block-table entries per pass (prep + main + cooldown within one pass).
- For single-pass sequences, `pass_len = num_blocks`.
- Update `PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT` (L494) to include `0` for `pass_len`.

**After** (struct excerpt):
```c
    int num_passes;
    int pass_len;           /**< blocks per pass (= num_blocks when single-pass) */
    int vendor;
```

---

## Step 2 — Remove folding, add per-section pass verification

**File**: `csrc/pulseqlib_dedup.c` (L1420–1530)

Replace the current multipass block (trailing-cooldown handling + whole-pass identity check + fold step + post-fold recount) with a new algorithm that **preserves the full block table** and verifies passes per-section.

### Phase A — Pass boundary detection (keep as-is)

Detect boundaries by `once_flag` transitions: `cur == first_once && prev_once_val != first_once`. Produces `pass_starts[]` and `num_passes_found`.

### Phase B — Reject uneven passes

Verify `num_blocks == num_passes_found * pass_len` where `pass_len = pass_starts[1] - pass_starts[0]`. No trailing-cooldown special-casing: the last pass's cooldown section runs to EOF and is naturally the same length as every other pass's cooldown because the boundary detector doesn't see a transition after the last pass.

If the check fails, return `PULSEQLIB_ERR_INVALID_ONCE_FLAGS`.

### Phase C — Count section sizes within first pass

Walk `block_table[pass_starts[0] .. pass_starts[0] + pass_len - 1]`:
- `num_prep_in_pass`: count leading blocks with `once_flag == 1`
- `num_cool_in_pass`: count trailing blocks with `once_flag == 2`
- `num_main_in_pass`: `pass_len - num_prep_in_pass - num_cool_in_pass`

Update `desc->num_prep_blocks = num_prep_in_pass` and `desc->num_cooldown_blocks = num_cool_in_pass`.

### Phase D — Fill reference definition arrays

Allocate three arrays holding `(id, once_flag)` pairs from the first pass:
- `prep_def[num_prep_in_pass]` from `block_table[pass_starts[0] .. pass_starts[0] + num_prep_in_pass - 1]`
- `main_def[num_main_in_pass]` from `block_table[pass_starts[0] + num_prep_in_pass .. ...]`
- `cool_def[num_cool_in_pass]` from `block_table[... .. pass_starts[0] + pass_len - 1]`

(In practice, compare directly against block_table; no separate allocation needed.)

### Phase E — Compare passes 1..N-1 per section

For each pass `p = 1 .. num_passes_found - 1`:
- **Prep section**: compare `block_table[pass_starts[p] + j]` `.id` and `.once_flag` against `block_table[pass_starts[0] + j]` for `j = 0 .. num_prep_in_pass - 1`
- **Main section**: compare `block_table[pass_starts[p] + num_prep_in_pass + j]` `.id` and `.once_flag` against reference for `j = 0 .. num_main_in_pass - 1`
- **Cooldown section**: compare `block_table[pass_starts[p] + num_prep_in_pass + num_main_in_pass + j]` `.id` and `.once_flag` against reference for `j = 0 .. num_cool_in_pass - 1`

On mismatch → `PULSEQLIB_ERR_INVALID_ONCE_FLAGS`.

### Phase F — Set descriptor fields (NO folding)

```c
desc->num_passes = num_passes_found;
desc->pass_len   = pass_len;
/* num_blocks stays as-is — full unfolded block table preserved */
```

For single-pass sequences (when `once_counter` matches expected), set `desc->pass_len = desc->num_blocks` before returning.

---

## Step 3 — Restrict `get_tr_in_sequence` to first pass

**File**: `csrc/pulseqlib_structure.c` (L262–577)

Currently `get_tr_in_sequence` operates on `[0, num_blocks)`. Change to `[0, pass_len)`:

- `imaging_end = desc->pass_len - desc->num_cooldown_blocks` (was `desc->num_blocks - desc->num_cooldown_blocks`)
- Pattern arrays (`seq_pat`, `block_dur`) sized to `desc->pass_len` (was `desc->num_blocks`)
- `first_repeating_segment` and period-finding loops bounded by `pass_len`
- Single-TR fallback: `tr_size = imaging_len` uses `pass_len`-derived `imaging_len`
- Validation checks: replace `desc->num_blocks` with `desc->pass_len` throughout

---

## Step 4 — Rewrite `build_scan_table` for per-pass block indexing

**File**: `csrc/pulseqlib_structure.c` (L98–205)

Currently the inner loop iterates `blk = 0 .. num_blocks-1` and writes `scan_table_block_idx[idx] = blk` — all passes share the same folded-pass indices.

Change to per-pass indexing:
- Inner loop: `blk = 0 .. pass_len - 1` (was `num_blocks`)
- Block-table base offset: `base = pass * pass_len`
- Index computation: `scan_table_block_idx[idx] = base + blk` (was just `blk`)
- Once-flag lookup: `desc->block_table[base + blk].once_flag` (was `desc->block_table[blk].once_flag`)

**Before** (fill loop):
```c
for (blk = 0; blk < desc->num_blocks; ++blk) {
    once = desc->block_table[blk].once_flag;
    if (once == 1 && play_prep) {
        desc->scan_table_block_idx[idx] = blk;
```

**After**:
```c
int base = pass * desc->pass_len;
for (blk = 0; blk < desc->pass_len; ++blk) {
    once = desc->block_table[base + blk].once_flag;
    if (once == 1 && play_prep) {
        desc->scan_table_block_idx[idx] = base + blk;
```

Both the count pass and the fill pass need this change.

---

## Step 5 — RF periodicity checks: no change needed

**File**: `csrc/pulseqlib_core.c` (L183–266)

`check_rf_amplitude_periodicity` and `check_rf_shim_periodicity` index `block_table[prep_blocks + tr_idx * tr_size + ...]`. These operate within first-pass scope via `num_trs` (which is derived from first-pass `imaging_len`). Since the block table is now unfolded but these only read first-pass data, no change is required.

Additionally, `num_averages` intrinsically produces periodic RF amp/shim patterns across the full block table (all passes have identical structural definitions), so periodicity checks on the full block table remain valid.

---

## Step 6 — Activate cross-pass RF/shim consistency check

**File**: `csrc/pulseqlib_core.c` (L270–327)

`check_cross_pass_rf_consistency` already implements the full comparison logic. The only change needed is removing the comment that says it's structurally a no-op:

```c
/*
 * check_cross_pass_rf_consistency --
 *   For multi-pass sequences, verify that the RF amplitude and shim ID
 *   patterns are identical across passes.  Pass 0 is the reference;
 *   passes 1..N-1 are compared position-by-position.
 */
```

With the unfolded block table, `scan_table_block_idx` for pass N now points to pass N's actual block-table entries, making the comparison meaningful.

---

## Step 7 — Verify segmentation `pass_size` unchanged

**File**: `csrc/pulseqlib_structure.c` (L1305)

`get_scan_table_segments` computes `pass_size = scan_table_len / num_passes`. After this refactor:
- `scan_table_len` is the same (pass × avg expansion of `pass_len` blocks per pass)
- `num_passes` is the same
- `pass_size` is the same

No change needed. Segmentation still operates on `[0, pass_size)` of the scan table and tiles `seg_id` across all passes.

---

## Step 8 — Build and test

1. Build: `cd tests/ctests/build && cmake .. && cmake --build .`
2. Run all 26 existing tests: `ctest --output-on-failure`
3. All tests must pass unchanged.
4. Manual sanity check: verify a multi-pass sequence produces distinct `scan_table_block_idx` values per pass.

---

## Step 9 — Update AGENT.md

- **§3 pipeline step 1**: `get_unique_blocks` no longer folds; preserves full block table, sets `pass_len`
- **§3 pipeline step 2**: `get_tr_in_sequence` operates on `[0, pass_len)` of the first pass
- **§3 pipeline step 3**: `build_scan_table` emits per-pass block-table offsets
- **§14 Gotchas**: remove "folding discards per-pass data" bullet; add note about `pass_len` vs `num_blocks`

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Trailing cooldown | Not special-cased | Last cooldown is identical to others; boundary detector naturally stops at EOF |
| Pass verification | Per-section (prep/main/cool separately) | Catches section-level mismatches that whole-pass comparison misses |
| Pass verification fields | `.id` and `.once_flag` only | Dynamic fields (freq/phase) legitimately vary across passes |
| RF periodicity checks | Stay on block_table, no change | First-pass scope via `num_trs`; periodic by construction across passes |
| Cross-pass RF check | Activate (remove no-op comment) | Becomes meaningful with unfolded block table |
| `pass_len` naming | `pass_len` (not `pass_size`) | `pass_size` already used in scan-table context for expanded size |
| Single-pass default | `pass_len = num_blocks` | Seamless fallback; all downstream code uses `pass_len` uniformly |
