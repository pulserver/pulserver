# Rename map v3 — pulseq module split, static-helper arg order, PulSeg vocabulary

Covers the refactor stages after test-01 (`docs/rename_map_v2.md`):

| Section | Stage |
|---|---|
| §1 raw-Pulseq module split | `PLAN_refactor_stage8_pulseq_module.md` |
| §2 static-helper argument order | `PLAN_test_01_c_api_hygiene.md` convention, applied below the public API |
| §3 PulSeg-spec vocabulary | `PLAN_refactor_stage9_spec_alignment.md` step 1 |

The convention all three sections are measured against (normative,
`PLAN_vendor_neutral_refactor.md` §2):

- method: `pulseg_<class>_<method>(<class> *self, <outputs>, <inputs>)`
- pure fn: `pulseg_<module>_<fn>(<outputs>, <inputs>)`
- ctor/dtor: `pulseg_<class>_init` / `pulseg_<class>_free`
- `const <class>*` self counts as self; a `FILE*`/path destination of a
  writer is an input (side-effect output), not an out-param.

**Buffer-pair rule (clarified here, §2.2).** A pointer and the extent or
capacity argument that immediately follows it (`char *buf, int bufsz`;
`int *out_sizes, int max_files`; `pulseg_virtual_segment *out, int max_out`)
form ONE logical argument for ordering purposes. Ordering applies to
buffer pairs, never to their halves separately.

---

## 1. Raw-Pulseq module split (stage 8)

`csrc/{include,src}/pulseq/` became a standalone module compilable with zero
pulseg headers. The `pulseg_pulseq_*` family lost its `pulseg_` prefix:

| Old (pulseg) | New (pulseq) |
|---|---|
| `pulseg_pulseq_file` | `pulseq_file` |
| `pulseg_pulseq_file_set` | `pulseq_file_set` |
| `pulseg_pulseq_block` | `pulseq_block` |
| `pulseg_shape_arbitrary` | `pulseq_shape` (pulseg keeps a typedef alias) |
| `pulseg_pulseq_file_read` | `pulseq_read` |
| `pulseg_pulseq_file_read_from_buffer` | `pulseq_read_from_buffer` |
| `pulseg_pulseq_file_set_read` | `pulseq_file_set_read` |
| `pulseg_pulseq_file_free` / `_set_free` | `pulseq_file_free` / `pulseq_file_set_free` |
| `pulseg_pulseq_block_init` / `_free` | `pulseq_block_init` / `pulseq_block_free` |
| `pulseg_pulseq_get_block` | `pulseq_get_block` |
| `pulseg_pulseq_get_raw_block_content_ids` | `pulseq_get_raw_block_content_ids` |
| `pulseg_pulseq_get_raw_extension` | `pulseq_get_raw_extension` |
| `pulseg_pulseq_decompress_shape` | `pulseq_decompress_shape` |
| `pulseg_pulseq_verify_signature` | `pulseq_verify_signature` |
| `PULSEG_ALLOC` / `PULSEG_FREE` (inside the module) | `PULSEQ_ALLOC` / `PULSEQ_FREE` |
| `PULSEG_ERR_*` (inside the module) | `PULSEQ_ERR_*` (same numeric values, so the boundary needs no mapping) |
| `extract_base_path` / `build_full_path` (were static) | `pulseq_path_dirname` / `pulseq_path_join` |

pulseg keeps typedef aliases for the leaf types, so IR-facing code and the
`.pge`/`.pseg` cache layout are untouched.

### 1.1 `raster_fallback` → `design_raster`

`pulseq_file` no longer embeds `pulseg_opts`. It carries only the rasters
the caller designed the sequence on, consulted for the ones the `.seq`
file's `[DEFINITIONS]` omits:

| Old | New |
|---|---|
| `pulseq_file.raster_fallback` | `pulseq_file.design_raster` |
| `pulseg_opts_get_pulseq_raster()` | `pulseg_opts_get_design_raster()` |

"Fallback" described the lookup order, not the thing: it is a design-time
property of the sequence, not a stand-in for a missing value. System
(hardware) rasters remain the layer above's concern — `pulseq` neither
stores nor consults them. `pulseg_convert_collection()` therefore takes an
explicit `opts` parameter, since the system half can no longer arrive
inside the parsed file.

---

## 2. Static-helper argument order

The public headers were brought to the convention in test-01 (185/187
conforming). The 44 candidate violations below the API surface were
unreviewed drift; the genuine ones are now fixed. All are file-local
`static` functions, so every call site is in the same translation unit.

### 2.1 Applied reorderings

| File | Function | New order |
|---|---|---|
| `freqmod/pulseg_freqmod.c` | `build_freq_mod_for_block` | `(desc, fmod, bte, bdef_idx, active_start_us, active_end_us, ref_time_us, target_raster_us)` |
| `freqmod/pulseg_freqmod.c` | `freq_mod_library_get` | `(lib, out_hw_waveform, out_num_samples, out_phase_rad, exec_stream_pos)` — now matches its public sibling `pulseg_get_freq_mod` (v2 row 3) |
| `safety/pulseg_safety.c` | `pulseg__select_canonical_tr_window_idx` | `(desc, start_block, block_count, amplitude_mode, num_instances, tr_duration_us, canonical_tr_idx)` |
| `safety/pulseg_safety.c` | `pulseg__build_pass_expanded_block_order` | `(desc, out_block_order, out_block_count, out_duration_us, pass_base)` |
| `safety/pulseg_safety.c` | `sa_transform_cache_lookup` | `(cache, out_re, out_im, def_id)` |
| `safety/pulseg_safety.c` | `sa_eval_pwl_transform` | `(out_re, out_im, f_hz, t_us, v, n_vtx)` |
| `safety/pulseg_safety.c` | `sa_eval_event_transform` / `_spectrum` / `_line` | `(ev, out_re, out_im, f_hz, cache)` — the family was split between `(ev, f_hz, …)` and `(f_hz, ev, …)`; now uniform |
| `safety/pulseg_safety.c` | `sa_eval_axis_spectrum` | `(ae, out_re, out_im, f_hz)` |
| `structure/pulseg_getters.c` | `pulseg__module_block_signature` | `(desc, out_block_def_id, out_adc_def_id, blk_idx)` |
| `structure/pulseg_getters.c` | `pulseg__get_rf_def_shape` | `(desc, num_channels, num_samples, rdef, which_shape)` |
| `structure/pulseg_getters.c` | `get_block_range` | `(desc, out_start, out_count, tr_type, tr_index)` |
| `structure/pulseg_seqdesc.c` | `seqdesc__build_adc_anchors_from_canonical` | `(coll, out_kzero, out_roles, desc, subseq_idx, block_start, n_walk)` |
| `structure/pulseg_structure.c` | `record_adc_label` | `(table_row, num_columns, limits, state, label_column_map, is_first)` |
| `structure/pulseg_structure.c` | `find_kspace_zero_crossings` | `(zero_indices, out_count, krss, n, threshold, count_only)` |
| `waveforms/pulseg_trajectory.c` | `decompress_block_arb` | `(desc, out_xp, out_fp, out_n, grad_event_id, grad_raster_us)` |
| `waveforms/pulseg_trajectory.c` | `traj_slice_canonical_axis` | `(out_k, adc_num_samples, canonical_k, n_samples, dt_us, block_time_offset_us, adc_delay_us, adc_dwell_us)` |
| `waveforms/pulseg_trajectory.c` | `compute_block_kspace` | `(desc, out_kx, out_ky, out_kz, out_num_samples, out_gx_const, out_gy_const, out_gz_const, block_table_idx, kzero_index, diag)` |
| `waveforms/pulseg_waveforms.c` | `compute_position_max_amplitudes_filtered` | `(desc, pos_max_gx, pos_max_gy, pos_max_gz, block_start, block_count, tr_group_labels, target_group)` |
| `waveforms/pulseg_waveforms.c` | `fill_rf_waveform_for_flat_block` | `(desc, time_mag, mag, phase, out_nch, block_idx, start_idx, t0)` |
| `io/pulseg_protocol.c` | `try_parse_rich` | `(pv, valstr)` |
| `recon/pulseg_recon.c` | `pgr_read_one_definitions_block`, `pgr_read_definitions_section`, `pgr_read_rf_shape`, `pgr_read_rf_def_library`, `pgr_read_one_seqdesc_subseq`, `pgr_read_seqdesc_section` | destination first, then `FILE *f`, then `do_swap`/`section_offset`, then the `diag`/`diag_size` sink — matching the public reader `pulseg_freq_mod_collection_read_cache_f(out_fmc, f, coll, shift_m, do_swap)` |
| `pulseq/pulseq_parse.c` | `init_standard_library`, `init_definitions_library`, `init_shapes_library`, `init_rf_shim_library`, `read_standard_library`, `read_label_library`, `read_delay_library`, `read_rf_shim_library` | `(target, target_count, [is_*_defined,] f, offset, …)` — same reader shape as above |

### 2.2 Justified exclusions

| Function(s) | Why it looks like a violation | Why it is not |
|---|---|---|
| `read_line_to`, `read_line`, `read_preamble_block` (`core/pulseg_bridge.c`) | `int bufsz` follows `char *buf` before `timeout_sec` | buffer pair (`buf`, `bufsz`) — conforming |
| `next_pipe_field` (`io/pulseg_protocol.c`) | `dstsz` after `dst` | buffer pair; `pp` is an in/out cursor (self) |
| `nav_split_merge`, `strip_pure_delays_scan` (`structure/pulseg_structure.c`) | source array pair precedes the destination pair | Siblings called back-to-back on the same data with the same shape. `nav_split_merge`'s leading pair is in/out (it frees and rebuilds `in[n].unique_block_indices`), so it IS the self argument; giving the two functions different argument orders would read worse than the drift they'd fix. |
| `compute_rf_bandwidth_fft` (`structure/pulseg_dedup.c`) | four trailing non-const pointers | `work_re`, `work_im`, `fft_in`, `fft_out` are caller-provided scratch, not results — scratch counts as an input |
| `record_adc_label`'s `table_row` | non-const pointer first | it IS the output; `num_columns` is its extent (buffer pair) |

### 2.3 Dead code removed in the same pass

| File | Removed |
|---|---|
| `cache/pulseg_cache.c` | `get_seq_file_sizes()` — no callers; kept alive only by a `(void)get_seq_file_sizes;` unused-function suppression, and two of its four parameters were `(void)`-discarded |
| `waveforms/pulseg_trajectory.c` | `linear_resample()` and `expand_block_axis_grad()` — both inside `#if 0` blocks labelled "intentionally disabled legacy helpers", with comments referencing file paths that no longer exist |

---

## 3. PulSeg-spec vocabulary (stage 9 step 1)

Renames only — zero layout changes. `.pge` fixtures stay byte-identical
(`4c4f3738393b27be2fdf865c4f74aa88` over all 53, unchanged).

| Old | New | Spec term |
|---|---|---|
| `pulseg_block_definition` | `pulseg_base_block` | BaseBlock (§3.1) |
| `PULSEG_BLOCK_DEFINITION_INIT` | `PULSEG_BASE_BLOCK_INIT` | — |
| `pulseg_sequence_descriptor.block_definitions` | `.base_blocks` | — |
| `pulseg_tr_segment` | `pulseg_virtual_segment` | VirtualSegment (§3.2) |
| `PULSEG_TR_SEGMENT_INIT` | `PULSEG_VIRTUAL_SEGMENT_INIT` | — |
| `scan_table*` (every identifier) | `exec_stream*` | Execution stream (§3.4) |
| `pulseg__build_scan_table` | `pulseg__build_exec_stream` | — |
| `pulseg__get_scan_table_segments` | `pulseg__get_exec_stream_segments` | — |
| `pulseg__compute_scan_table_tr_start` | `pulseg__compute_exec_stream_tr_start` | — |
| `scan_table_pos` (public param, `pulseg_freqmod.h`) | `exec_stream_pos` | — |

### 3.1 Not renamed, and why

- **`block_table` / `pulseg_block_table_element` / `bte`.** The spec term
  for its contents is *SegmentInstance* (§3.3), but `pulseg_block_instance`
  is already taken by the public resolved *view* over that table
  (`pulseg_get_block_instance`). Renaming the storage to the view's name
  would collide; renaming the view would break the public API for a purely
  cosmetic gain. The §2 concept mapping already records the correspondence.
- **Truth-fixture filenames** (`*_scan_table.truth`, `*_scan_table.bin`,
  and the `"_scan_table"` string literals in `tests/ctests/test_sequences.c`,
  `tests/cpptests/{dump_recon_cache,test_seq_desc_truth,test_trajectory_cache_loader}.cpp`,
  `tests/utils/expected/MANIFEST.json`). These name artifacts produced by
  the MATLAB truth generator; renaming them would force a truth
  regeneration, which is exactly what the byte-equality gate exists to
  prevent. The C identifiers are `exec_stream*`; the artifacts on disk keep
  their generated names.

### 3.2 Blast radius

Confined to `external/pulserver/`. None of these symbols appear in
`src_psd`, `src_gelib`, `src_common`, `src_rtp`, `src_vre`, or any `.e`
file — the private repo reaches the data model only through public getters,
whose *parameter* names are invisible to callers. The symlink farm was
regenerated (`scripts/regen_pulseglib_links.sh`: 46/46 links resolve).

---

## 4. New API (stage 9 step 2)

`pulseg_get_block_instance_at(coll, inst, subseq_idx, exec_stream_position)`
— the random-access form of `pulseg_get_block_instance()`, resolving the
per-instance view (PulSeg SegmentInstance §3.3: amplitude / phase /
frequency / shot index / rotation / duration) without moving or consulting
the cursor. It is a VIEW: no per-instance storage was added, the values are
read from the existing block / RF / gradient / ADC / rotation tables on
each call. Both entry points now share one `resolve_block_instance()`
implementation.

Exposed as `pulseg::Collection::get_block_instance_at(ss, pos)` (cxx). The
pybind binding was dropped when the Python analysis layer was removed; the
C and C++ entry points are unchanged.

Covered by `tests/ctests/test_block_instance.c` on one cartesian
(`gre_2d_3sl_3avg`) and one rotated (`mprage_noncart_3d_3sl_1avg_userotext1`)
fixture: every execution-stream position must equal both the underlying
tables and the cursor-driven view.
