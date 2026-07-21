# Rename map v2 — public C API hygiene audit (test-01)

Audit date: 2026-07-08. Scope: every function declared in
`csrc/include/pulseg/*.h` (14 headers), checked against the pseudo-OOP
convention (`PLAN_vendor_neutral_refactor.md` §2 / `PLAN_test_01...md`):

- method: `pulseg_<class>_<method>(<class> *self, <outputs>, <inputs>)`
- pure fn: `pulseg_<module>_<fn>(<outputs>, <inputs>)`
- ctor/dtor: `pulseg_<class>_init` / `pulseg_<class>_free`
- `const <class>*` self counts as self; a FILE*/path destination of a
  writer is an input (side-effect output), not an out-param.

Functions not listed below were audited and found conforming (`none`).
Only rows with `naming` or `arg-order` reasons, or explicitly
justified exclusions, are detailed.

## Applied renames / reorders

| # | Old signature | New signature | Reason |
|---|---|---|---|
| 1 | `pulseg_free_trajectory(pulseg_trajectory *traj)` (`pulseg_trajectory.h:160`) | `pulseg_trajectory_free(pulseg_trajectory *traj)` | naming — verb-first `free_X`, inconsistent with conforming siblings `pulseg_tr_gradient_waveforms_free`, `pulseg_tr_waveforms_free` |
| 2 | `pulseg_free_sequence_description(pulseg_sequence_description *desc)` (`pulseg_convert.h:88`) | `pulseg_sequence_description_free(pulseg_sequence_description *desc)` | naming — same pattern as #1 |
| 3 | `pulseg_get_freq_mod(coll, subseq_idx, scan_table_pos, out_hw_waveform, out_num_samples, out_phase_rad)` (`pulseg_freqmod.h:166`) | `pulseg_get_freq_mod(coll, out_hw_waveform, out_num_samples, out_phase_rad, subseq_idx, scan_table_pos)` | arg-order — outputs must precede inputs; sibling `pulseg_freq_mod_collection_get` already uses the correct order |
| 4 | `pulseg_get_prep_segment_table(coll, subseq_idx, out_ids)` (`pulseg_collection.h:294`) | `pulseg_get_prep_segment_table(coll, out_ids, subseq_idx)` | arg-order |
| 5 | `pulseg_get_main_segment_table(coll, subseq_idx, out_ids)` (`pulseg_collection.h:302`) | `pulseg_get_main_segment_table(coll, out_ids, subseq_idx)` | arg-order |
| 6 | `pulseg_get_cooldown_segment_table(coll, subseq_idx, out_ids)` (`pulseg_collection.h:310`) | `pulseg_get_cooldown_segment_table(coll, out_ids, subseq_idx)` | arg-order |
| 7 | `pulseg_get_canonical_segment_sequence(coll, subseq_idx, out_ids)` (`pulseg_collection.h:328`) | `pulseg_get_canonical_segment_sequence(coll, out_ids, subseq_idx)` | arg-order (`out_ids` stays nullable for count-only query) |
| 8 | `pulseg_get_definitions(coll, subseq_idx, out, num_entries)` (`pulseg_collection.h:503`) | `pulseg_get_definitions(coll, out, num_entries, subseq_idx)` | arg-order |
| 9 | `pulseg_get_segment_block_def_indices(coll, seg_idx, out_ids)` (`pulseg_collection.h:639`) | `pulseg_get_segment_block_def_indices(coll, out_ids, seg_idx)` | arg-order |
| 10 | `pulseg_protocol_get_float(p, param_id, out)` (`pulseg_protocol.h:295`) | `pulseg_protocol_get_float(p, out, param_id)` | arg-order |
| 11 | `pulseg_protocol_get_int(p, param_id, out)` (`pulseg_protocol.h:297`) | `pulseg_protocol_get_int(p, out, param_id)` | arg-order |
| 12 | `pulseg_protocol_get_bool(p, param_id, out)` (`pulseg_protocol.h:299`) | `pulseg_protocol_get_bool(p, out, param_id)` | arg-order |
| 13 | `pulseg_protocol_get_stringlist(p, param_id, idx_out)` (`pulseg_protocol.h:309`) | `pulseg_protocol_get_stringlist(p, idx_out, param_id)` | arg-order |
| 14 | `pulseg_bridge_validate(b, proto, duration, info, infosz)` (`pulseg_bridge.h:99`) | `pulseg_bridge_validate(b, duration, info, infosz, proto)` | arg-order (zero call sites outside decl/def — safe, no callers to patch) |
| 15 | `pulseg__build_scan_table(desc, num_averages, diag)` (`pulseg_internal.h:778`) | `pulseg__build_scan_table(desc, diag, num_averages)` | arg-order — inconsistent with sibling `pulseg__get_scan_table_segments(desc, diag, opts)` |
| 16 | `pulseg__resolve_segment(out_desc, out_local_seg, coll, seg_idx)` (`pulseg_internal.h:832`) | `pulseg__resolve_segment(coll, out_desc, out_local_seg, seg_idx)` | arg-order — self (`coll`) was not first |
| 17 | `pulseg__resolve_block(out_desc, out_seg, out_local_blk, coll, seg_idx, blk_idx)` (`pulseg_internal.h:838`) | `pulseg__resolve_block(coll, out_desc, out_seg, out_local_blk, seg_idx, blk_idx)` | arg-order — self (`coll`) was not first |

Rows 3 and 15–17 are internal-consistency findings beyond the two
seed violations named in the plan; row 3 is the plan's named seed.

## Cross-checked `pulseg_free_*` / `pulseg_get_*` naming (per plan step 1.2)

Conforming `*_free` (no change): `pulseg_tr_gradient_waveforms_free`,
`pulseg_tr_waveforms_free`, `pulseg_recon_cache_free`,
`pulseg_collection_free`, `pulseg_freq_mod_collection_free`,
`pulseg_mech_resonances_spectra_free`, `pulseg_pns_result_free`,
`pulseg_pulseq_file_free`, `pulseg_pulseq_file_set_free`,
`pulseg_pulseq_block_free`, `pulseg_sequence_descriptor_free`,
`pulseg_segment_table_result_free`, `pulseg__uniform_grad_waveforms_free`.
Non-conforming: `pulseg_free_trajectory`, `pulseg_free_sequence_description`
(rows 1–2 above).

`pulseg_get_freq_mod` vs `pulseg_freq_mod_collection_get`: intentional
duplication (coll-level convenience wrapper + fmc-level method) —
**both kept**, arg order now mutually consistent (self, outputs,
inputs) after row 3.

## Justified exclusions (would be a violation under a stricter reading, but out of scope / behavior-risk)

| Function(s) | Why flagged | Why excluded |
|---|---|---|
| `pulseg_cursor_next/rewind/mark/reset/get_info(coll, ...)` (`pulseg_collection.h`) | Drops the `collection` class segment (`pulseg_cursor_X` not `pulseg_collection_cursor_X`) | Not a `free_*`/`get_*` verb-prefix or arg-order defect — it's a deliberate compound-noun sub-object name (`cursor`), out of the plan's stated cross-check scope (step 1.2 targets `free_*`/`get_*` naming and arg order only). Renaming would touch the widest blast radius in the file (scan-loop hot path, `@rsp` code) for a purely cosmetic gain; not attempted here. |
| `pulseg_check_safety(coll, diag, opts, num_forbidden_bands, forbidden_bands, pns_model, pns_threshold_percent)` and `pulseg_calc_pns`/`pulseg_calc_mech_resonances` (`pulseg_safety.h`) | Named `pulseg_check_safety` / `pulseg_calc_pns`, not `pulseg_safety_check`/`pulseg_pns_calc` | Arg order already conforms (self, outputs, inputs); naming already matches the established `pulseg_<verb>_<module>` module-fn convention used consistently across this header (`pulseg_calc_mech_resonances`, `pulseg_calc_pns`) — no defect, `none`. |
| `pulseg_bridge_open/open_with_opts/close/alive/list_protocol/generate` (`pulseg_bridge.h`) | n/a | Audited, conforming (ctor/dtor pattern, `generate`'s `output_path` is a side-effect input per the convention's writer-destination carve-out). `none`. |

## Behavior-neutral guard (step 1.4)

No signature change in this pass alters struct layout or cache-facing
behavior. All 17 changes are argument reorderings or pure renames of
C function symbols; none touch `pulseg_*` struct definitions, the
binary cache format, or `.pge`/`.pseg` on-disk layout.

## Known safety-relevant call site

`pulseg_get_freq_mod` (row 3) is called from `@rsp` scan-time code in
the private repo: `pulserver-interpreter/src_psd/pulserver_implementation/PulserverImplementationScan.e:515`
(and its EPIC-generated mirror `pulserver.tgt.cpp:18028`, not hand-edited).
Freqmod + fovshift SIM suites must be re-run after this change and
compared against the pre-change pass counts.

## Private-repo (`pulserver-interpreter`) call sites patched

Discovered by grepping `src_psd`/`src_gelib`/`src_common`/`src_rtp`/`src_vre`
for every symbol in this map. All are `@host`-section EPIC source; the
matching `.host.cpp`/`.tgt.cpp` mirrors are EPIC-generated (`Do not edit
this file`) and were left untouched — they regenerate from the `.e`
sources at the next `build_psd.sh` run.

| Function | `.e` file | Call sites |
|---|---|---|
| `pulseg_get_freq_mod` | `PulserverImplementationScan.e:515` | 1 |
| `pulseg_get_canonical_segment_sequence` | `PulserverImplementationPredownload.e:897,903` | 2 |
| `pulseg_protocol_get_float/int/bool/stringlist` | `PulserverImplementationCVInit.e` | 39 |
| `pulseg_bridge_validate` | `PulserverImplementationCVEval.e:111` | 1 |

No `.pge`/`.pseg` fixture-affecting code paths were touched (all edits are
argument reorderings with identical semantics).
