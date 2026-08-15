# PulSeg conformance

**Reference:** PulSeg IR specification v2.1-alpha (spec.md + spec_candidate.md,
J-F Nielsen, M. Cencini) · **Audited implementation:** `csrc` (`pulseg_*`)

Pulserver's in-memory GE representation conforms to the PulSeg IR
**semantically, as a derived class**: every quantity the spec's four
structures carry is present, stored as a lossless id-indexed compression of
the canonical content plus interpreter-specific attributes, and the
conversion Pulseq → GE is fused into one pass — no intermediate IR file is
materialized. Where the layout diverges structurally, this document states
the equivalence; where pulserver carries more, each extra is classified as
either a **user-param candidate** (expressible through the proposed
`user_int[]` / `user_float[]` attributes, below) or **derived data** an
implementation recomputes and need not exchange.

## 1. Mapping

### BaseBlock (§3.0–3.1) ↔ `pulseg_base_block` + definition libraries

| Spec | Pulserver | Notes |
|---|---|---|
| `id` | `pulseg_base_block.id` | content-deduplicated on `(gx_id, gy_id, gz_id, rf_id, adc_id, duration)` |
| `block.blockDuration` | `duration_us` | µs integer (see §5) |
| `block.rf` | `rf_id` → `rf_definitions[]` + `shapes[]` | normalized mag/phase/time shapes; per-instance amplitude lives in the instance table, per the spec's normalization rules |
| `block.gx/gy/gz` | `g*_id` → `grad_definitions[]` | identity is **timing-based** exactly as §3.1 defines it: trapezoid `(delay, rise, flat, fall)`, extended `(delay, time_shape_id)`; shot variants ride `grad_table_element.shape_id` |
| `block.adc` | `adc_id` → `adc_definitions[]` | `(num_samples, dwell, delay)`, not normalized |
| reserved ids 0/1 (delays) | pure-delay base blocks + `is_dynamic_delay` | the constant/variable delay split of the reserved ids maps onto the per-position static/adjustable classification |

The definition libraries are not a deviation: they are the compression the
spec's *shape*-level identity implies. A `BaseBlock.block` is reconstructed
by hydrating the referenced definitions, and two base blocks sharing a
waveform share its storage.

### VirtualSegment (§3.2) ↔ `pulseg_virtual_segment`

| Spec | Pulserver |
|---|---|
| `id` | segment index (TRID-keyed, the pge2 convention) |
| `base_block_ids` | `unique_block_indices[]` → base block ids |
| `name` | — (not stored; would map directly) |

### SegmentInstance (§3.3) ↔ instance tables + execution runs

Pulserver column-normalizes the instance arrays into per-event tables
referenced from per-block rows — one `pulseg_block_table_element` per
block, whose `rf_id`/`g*_id`/`adc_id` index `pulseg_rf_table_element`
(`amplitude`, `phase_offset`, `freq_offset`), `pulseg_grad_table_element`
(`amplitude`, instance `shape_id`) and `pulseg_adc_table_element`
(`phase_offset`, `freq_offset`). This is the spec's content at the spec's
granularity — §3.3.1 counts one gradient event per block, which is exactly
one block-table row — factored so that repeated values are stored once.

| Spec | Pulserver |
|---|---|
| `rf_amplitude` / `rf_phase_offset` / `rf_frequency_offset` | `rf_table_element` fields |
| `gradient_amplitude[N][3]` | per-axis `grad_table_element.amplitude` (signed) |
| `gradient_shot_index` | instance `shape_id` on the timing-identified definition |
| `adc_phase_offset` / `adc_frequency_offset` | `adc_table_element` fields |
| `block_duration[]` | `block_table_element.duration_us` |
| `rotation_matrix` | `rotation_id` → deduplicated rotation library (3×3, parsed once from the file's quaternions) |
| `physio_trigger` | segment-level `trigger_id` (INPUT type) |
| `label` | the label table (richer: MRD counters, see §3) |

The rotation library is a lossless compression of §3.3's inline
`float[3][3]` per gradient event: `G_phys = R · diag(s) · G_base` is
composed identically, with `R` fetched by id. The spec's own intent note
applies: this `R` is the *trajectory* rotation (multishot spokes/
interleaves), never the console's FOV reorientation.

**Execution stream**: `pulseg_exec_run` stores the instance list as
contiguous runs `(emit_start, block_start, length, tr_id, avg_id)` — a
run-length encoding of `execution_stream[]` that reproduces it exactly.

### Top level (§3.4)

`pulseg_collection` holds one descriptor per subsequence (one Pulseq file
of a `NextSequence` chain); each descriptor is one complete PulSeg
representation. `pulseg_version` maps to the cache header version triplet;
`source_file` to the staged `.seq` path.

## 2. Proposed spec amendment: per-class user parameters

The one addition that lets pulserver's remaining attributes ride the spec
without new classes: give **every** spec class two optional attributes,

```
user_int   : int[]     (optional, default empty)
user_float : float[]   (optional, default empty)
```

with meanings assigned by an implementation-published registry of named
positions (pulserver's is the `PULSEG_PARAM_*` constant set; consumers read
through named accessors — `pulseg_cursor` getters — never positional
literals). Pulserver would register:

| Class | user_int positions | user_float positions |
|---|---|---|
| SegmentInstance (per block row) | `ONCE`, `NOROT`, `NOPOS`, `PMC`, `NAV` flags; `digitalout_id`; `rf_shim_id`; `TRID`; MRD label columns | — |
| VirtualSegment | `is_nav`; input `trigger_id` | — |
| BaseBlock (per gradient definition) | — | representative `energy`, `slew_rate`, aggregate amplitude/slew bounds |

Everything else pulserver holds is **derived data** — recomputable from the
core content, so not proposed for exchange: the per-position
`pulseg_block_initial_state` (the segment's entry state for pulse
generation and the minseq gradient-heating check: representative base
block, per-axis gradient definition/shape/amplitude, RF amplitude, the
`rf_grad_constant` PMC eligibility), the `has_rotation`/`has_adc`/
`is_dynamic_delay` per-position summaries, the segment timing anchors, and
the TR descriptor (§4).

## 3. Where the flattened loop falls short

`stream2loop`'s 24-column row is documented in the reference implementation
as a *convenience/debug view*, and it should stay one:

- **Footprint.** Every row carries all event columns whether the block has
  the event, plus the 3×3 rotation inline (nine floats where an id into a
  deduplicated library suffices) — on a stack-of-spirals or radial scan the
  same matrices repeat per partition, and on a million-block 3D scan the
  materialized loop is an order of magnitude beyond the id-indexed form.
- **Placement.** The loop carries per-axis gradient *energy* in the event
  stream. Energy is not an execution parameter: it is consumed once, to
  determine each segment's initial state for pulse generation so the
  minimum-TR (gradient heating) check is computed correctly — alongside
  the initial per-axis gradient shape ids from which the canonical TRs for
  mechanical-resonance and PNS checks are formed. These belong with the
  segment definition (as derived/user data), not on every instance row.

## 4. TR, TRID and the structural-TR declaration

- A PulSeg **virtual segment** is pulserver's SEGMENT: a reusable unit with
  no periodicity assumption. TR is a *safety/efficiency* concept layered on
  top and stays out of the IR core.
- **TRID** marks segment-instance boundaries (§4.2) and, in pulserver,
  additionally allows the optional refinement of isolating TRs inside
  hyperTRs for the RF safety checks (coil protection, SAR). Absent TRID,
  the conservative check runs and the hyperTR-level guard plus the
  hardware monitor still stand — TRID is never load-bearing for safety.
- **`TRSize` definition (proposed convention):** the design side MAY write
  the structural TR's block count into `[DEFINITIONS]` (pulserver's writer
  does, automatically, whenever detection succeeds; one value per file of a
  `NextSequence` chain). A consumer may use it — a reconstruction derives
  its sequence description only when it is present; an interpreter may
  read base definitions off the first TR and verify the pattern repeats —
  or ignore it outright, with full detection as the fallback and the check.
- With repeats and averages materialized by the designer, prep/cooldown
  bookkeeping is not information an interpreter needs from the stream and
  is being removed from pulserver's TR descriptor.

## 5. Documented deviations (version-bump candidates)

1. **Granularity**: pulserver stores times as integer microseconds (the
   32-bit scanner target), where the spec speaks seconds. Lossless for any
   raster-aligned sequence; the IR is a structural pattern rather than an
   interchange format, so native units are declared rather than converted.
2. **Endianness marker**: pulserver's serialized form (the `.pge` cache)
   opens with an `0x01020304` marker and byte-swaps on mismatch. The spec
   defines no serialization; if one is added, the marker + version triplet
   header is proposed as-is.
3. **`user_int`/`user_float`** per class, as §2.
4. **`TRSize`** definition convention, as §4.
