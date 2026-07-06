# Pulseg Cache Binary Wire-Format Specification

**Version**: 2.0.0  
**Last Updated**: 2026-06-25

## Overview

The pulseg cache file (`.pge` companion to `.seq`) is a binary serialization of a `pulseg_collection` object. As of cache format **v2.0.0** the payload is split into **per-consumer sections**, so each PSD phase / consumer deserializes only the data it needs. All integer and floating-point fields are stored as 4-byte values; all endianness is determined by the file header marker.

The whole `.pge` is a pure function of the loaded collection (shift- and rotation-independent). It is produced in one shot at load time by `pulseg__write_cache` (see `pulseg_cache.c`), which emits the four base sections (COMMON, ROTATIONS, SHAPES, SCANLOOP) and then appends TRAJECTORY, SEQDESC and FREQMOD.

---

## File Header

The cache file begins with a fixed header (28 bytes):

| Offset | Field | Type | Count | Size | Description |
|--------|-------|------|-------|------|-------------|
| 0x00 | Endian marker | int32 | 1 | 4 | `0x01020304`; if byte-swapped, indicates big-endian file |
| 0x04 | Version major | int32 | 1 | 4 | `PULSEG_CACHE_VERSION_MAJOR = 2` |
| 0x08 | Version minor | int32 | 1 | 4 | `PULSEG_CACHE_VERSION_MINOR = 0` |
| 0x0C | Version revision | int32 | 1 | 4 | `PULSEG_CACHE_VERSION_REVISION = 0` |
| 0x10 | Vendor | int32 | 1 | 4 | `PULSEG_VENDOR` constant; identifies hardware/vendor context |
| 0x14 | Source seq file size | int32 | 1 | 4 | Byte count of original `.seq` file; used for cache validity check |
| 0x18 | Number of sections | int32 | 1 | 4 | Count of section entries in the table-of-contents (up to 7) |

After the header, the file contains:
- **Section table** (immediate): `num_sections × 12` bytes (3 int32 per entry)
- **Section data**: Variable-length payloads starting at offsets specified in the table

### Version Policy

The reader hard-fails on any mismatch of **major, minor AND revision** — there is no
intra-version fallback. Stale caches are regenerated, not partially parsed.

- **v1.x**: legacy monolithic layout (CHECK/GENINSTRUCTIONS/SCANLOOP byte-identical payloads).
  No longer produced or read.
- **v2.0.0** (current): per-consumer section split; adds the `version_revision` header word.

---

## Section Table of Contents (TOC)

Immediately after the 28-byte file header, a TOC of `num_sections` entries follows. Each entry is 12 bytes:

| Offset in entry | Field | Type | Count | Size | Description |
|---|---|---|---|---|---|
| 0x00 | section_id | int32 | 1 | 4 | See section IDs table below |
| 0x04 | offset | int32 | 1 | 4 | Absolute byte offset from start of file to section data |
| 0x08 | size | int32 | 1 | 4 | Byte count of section data (not including padding) |

### Section IDs

| ID | Name | Purpose | Reader / Consumer | Notes |
|---|---|---|---|---|
| 1 | COMMON | Collection header + per-descriptor structure/scaling/defs (RF/grad/ADC defs, segment defs+anchors, segment table, labels, generic [DEFINITIONS]) | pulseg_load_geninstructions_cache() (+SHAPES); pulseg_load_scanloop_cache() (+ROTATIONS+SCANLOOP); trajectory_cache_reader.cpp | Must be read first; augment sections attach to the descriptors it allocates. Excludes rotations, raw shapes, scan table |
| 2 | ROTATIONS | 3×3 rotation-matrix library (per descriptor) | scan (load_scanloop); trajectory_cache_reader.cpp | Augment section |
| 3 | SHAPES | RF mag/phase/time + gradient compressed shape sample arrays (per descriptor) | pulsegen (load_geninstructions) | Augment section |
| 4 | SCANLOOP | scan_table (4×len) + variable_grad_flags (per descriptor) | scan (load_scanloop) | Augment section |
| 5 | FREQMOD | Frequency-modulation **base only** (shift-independent) | scan; **NOT parsed by mrdserver** | Applied PSD-side; data at recon already centered |
| 6 | TRAJECTORY | Trajectory library (kshots, encoding spaces, table) | trajectory_cache_reader.cpp | Non-Cartesian trajectory data |
| 7 | SEQDESC | Event lists, RF shapes, shims | trajectory_cache_reader.cpp (optional) | Per-subsequence details; graceful skip if absent |

> **Note (Stage 3, planned):** TRAJECTORY(6) + SEQDESC(7) are slated to merge into a single
> RECON section (id 6) carrying trajectory + seqdesc + generic [DEFINITIONS] + a rotations copy.
> Not yet implemented.

---

## Section Data: Field-by-Field Layout

The **COMMON** section carries a **collection-level header** followed by **per-subsequence
descriptors** (the field-by-field layout below, minus the rotations, raw shapes and scan-table
blocks). The **augment sections** (ROTATIONS, SHAPES, SCANLOOP) carry no collection scalars: they
write a single `num_subsequences` token (validated against COMMON on read) followed by one
per-descriptor region each. Fields are written via `fwrite()` in the following order:

### Collection-Level Header (COMMON section only)

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| num_subsequences | int32 | 1 | 4 | Number of loaded subsequences |
| num_repetitions | int32 | 1 | 4 | Repetition count |
| total_unique_segments | int32 | 1 | 4 | Total distinct segments across all subsequences |
| total_unique_adcs | int32 | 1 | 4 | Total distinct ADC definitions |
| total_blocks | int32 | 1 | 4 | Total block count |
| total_duration_us | int32 | 1 | 4 | Entire sequence duration (microseconds) |

### Per-Subsequence Header (COMMON section)

For each of the `num_subsequences` entries:

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| sequence_index | int32 | 1 | 4 | Index into the parent collection's sequence array |
| adc_id_offset | int32 | 1 | 4 | Offset applied to ADC IDs for this subsequence |
| segment_id_offset | int32 | 1 | 4 | Offset applied to segment IDs for this subsequence |
| block_index_offset | int32 | 1 | 4 | Offset applied to block indices for this subsequence |

### Per-Subsequence Descriptor

A full `pulseg_sequence_descriptor` follows immediately. The layout is:

#### Scalar Fields (44 bytes)

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| num_prep_blocks | int32 | 1 | 4 | Preparation blocks (e.g., pre-pulses) |
| num_cooldown_blocks | int32 | 1 | 4 | Cooldown blocks after main sequence |
| rf_raster_us | int32 | 1 | 4 | RF sampling raster (microseconds) |
| grad_raster_us | int32 | 1 | 4 | Gradient sampling raster (microseconds) |
| adc_raster_us | int32 | 1 | 4 | ADC dwell/raster (microseconds) |
| block_raster_us | int32 | 1 | 4 | Block duration raster (microseconds) |
| ignore_fov_shift | int32 | 1 | 4 | Boolean flag; if set, FOV shift is ignored |
| enable_pmc | int32 | 1 | 4 | Boolean flag; parallel motion correction |
| ignore_averages | int32 | 1 | 4 | Boolean flag; ignore averaging |
| num_passes | int32 | 1 | 4 | Number of separate passes / phases |
| vendor | int32 | 1 | 4 | Vendor identifier (e.g., PULSEG_VENDOR_GEHC) |
| fov[0..2] | float | 3 | 12 | Field of view (x, y, z) in mm |
| matrix[0..2] | int32 | 3 | 12 | Matrix dimensions (x, y, z) |
| nav_fov[0..2] | float | 3 | 12 | Navigator FOV (x, y, z) in mm |
| nav_matrix[0..2] | int32 | 3 | 12 | Navigator matrix dimensions (x, y, z) |

#### Block Definitions

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_unique_blocks | int32 | 1 | 4 | Count of unique block types |

For each block definition (7 int32 per entry):

| Field | Type | Count | Bytes |
|-------|------|-------|-------|
| id | int32 | 1 | 4 |
| duration_us | int32 | 1 | 4 |
| rf_id | int32 | 1 | 4 |
| gx_id | int32 | 1 | 4 |
| gy_id | int32 | 1 | 4 |
| gz_id | int32 | 1 | 4 |
| adc_id | int32 | 1 | 4 |

#### Block Table

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_blocks | int32 | 1 | 4 | Total block instances |

For each block table entry (16 int32 per entry):

| Field | Type | Count | Bytes |
|-------|------|-------|-------|
| id | int32 | 1 | 4 |
| duration_us | int32 | 1 | 4 |
| rf_id | int32 | 1 | 4 |
| gx_id | int32 | 1 | 4 |
| gy_id | int32 | 1 | 4 |
| gz_id | int32 | 1 | 4 |
| adc_id | int32 | 1 | 4 |
| digitalout_id | int32 | 1 | 4 |
| rotation_id | int32 | 1 | 4 |
| once_flag | int32 | 1 | 4 |
| norot_flag | int32 | 1 | 4 |
| nopos_flag | int32 | 1 | 4 |
| pmc_flag | int32 | 1 | 4 |
| nav_flag | int32 | 1 | 4 |
| freq_mod_id | int32 | 1 | 4 |
| rf_shim_id | int32 | 1 | 4 |

#### RF Definitions

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_unique_rfs | int32 | 1 | 4 | Unique RF pulse types |

For each RF definition:

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| id | int32 | 1 | 4 | |
| mag_shape_id | int32 | 1 | 4 | Magnitude waveform shape ID |
| phase_shape_id | int32 | 1 | 4 | Phase waveform shape ID |
| time_shape_id | int32 | 1 | 4 | Time/sampling shape ID |
| delay | int32 | 1 | 4 | RF pulse delay (microseconds) |
| num_channels | int32 | 1 | 4 | Transmit channels (usually 1, may be >1 for pTx) |
| flip_angle_deg | float | 1 | 4 | Nominal flip angle (degrees) |
| act_amplitude_hz | float | 1 | 4 | Actual peak amplitude (Hz) |
| area | float | 1 | 4 | Integral of B1 magnitude (arbitrary units) |
| abs_width | float | 1 | 4 | Fractional width with nonzero B1 |
| eff_width | float | 1 | 4 | Effective rectangular pulse width |
| duty_cycle | float | 1 | 4 | RF duty cycle within TR |
| max_pulse_width | float | 1 | 4 | Longest contiguous B1 segment (seconds) |
| duration_us | float | 1 | 4 | Total RF duration (microseconds) |
| isodelay_us | int32 | 1 | 4 | Isodelay from center to echo (microseconds) |
| bandwidth_hz | float | 1 | 4 | Estimated bandwidth (Hz) |
| base_amplitude_hz | float | 1 | 4 | Base nominal peak amplitude (Hz) |
| num_samples | int32 | 1 | 4 | Waveform sample count |
| num_bands | int32 | 1 | 4 | Number of frequency bands (multiband; ≥1) |
| band_freq_offsets_hz[0..7] | float | 8 | 32 | Per-band center frequency offsets (Hz) |
| band_bandwidth_hz | float | 1 | 4 | Per-band bandwidth (Hz) |
| total_b1sq_power | float | 1 | 4 | Integral of \|B1(t)\|² (arbitrary units) |
| vendor | int32 | 1 | 4 | Vendor ID for interpretation of above fields |

#### RF Table

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| rf_table_size | int32 | 1 | 4 | Number of RF table entries |

For each RF table entry (5 int32 per entry):

| Field | Type | Count | Bytes |
|-------|------|-------|-------|
| id | int32 | 1 | 4 |
| amplitude | float | 1 | 4 |
| freq_offset | float | 1 | 4 |
| phase_offset | float | 1 | 4 |
| rf_use | int32 | 1 | 4 |

#### Gradient Definitions

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_unique_grads | int32 | 1 | 4 | Unique gradient waveform types |

For each gradient definition:

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| id | int32 | 1 | 4 | |
| type | int32 | 1 | 4 | Waveform type (e.g., trapezoid, arbitrary) |
| rise_time_or_unused | float | 1 | 4 | Ramp time (trapezoid) |
| flat_time_or_unused | float | 1 | 4 | Flat-top time |
| fall_time_or_num_uncompressed | float | 1 | 4 | Ramp-down time or uncompressed sample count |
| unused_or_time_shape_id | int32 | 1 | 4 | Reserved or time-shape ID |
| delay | float | 1 | 4 | Gradient pulse delay (microseconds) |
| num_shots | int32 | 1 | 4 | Number of amplitude shots (up to MAX_GRAD_SHOTS=16) |
| shot_shape_ids[0..15] | int32 | 16 | 64 | Shape IDs for each shot |
| max_amplitude[0..15] | float | 16 | 64 | Max amplitude per shot (Hz/m) |
| min_amplitude[0..15] | float | 16 | 64 | Min amplitude per shot (Hz/m) |
| slew_rate[0..15] | float | 16 | 64 | Slew rate per shot (Hz/m/s) |
| energy[0..15] | float | 16 | 64 | Energy integral per shot |
| first_value[0..15] | float | 16 | 64 | Initial amplitude value per shot |
| last_value[0..15] | float | 16 | 64 | Final amplitude value per shot |

#### Gradient Table

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| grad_table_size | int32 | 1 | 4 | Number of gradient instances |

For each entry (3 int32 per entry):

| Field | Type | Count | Bytes |
|-------|------|-------|-------|
| id | int32 | 1 | 4 |
| shot_index | int32 | 1 | 4 |
| amplitude | float | 1 | 4 |

#### ADC Definitions

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_unique_adcs | int32 | 1 | 4 | Unique ADC readout types |

For each ADC definition (4 int32 per entry):

| Field | Type | Count | Bytes |
|-------|------|-------|-------|
| id | int32 | 1 | 4 |
| num_samples | int32 | 1 | 4 |
| dwell_time | float | 1 | 4 |
| delay | float | 1 | 4 |

#### ADC Table

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| adc_table_size | int32 | 1 | 4 | Number of ADC instances |

For each entry (3 int32 per entry):

| Field | Type | Count | Bytes |
|-------|------|-------|-------|
| id | int32 | 1 | 4 |
| freq_offset | float | 1 | 4 |
| phase_offset | float | 1 | 4 |

#### Frequency Modulation Definitions (Legacy)

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| num_freq_mod_defs | int32 | 1 | 4 | **Always 0** in current caches; deprecated |

#### RF Shim Definitions

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_rf_shims | int32 | 1 | 4 | Shim coil definitions for pTx |

For each shim definition:

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| id | int32 | 1 | 4 | Shim ID |
| num_channels | int32 | 1 | 4 | Number of channels (N_ch) |
| magnitudes[0..N_ch-1] | float | N_ch | 4×N_ch | Shim coefficient magnitudes |
| phases[0..N_ch-1] | float | N_ch | 4×N_ch | Shim coefficient phases (radians) |

#### Rotations (Spatial) — ROTATIONS section (id 2)

> Emitted in the standalone **ROTATIONS** augment section, **not** in COMMON. The section is a
> `num_subsequences` token followed by one of these per-descriptor regions.

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_rotations | int32 | 1 | 4 | Number of 3×3 rotation matrices |

For each rotation matrix:

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| matrix[0..8] | float | 9 | 36 | Row-major 3×3 rotation (9 floats) |

#### Trigger Events

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_triggers | int32 | 1 | 4 | Number of trigger definitions |

For each trigger (5 int32 per entry):

| Field | Type | Count | Bytes |
|-------|------|-------|-------|
| type | int32 | 1 | 4 |
| duration | int32 | 1 | 4 |
| delay | int32 | 1 | 4 |
| trigger_type | int32 | 1 | 4 |
| trigger_channel | int32 | 1 | 4 |

#### Shapes (Waveforms) — SHAPES section (id 3)

> Emitted in the standalone **SHAPES** augment section, **not** in COMMON. The section is a
> `num_subsequences` token followed by one of these per-descriptor regions.

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_shapes | int32 | 1 | 4 | Number of arbitrary waveforms |

For each shape:

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| num_uncompressed_samples | int32 | 1 | 4 | Original (pre-compression) sample count |
| num_samples | int32 | 1 | 4 | Stored sample count (N_s) |
| samples[0..N_s-1] | float | N_s | 4×N_s | Waveform sample values (if N_s > 0) |

#### TR Descriptor (10 fields)

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| num_prep_blocks | int32 | 1 | 4 | Prep block count |
| num_cooldown_blocks | int32 | 1 | 4 | Cooldown block count |
| tr_size | int32 | 1 | 4 | Total blocks in one TR |
| num_trs | int32 | 1 | 4 | Number of TR repetitions in sequence |
| num_prep_trs | int32 | 1 | 4 | Prep TR count |
| degenerate_prep | int32 | 1 | 4 | Boolean: prep TRs are degenerate |
| num_cooldown_trs | int32 | 1 | 4 | Cooldown TR count |
| degenerate_cooldown | int32 | 1 | 4 | Boolean: cooldown TRs are degenerate |
| imaging_tr_start | int32 | 1 | 4 | First imaging TR index |
| tr_duration_us | float | 1 | 4 | TR duration (microseconds) |

#### Segment Definitions

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_unique_segments | int32 | 1 | 4 | Unique segment types |

For each segment definition:

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| start_block | int32 | 1 | 4 | First block index |
| num_blocks | int32 | 1 | 4 | Block count (N_b) |
| max_energy_start_block | int32 | 1 | 4 | Block with max gradient energy |
| unique_block_indices[0..N_b-1] | int32 | N_b | 4×N_b | Block indices in segment |
| has_digitalout[0..N_b-1] | int32 | N_b | 4×N_b | Digital output flags per block |
| has_rotation[0..N_b-1] | int32 | N_b | 4×N_b | Rotation flags per block |
| norot_flag[0..N_b-1] | int32 | N_b | 4×N_b | No-rotation flags per block |
| nopos_flag[0..N_b-1] | int32 | N_b | 4×N_b | No-position flags per block |
| trigger_id | int32 | 1 | 4 | Associated trigger ID |
| is_nav | int32 | 1 | 4 | Boolean: navigator segment |

#### Segment Table

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| num_unique_segments | int32 | 1 | 4 | Redundant; segment count |
| num_prep_segments | int32 | 1 | 4 | Prep segment count (N_p) |
| prep_segment_table[0..N_p-1] | int32 | N_p | 4×N_p | Prep segment indices (if N_p > 0) |
| num_main_segments | int32 | 1 | 4 | Main segment count (N_m) |
| main_segment_table[0..N_m-1] | int32 | N_m | 4×N_m | Main segment indices (if N_m > 0) |
| num_cooldown_segments | int32 | 1 | 4 | Cooldown segment count (N_c) |
| cooldown_segment_table[0..N_c-1] | int32 | N_c | 4×N_c | Cooldown segment indices (if N_c > 0) |

#### Label Table & Limits

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| label_num_columns | int32 | 1 | 4 | Columns in label matrix |
| label_num_entries | int32 | 1 | 4 | Rows in label matrix |
| label_table[0..entries×cols-1] | int32 | entries×cols | 4×entries×cols | Label indices (if entries > 0) |
| label_limits | struct | 1 | 80 | Global label min/max (10 × 2 int32) |

#### Generic Definitions

| Prefix | Type | Count | Bytes | Description |
|--------|------|-------|-------|---|
| num_definitions | int32 | 1 | 4 | Count of key–value pairs |

For each definition:

| Field | Type | Description |
|-------|------|---|
| name_length | int32 | String length (bytes) |
| name | char | Name (no null terminator in file) |
| value_size | int32 | Number of values |
| values[0..value_size-1] | length-prefixed string | Value strings (each: int32 length + bytes) |

#### Scan Table — SCANLOOP section (id 4)

> Emitted in the standalone **SCANLOOP** augment section, **not** in COMMON. The section is a
> `num_subsequences` token followed by one of these per-descriptor regions (scan_table plus the
> `variable_grad_flags`).

| Field | Type | Count | Bytes | Description |
|-------|------|-------|-------|---|
| scan_table_len | int32 | 1 | 4 | Number of scan table entries |
| scan_table_block_idx[...] | int32 | scan_table_len | 4×len | Block indices (if len > 0) |
| scan_table_tr_id[...] | int32 | scan_table_len | 4×len | TR IDs (if len > 0) |
| scan_table_seg_id[...] | int32 | scan_table_len | 4×len | Segment IDs (if len > 0) |
| scan_table_avg_id[...] | int32 | scan_table_len | 4×len | Average IDs (if len > 0) |

---

## String Encoding

All strings in the cache are **length-prefixed**, not null-terminated:

```c
int32_t length;      // byte count of string content
char    string[len]; // raw bytes (NO null terminator in file)
```

When reading, append a null terminator in memory for C-string compatibility.

---

## Endianness & Byte-Swapping

The endian marker `0x01020304` at file start indicates the file byte order:
- **Native endian match**: No swap needed
- **Endian mismatch**: Byte-swap all 4-byte fields after reading

All read/write operations use `fread()/fwrite()` with 4-byte alignment; no padding or alignment bytes are present.

---

## Section 5 (FREQMOD) — Special Note

**mrdserver's trajectory_cache_reader.cpp intentionally does NOT parse Section 5** (FREQMOD).  
Off-isocenter frequency shifts are applied **at the PSD level** during sequence execution. By the time data arrives at the recon pipeline, all spatial shifts have already been applied; the recon receives centered k-space data. Section 5 stores only the **shift-independent base**; the shift-dependent plan waveforms are recomputed PSD-side at scan time and are not cached.

---

## Version Compatibility

Readers should:

1. Always check the endian marker first
2. Validate version_major, version_minor AND version_revision all match expectations
   (the writer's reader hard-fails on any mismatch; there is no intra-version fallback)
3. Skip unknown sections gracefully
4. Read COMMON (id 1) before any augment section (ROTATIONS/SHAPES/SCANLOOP), which attach to the
   descriptors COMMON allocates
5. Treat sequence descriptions (Section 7) as optional; degrade gracefully if absent


---

# Truth file formats (TruthBuilder ground truth)

These `.bin` files sit alongside each `.seq` fixture under `expected/` and
are produced by `+testutils/TruthBuilder.m` from the official Pulseq
toolbox. They are the **independent** ground truth that pulseg output
is compared against; they are NOT pulseg cache sections.

## `<base>_trajectory.bin` (Phase A MVP)

Per-ADC k-space samples computed via `mr.Sequence.calculateKspacePP()`.

```
int32  num_adcs
int32  is_cartesian        // 1 = no per-ADC k-data follows; 0 = data follows
if is_cartesian == 0:
    int32  ndim            // 2 or 3
    repeat num_adcs times:
        int32  num_samples
        float32[ndim * num_samples]  k_interleaved   // column-major: dim varies fastest
```

**Cartesian classification**: a sequence is marked Cartesian when
`base_rot == eye(3)` AND, for every ADC, the gradient is constant across
the active ADC window on every used axis (i.e. `dk/dt` is constant).
This matches pulseg's cache behaviour, which omits the trajectory
section for Cartesian acquisitions.

**Units**: k-space samples are in 1/m (Pulseq toolbox convention),
cumulative since the last excitation. `num_adcs` counts ADCs across all
averages; the per-average trajectory is tiled `num_averages` times.

**Phase A limitations**:
- No per-ADC encoding-space ID, rotation ID, or label tuple.
- Cartesian-vacuous fallback when `calculateKspacePP()` raises (e.g.
  Octave bug in `rotate3D` -> `makeExtendedTrapezoid` for sequences using
  rotation extensions).

## Other companion files (already produced)

- `<base>.pge`                — pulseg cache binary (v2.0.0 sections 1–7)
                                produced by the `write_cache` CLI as a
                                post-pass to `run_generators.m`. Listed
                                under each entry's `companion_files` in
                                `MANIFEST.json`.
- `<base>_meta.txt`           — human-readable summary
- `<base>_tr_waveform.bin`    — canonical TR waveforms per group
- `<base>_segment_def.bin`    — segment definitions and energies
- `<base>_freqmod_def.bin`    — freq-mod definitions
- `<base>_freqmod_plan.bin`   — projected freq-mod library
- `<base>_scan_table.bin`     — per-block scan-loop entries
- `<base>_label_state.bin`    — per-scan / per-ADC label state truth

See `+testutils/TruthBuilder.m` for the field-by-field layout of each.
