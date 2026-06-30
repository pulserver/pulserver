# Pulserver as a Concrete Implementation of PulSeg

**Status:** draft · **Target spec:** PulSeg IR `2.1-alpha`

This document describes how the Pulserver interpreter realizes the
[PulSeg intermediate representation](https://github.com/HarmonizedMRI/pulseg). It is written
for sequence developers, vendor-interpreter authors, and the PulSeg maintainers. It contains
no vendor-proprietary scanner internals; it describes only how Pulserver's data model maps onto
the abstract PulSeg concepts.

The central claim is:

> **Pulserver's in-memory sequence representation is a concrete realization (a "derived class")
> of the PulSeg abstract IR.** Every PulSeg structural concept has a direct Pulserver counterpart,
> and every additional field Pulserver carries is either (a) a lossless re-encoding of canonical
> PulSeg content, or (b) non-normative metadata explicitly permitted by the specification.
> Pulserver therefore *conforms semantically* to PulSeg without ever materializing an
> intermediate IR file.

---

## 1. Architecture: fused conversion, semantic conformance

A naïve pipeline converts in two materialized hops:

```
Pulseq .seq  ──►  PulSeg IR (file)  ──►  vendor format
```

Pulserver instead **fuses** the two hops: it parses a Pulseq `.seq` directly into its
vendor-specific representation in a single pass, and never writes an intermediate IR file.

```
Pulseq .seq  ──►  Pulserver representation  ( ≡ a valid PulSeg instance, projectable on demand )
```

Conformance is therefore a property of the *object*, demonstrable by projection, rather than of
a file exchanged between tools. This avoids both an explicit conversion step and the
double parse/emit cost of a materialized intermediate, while preserving the ability to emit a
canonical PulSeg view for validation (see §10).

PulSeg is **not** an inter-site interchange format. The shared, portable content format is
Pulseq itself; PulSeg is the abstract structural pattern that a vendor instantiates when it needs
structure Pulseq does not carry (notably the *segment* and *TR* concepts). Pulserver is one such
instantiation.

---

## 2. Concept mapping

| PulSeg concept (spec §) | Pulserver counterpart | Notes |
|---|---|---|
| **BaseBlock** (§3.1) — a normalized Pulseq block, `id ≥ 2` | Unique, deduplicated block definition (normalized RF / gradient / ADC events) | Pulserver dedups identical normalized blocks; the dedup key is exactly PulSeg's "same normalized base-block structure". |
| **Reserved delay blocks** `id 0`/`1` (§3.1) | Pure-delay blocks, constant vs variable duration | Variable-duration delays are the TE/TR-fill blocks (bounded by Pulserver's min/max-TR machinery). |
| **VirtualSegment** (§3.2) | **Pulserver Segment** | A `TRID`-keyed, reusable, ordered list of base blocks mapped to a contiguous region of sequencer instruction memory. See §3. |
| **SegmentInstance** (§3.3) | Per-instance resolved block parameters | RF amplitude/phase/frequency, signed per-axis gradient scaling, rotation, ADC offsets, block durations, shot index. |
| **Execution stream** (§3.4) | Ordered traversal of segment instances (the cursor) | A multi-subsequence collection is the concatenation of execution streams; subsequence boundaries are carried as non-normative metadata. |
| **`TRID` boundary annotation** (§4.2) | `TRID` segment labels (inherited from the pge2 convention) | Identical mechanism: first block of each instance carries `TRID`; repeats share structure. |

---

## 3. Segment vs TR — Pulserver supplies the distinction PulSeg needs

PulSeg's **virtual segment** is, mechanically, defined by the `TRID` label and the "repeated
`TRID` ⇒ same structure" rule (§4.2). That is exactly Pulserver's **Segment**: a concrete,
reusable unit of sequencer memory. Pulserver maps each segment to a contiguous region of
instruction memory and replays it with per-instance amplitudes/phases/offsets.

Pulserver additionally maintains a separate **TR** concept that PulSeg deliberately does *not*
require:

- The **Segment** is the *structural / memory* unit. It carries no assumption of periodicity.
- The **TR** is an *abstract periodicity* unit used for vendor safety checks (worst-case-TR RF
  and SAR limits) and for efficient vendor-neutral analyses (mechanical resonance, PNS) that
  rely on periodicity.

This is the right division for an IR. Pulseq does not restrict sequences to be periodic, so an
IR whose fundamental unit *was* the TR would be strictly less expressive than the format it
represents. By keeping the **Segment** (non-periodic, reusable) as the PulSeg unit, and treating
**TR** as an optional periodicity property layered on top, Pulserver supports both periodic
sequences (the common case, where it leverages periodicity for safety/efficiency) and
non-periodic ones (e.g., a fast-spin-echo train with individually optimized echoes), which are
simply represented as distinct segments / unique instances.

Per PulSeg §5 / scanner-backend scope, all TR-based safety validation lives in Pulserver's
backend layer and is **out of PulSeg IR scope**.

---

## 4. Normalization and units

Pulserver normalizes block definitions exactly as PulSeg §3.1 requires:

- single-channel RF normalized to peak magnitude `1.0`;
- multi-channel (dynamic pTx) RF normalized by a single global factor across channels, preserving
  inter-channel relative amplitude;
- gradients normalized per axis (peak `1.0`, or `|amplitude| = 1.0` for trapezoids);
- ADC windows copied unnormalized.

Per-instance physical values are recovered by `physical = normalized × scale`, with signed
gradient scaling and optional rotation, matching §3.3.

**Units.** Because PulSeg is not an interchange format, Pulserver stores values in vendor-native
units internally (e.g., integer microseconds for durations) for hardware efficiency. This is
standard derived-class behavior: different internal storage, same conceptual contract. When a
canonical PulSeg view is emitted (§10), durations are expressed in seconds, frequencies in Hz,
phases in radians, per the spec.

---

## 5. Multishot gradients

Pulserver supports gradient events whose **shape varies per instance** while the **timing is
fixed** — the *shot variant* concept (PulSeg §3.1, "Multishot base gradients", and the
`gradient_shot_index` instance field, §3.3). Two cases are distinguished, exactly as the spec
requires:

- **Shapes related by scaling or rotation** (phase-encode blip tables, radial spokes, rotated
  spiral interleaves) → represented as a **single** shot variant, varied per instance via signed
  amplitude scaling and/or rotation. Pulserver already detects this case (a single
  `(timing-id, shape-id)` tuple shared across all instances) and emits an ordinary single-shot
  base gradient.
- **Genuinely independent co-timed shapes** (e.g., independently optimized spiral interleaves) →
  represented as **multiple shot variants** of one multishot base gradient sharing one timing
  structure, selected per instance by shot index.

Timing-structure identity follows §3.1: `(delay, rise, flat, fall)` for trapezoids,
`(delay, time-shape-id)` for extended trapezoids, `(delay, num_samples)` for uniform-raster
arbitrary gradients. This is the only PulSeg feature for which the reference (MATLAB) converter
returns *unsupported* while Pulserver provides full support.

---

## 6. RF: dynamic and static pTx

- **Single-Tx** and **dynamic pTx** (each transmit channel an independent waveform) map directly
  onto PulSeg core: the per-channel relative amplitudes live in the normalized base block, and a
  scalar per-instance `rf_amplitude` scales them together (§3.1 multi-channel rule).
- **Static pTx** (one RF shape reused across per-instance per-channel complex shim weights) is a
  factorization PulSeg core does not yet express. Per the maintainers' "core first" direction it
  is a **future extension**. Pulserver supports it today and carries the per-channel shim weights
  as **non-normative metadata** (permitted by the spec's "implementations MAY include additional
  metadata" clause), so dynamic-pTx and single-Tx sequences remain strictly core-conformant.

---

## 7. Frequency and FOV

PulSeg core carries **scalar** RF and ADC frequency offsets (§3.3,
`rf_frequency_offset` / `adc_frequency_offset`), which Pulserver realizes as the scanner's
per-event carrier/demodulation frequency offset. Pulserver's per-instance frequency stepping
(its frequency-modulation plan) materializes precisely into these scalar arrays — conformant.

Pulserver additionally realizes **FOV shifts** as a time-varying frequency derived from the
gradient waveform and a per-instance shift vector, `ω(t) = γ · G(t) · Δr`, together with scalar
phase-compensation terms that pin the phase reference (RF isocenter, ADC echo time). These are
**derived realizations**, not independent IR content:

- the time variation comes entirely from the base-block gradient shape, already in the IR;
- the phase-compensation terms are deterministic bookkeeping over canonical phase;
- the only genuine degree of freedom is the per-instance shift `Δr`.

How an FOV shift (and FOV reorientation) is realized — frequency modulation vs phase modulation —
depends on what a given scanner exposes, so it is a **realization detail kept outside PulSeg
core**. The abstract, vendor-neutral description of FOV shift/orientation (covering both the
frequency-based approach used by Pulserver and the phase-based approach used by Pulseq's native
FOV transform) is the subject of a separate, deferred FOV specification. Note that PulSeg core's
`rotation_matrix` is for **trajectory-shot rotation** and is distinct in intent from FOV
reorientation, even though the two are mathematically identical.

---

## 8. Non-normative extensions

Beyond the PulSeg core, Pulserver extracts and carries additional metadata, all of which is
non-normative and ignorable by a pure PulSeg reader:

- k-space trajectory, encoding spaces, and ISMRMRD-style label limits (for reconstruction);
- the frequency-modulation plan (the efficient form of the scalar offsets in §7);
- per-segment safety/TR annotations (worst-case-TR, trigger metadata, navigator flags);
- RF/gradient statistics and mechanical-resonance / PNS analysis inputs.

None of these are required to interpret the core IR, in line with the spec's guidance that
compliant readers ignore unknown fields.

---

## 9. Conformance checklist

| PulSeg requirement | Pulserver |
|---|---|
| Base blocks normalized per §3.1 | ✔ (RF single/dynamic-pTx, gradients per axis, ADC copied) |
| Reserved delay IDs `0`/`1` (constant/variable) | ✔ (pure-delay blocks; variable = TE/TR fill) |
| Virtual segment = `TRID`-keyed reusable unit | ✔ (Pulserver Segment) |
| Periodicity NOT assumed by the IR | ✔ (TR is a separate, optional overlay) |
| Segment instances with per-event scale/phase/freq/duration | ✔ |
| Signed gradient scaling + rotation | ✔ |
| Scalar RF/ADC frequency offsets | ✔ |
| Multishot shot variants + `gradient_shot_index` | ✔ (independent shapes); scale/rotation families collapsed to single shot |
| Lossless w.r.t. supported execution model | ✔ (projectable to canonical PulSeg) |
| Static pTx shim | extension (carried as metadata) |
| FOV shift / reorientation | deferred spec (realization detail) |

---

## 10. Validation strategy

Pulserver's fused converter is validated against the reference PulSeg path as a
**differential oracle**: for any sequence in the shared **core** subset — single-shot or
scale/rotation gradients, dynamic pTx, scalar frequency offsets — the canonical PulSeg view
emitted by Pulserver must agree with the reference (MATLAB / pure-Python) `Pulseq → PulSeg`
converter. Any divergence on that subset is, by definition, a conformance defect.

Features outside the shared oracle (independent-shape multishot, static-pTx shim, FOV
realization) are covered by Pulserver's own simulation-based validation, since the reference
converters return *unsupported* for them.

This gives a low-risk path to growing confidence: the core conformance is continuously
cross-checked against an independent implementation, and the vendor-specific extensions are
exercised by end-to-end scanner-simulation tests.
