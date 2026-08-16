# Pulserver's overlay: a TR, for safety's sake

PulSeg's unit is the segment — a memory concept, agnostic to periodicity.
Worst-case RF and SAR limits, and the periodicity-exploiting safety analyses
({doc}`../safety/mechanical_resonance`, {doc}`../safety/pns`), are stated and
evaluated **per TR** instead: a canonical window that repeats, over which a
worst case can be computed once and trusted for the whole scan. PulSeg does
not supply that window, so Pulserver adds it as an overlay, purely to give
the safety layer something exact to evaluate. Nothing about pulse generation
or scanner playback needs it.

## TR detection

The TR is not in the file, and Pulseq has no TR concept. Detection runs on
the imaging region only — prep and cooldown are stripped first — and looks
for the **shortest period starting at offset 0**, in two passes:

1. **Timing/id pattern.** Each block position becomes a token: its duration
   if the duration is fixed, or the negated definition id if it is variable.
   The shortest $l$ with `s[i] == s[i+l]` for all $i < l$ wins, verified
   across the whole region.
2. **Structural fallback.** If no exact token period exists, the search runs
   again on block *structure* — same duration, same pattern of which event
   slots are occupied — which recovers the period when interleaves give
   different definition ids to structurally identical blocks.

The search deliberately starts at offset 0 rather than scanning all offsets.
Searching from arbitrary offsets finds short sub-patterns that do not span
the region — `[rephaser, nav, rephaser, nav]` inside an EPI readout train is
a real example — and would report a TR shorter than any actual TR.

If neither pass finds a period, the whole imaging region is one TR. That is
the correct answer for a genuinely aperiodic sequence, and it is why the
safety analyses still work on one: they just get a large window. Prep and
cooldown are then compared against the discovered pattern; "degenerate"
means structurally identical to the body, which is what lets the safety
analyses use one imaging TR as their window instead of a whole pass.

## What may vary between TR instances

This is the practical question for a sequence author, and the answers are
enforced rather than advisory — `pulseg_check_consistency` runs inside
conversion, so a sequence that breaks a rule fails to load at all.

| Varies across TR instances | Accepted | Consequence |
| --- | --- | --- |
| Signed gradient amplitude scaling | yes | The phase-encode table. One definition, per-instance scale. |
| RF phase offset | yes | RF spoiling, phase cycling. |
| RF frequency offset | yes | Slice selection across a multi-slice loop. |
| Per-block rotation | yes | Radial spokes, rotated interleaves, oblique prescriptions. |
| Pure-delay duration | yes, with a caveat | TE/TR fill. A duration that changes *monotonically* across TRs breaks the timing period, so TR detection falls back to one large TR — legal, but the analyses lose their small window. |
| **RF amplitude (variable flip angle)** | **yes** (default; `allow_variable_rf_amplitude`) | FSE with individually optimized echo trains. See below for how the worst case is computed. |
| **RF shim weights** | **no** | Rejected: `RF shim id pattern is not periodic across canonical TRs`. |

RF shim weights are still rejected because pTx VOP SAR is a per-pulse
quadratic form in the shim vector, and no single scalar ordering dominates
every shim across every instance the way amplitude does — so there is no
safe envelope to fall back to, and the check fails on the first mismatch.

RF amplitude is different: worst-case RF and SAR limits are computed **per
canonical TR**, and when the amplitude pattern isn't periodic, Pulserver
splits the safety array in two instead of rejecting. The **worst-B1rms real
instance** — the actual TR instance with the highest Σ*A*² over its
pulses — feeds the time-averaged limits (SAR, amplifier duty cycle), because
those integrate over a real window and a synthetic "worst everywhere" TR
would misrepresent it. A **positional-max envelope** (the per-position
maximum amplitude across all instances) feeds the peak-only limits, because
that envelope dominates every instance at every position, which a single
real instance cannot guarantee. For periodic sequences the two coincide with
the canonical TR, so this is bit-identical to the old behavior; for an FSE
train with per-instance-optimized refocusing angles the two usually coincide
with each other too, and only diverge when different instances peak at
different positions within the TR.

The still-workable fallback — needed only for shim variation, or if a
sequence needs to bypass the default via `allow_variable_rf_amplitude=0` —
is to express the acquisition as **separate subsequences**, one per distinct
RF pattern. Each is internally periodic, each gets its own worst-case
analysis, and segment deduplication (see {doc}`pulseg`) collapses whatever
structure they share back into one region of instruction memory — so the
cost of the split is bookkeeping, not memory. Everything in the "yes"
column, by contrast, leaves the RF energy of a TR unchanged (or is handled
by the envelope above), which is why it doesn't need splitting.

## The cache, and why it has sections

Conversion is the expensive step (see {doc}`../benchmarks`). The result is
written to a binary cache beside the `.seq` (`.pseg` by default,
vendor-selectable), so it is paid once per prescription rather than once per
scan — but the consumers of that cache are not the same machine, and the
sections exist because of who reads them:

| Section | Holds | Scales with |
| --- | --- | --- |
| `COMMON` | Collection and subsequence metadata, segment definitions, TR descriptor | Sequence structure |
| `SHAPES` | The deduplicated RF, gradient and ADC waveform library | Unique definitions |
| `INSTANCES` | Per-instance block, RF, gradient and ADC tables | **Scan length** |
| `ROTATIONS` | Per-instance rotation matrices | **Scan length** |
| `SCANLOOP` | The execution stream | **Scan length** |
| `DEFINITIONS` | The `.seq` `[DEFINITIONS]` key/value pairs | Nothing |
| `TRAJECTORY` | k-space trajectory and encoding spaces, for reconstruction | Scan length |
| `SEQDESC` | Event list and RF shape metadata, for reconstruction | Scan length |
| `FREQMOD` | The frequency-modulation plan | Scan length |
| `VENDOR` | An opaque vendor blob, written only if a callback supplies one | — |

The interpreter runs a sequence in two distinct phases, and they need
different halves of that table. **Pulse generation** builds the hardware
waveform images for each segment; it needs the waveform library and the
segment definitions, and nothing that scales with scan length, because
everything it resolves per (segment, block position) is frozen into the
segment definitions' initial-state records at parse time —
`pulseg_load_geninstructions_cache` reads **COMMON + SHAPES only**. **The
scan loop** walks the execution stream, setting per-instance amplitudes,
phases, frequencies and rotations as it goes, so it needs all of it:
`pulseg_load_scanloop_cache` reads COMMON + INSTANCES + ROTATIONS + SHAPES +
SCANLOOP. Reconstruction reads the independent `TRAJECTORY` and `SEQDESC`
sections.

The split is not a micro-optimisation. Pulse generation runs where memory is
scarcest, and the sections it does not read are precisely the ones that grow
without bound as a scan gets longer — measured scaling is in
{doc}`../benchmarks`. That is also the clearest way to see why base-block
deduplication ({doc}`pulseg`) matters: it is what makes the pulse-generation
working set a function of how many *distinct shapes* a sequence contains,
rather than of how long it runs. The representation is therefore not
compression for its own sake: it preserves exactly the instance state a
safety verdict needs before a scan, and that reconstruction needs afterwards.
