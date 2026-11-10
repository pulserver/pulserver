# Safety checks

```{admonition} TL;DR
:class: tldr

- Every design is checked on the physical gradient axes, rotated to the prescription, before it is stored.
- All checks are mandatory on a scanner except sound pressure; a violation stores nothing.
- SAR and heating run on the scanner, in the vendor's routines; pulserver supplies their inputs.
```

Your sequence function designs a scan under per-axis limits, but on an oblique
slice two logical axes add up on one physical gradient coil. pulserver checks
the scan as it will play, with the limits the scanner sends, for generated and
imported sequences alike. Lesson 3 of the {doc}`Course <../examples/course>`
runs the checks on a design that fails.

The checks compute estimates; passing them does not establish scanner or
patient safety.

## Capabilities

:::{container} capabilities

- **Checks every file of a `NextSequence` chain and refuses a violation, storing nothing.**

  Code: {func}`~pulserver.ir.check`, called by the design calls. Tests: *a design beyond the scanner limits is refused and stores nothing* (`test_design_service.py`); *an import beyond the scanner limits is refused* (`test_design_service.py`).
- **Rotates each file to the prescription, after each block's own rotation, and leaves `NOROT` blocks unrotated.**

  Code: `check(rotation=...)`. Tests: *a gradient within the limit on each logical axis can exceed it on a physical one*; *a block labelled NOROT is checked as it plays unrotated*; *a reflected prescription is checked as it plays*; *a rotation that is not orthonormal is refused* (`test_ir_checks.py`).
- **Designs under per-logical-axis limits and checks against the coil's per-physical-axis limits.**

  Code: `design_max_grad`, `design_max_slew` in the call's limits. Test: *an oblique design is held under the design limits and checked against the scanner's* (`test_design_service.py`).
- **Lowers the slew rate of a scan past the nerve or slew-rate limit on the envelope of its repetition and designs the scan once at that rate, and the console's protocol on its representative sequence; refuses a strict design as designed.**

  Code: `pulserver.host._slew`, the representative `sequence` of {class}`~pulserver.design.Evaluation`, {meth}`~pulserver.design.SequencePlugin.envelope` and {func}`~pulserver.design.repetition_envelope`, {attr}`~pulserver.design.SequencePlugin.strict_limits`, the `DesignMaxSlew` and `StrictLimits` definitions. Tests: *a design past the nerve limit is designed once at the slew rate its envelope finds*; *a design its envelope misjudges is designed again at a lower slew rate*; *a strict plugin is refused as designed and says so in its files*; *a sequence setting StrictLimits unannounced is refused as designed*; *validate reports the slew rate generate designs at*; *a shipped FSE at the hardware slew rate is designed once within the nerve limit*; *a lower slew rate is not tried for an amplitude past its limit* (`test_slew_derate.py`); *the envelope of a shipped scan reads at least what the scan does* (`test_envelope.py`).
- **Designs a scan that follows motion-correction poses for every rotation, and checks it so.**

  Code: `design_max_grad_any_rotation`, `design_max_slew_any_rotation` and `pmc_enable` in the call's limits; `CheckLimits.follows_poses`. Tests: *a navigated scan at a site following poses is designed for every rotation*; *the vector is held to the axis limit only where poses may turn the scan*; *a sequence setting EnablePmc unannounced is refused where poses could turn it* (`test_any_rotation.py`).
- **Checks that the file's rasters and the scanner's are integer multiples of each other.**

  Code: `check_raster_times` in `src/cpp/ir/dedup.cpp`, run by {func}`~pulserver.ir.convert`. Test: *a file raster converts only as an integer multiple of the scanner's* (`test_ir.py`).
- **Checks timing: gradient continuity, dead times and ringdown.**

  Code: `pypulseqpp.check_timing`. Test: *a dead time the file does not leave is a timing problem* (`test_ir.py`).
- **Checks gradient amplitude and slew rate on the physical axes.**

  Code: `pypulseqpp.safety.check_max_grad`, `check_max_slew`. Test: *limits without check limits check timing and gradients only* (`test_limits.py`).
- **Refuses an RF pulse peaking above the transmit chain's peak B1, `max_b1`.**

  Code: `_peak_b1` in {func}`~pulserver.ir.check`. Test: *an RF pulse peaking above the coil's B1 limit is refused* (`test_ir_checks.py`).
- **Refuses readouts that open less than `adc_gap_us` after the previous one ends, and a block that plays an RF pulse and an ADC together.**

  Code: `CheckLimits(adc_gap=...)`. Tests: *readouts closer than the receiver restarts are refused*; *a block playing an RF pulse and an ADC is refused* (`test_ir_checks.py`).
- **Refuses an arbitrary gradient longer than one axis's waveform memory, `ir_wave_max_samples` at `ir_wave_raster_us`.**

  Code: `CheckLimits(wave_memory=...)`. Test: *an arbitrary gradient longer than the waveform memory is refused* (`test_ir_checks.py`).
- **Checks PNS under the scanner's nerve model (chronaxie or SAFE), its response summed over the axes, up to a fraction of its threshold.**

  Code: `CheckLimits(pns=..., pns_limit=...)`, `pypulseqpp.safety.check_pns`. Test: *PNS is refused from the fraction of threshold the limits allow* (`test_ir_checks.py`).
- **Refuses gradient trains in the forbidden bands of mechanical resonance, on the physical axis they play on, read over a 20 ms window: a train shorter than that drives a band less.** A band whose table states no amplitude is held to 8 mT/m. A stated amplitude is read as an echo-spacing lockout reads it: the band is refused when it holds a line above 6 mT/m while the axis's gradient exceeds the amplitude. A window that exceeds a band is read again over 80 ms; a steady line that then lies outside the band, at 80 % of its amplitude or more, is leakage through the 20 ms window and is not refused.

  Code: `CheckLimits(bands=...)`, `RESONANCE_WINDOW`, `RESONANCE_FLOOR`, `pypulseqpp.safety.check_mech_resonance`. Tests: *a gradient train in a forbidden band is refused on the axis it plays* (`test_ir_checks.py`); *a train drives a band over the 20 ms memory of the resonance* (`test_ir_checks.py`); *a band stating no amplitude is held to 8 mT/m* (`test_ir_checks.py`); *a stated amplitude limits the gradient while the band holds a line* (`test_ir_checks.py`); *the prescription moves a train into the forbidden band of another axis* (`test_ir_checks.py`).
- **Holds the loudest repetition to 140 dB peak and 99 dB(A) average (IEC 60601-2-33), and writes both levels into the cache.**

  Code: `CheckLimits(acoustic=...)`, {func}`~pulserver.ir.spl_levels`, {class}`~pulserver.ir.SplLevels`. Tests: *a repetition too loud on the physical axis it plays on is refused* (`test_ir_checks.py`); *the levels the cache carries are those the check holds to its limits* (`test_ir_checks.py`).
- **For pTx, computes SAR ratios from the VOPs and writes them into the cache for the vendor's SAR routine.**

  Code: {func}`~pulserver.ir.sar_ratios`, {class}`~pulserver.ir.SarRatio`. Tests: *a repetition's SAR is measured against the same repetition of reference pulses* (`test_ir_checks.py`); *the checks apply no SAR limit*; *the reference pulse is played in the default shim*; *the file's safety factor raises the local ratio and not the global* (`test_ir_checks.py`).
- **Refuses a VOP file made for a transmit configuration other than the one the scanner reports.**

  Code: `CheckLimits(vop_coil=...)`. Tests: *the VOP file must name the transmit configuration the scanner reports* (`test_ir_checks.py`); *a design is made under the VOPs of the exam's transmit coil* (`test_console.py`).
- **Writes the sequence's request for SAR burst limits into the cache, where the scanner reads it.**

  Code: `EnableSarBurstMode` definition. Test: *a scan asking for SAR burst limits says so in its cache* (`test_ir_repetition.py`).
- **Makes the limits and the VOP file part of the design's identity.**

  Code: {func}`~pulserver.host.design_identity`. Test: *a design is one of the VOP file contents its SAR ratios came from* (`test_design_service.py`).

:::

## Division of responsibility

- **The design limits**, by the interpreter, from the scanner's gradient
  specification and the prescription's rotation.
- **The waveforms played**, those of the file. A slew rate is lowered by
  designing again, so the file is what plays and states the slew rate in its
  `DesignMaxSlew` definition; an interpreter refuses what it cannot play and
  changes no waveform.
- **SAR, RF coil heating and gradient heating**, in the vendor's routines,
  from the RF pulses and gradients the interpreter passes on, as for a product
  sequence.
- **Resampling onto the scanner's rasters**, by decimation or
  piecewise-constant hold, in the vendor's instruction generation.

## Method

```{figure} ../_static/safety.svg
:figclass: only-light

Rotated to the prescription, checked on the physical axes, then refused or
converted into the IR cache with its sound pressure levels and SAR ratios.
```

```{figure} ../_static/safety-dark.svg
:figclass: only-dark

Rotated to the prescription, checked on the physical axes, then refused or
converted into the IR cache with its sound pressure levels and SAR ratios.
```

### Physical frame

Two logical axes at 0.8 of the amplitude limit put $0.8\sqrt{2}$ of it on one
physical axis at 45°. So the sequence is designed under `design_max_grad` and
`design_max_slew`, the share of each coil's limits the prescription's rotation
leaves a logical axis with all three active, less the design margin of
`pypulseqpp.apply_system_derates`. Every scan is then checked as it plays:
rotated, on each physical axis, against `max_grad` and `max_slew`, the coil's
own per-axis limits. Each physical gradient has its own amplifier, so these
are the limits of the scan's own rotation; the gradient summed over the axes is
not limited by them. The rotation arrives in the protocol's `fov_rotation_ij`
entries; your sequence function never sees it.

A scan that follows motion-correction poses is turned again during the scan,
by rotations no check sees: where the site follows poses (`pmc_enable`) and
the sequence sets `EnablePmc`, it is designed under the `_any_rotation`
limits, and its gradient amplitude and slew rate summed over the three axes, a
root sum of squares, are held to the per-axis limits, the most any rotation
can put on one axis.

### Slew rate and nerve stimulation

A gradient ramp stimulates peripheral nerves by its field's rate of change,
which sums over the axes. The nerve model responds to a ramp of amplitude step
$\Delta G$ at slew rate $S$ with $\Delta G / (b\,(c + \Delta G/S))$, for
rheobase $b$ and chronaxie $c$: a short ramp at a high slew rate stimulates
less than its slew rate alone suggests, so the slew rate a scan may play
depends on the amplitudes it ramps. The design limits are the coil's; a scan
whose response would exceed `pns_limit` is designed at a lower slew rate,
chosen from the response measured at the last one tried, as a vendor sequence
lowers its ramps for its own waveforms.

The slew rate is searched on a bound of the scan rather than on the scan.
Every copy of a scan's repetition plays the same blocks, which differ from
copy to copy only in their gradient amplitudes and rotation: the phase and
partition encodes and their rewinders, a spoke's angle. At each block these
are, per copy, one vector per gradient shape, and the copies' vectors fill a
line or a plane. The envelope plays the repetition at the vertices of an
interval or a polygon circumscribing them, blocks that vary together held at
the same vertex. Slew rate on any axis, under any prescription rotation, and
the nerve response, a linear filter of it, are convex in those vectors, so the
envelope reads at least what the scan does, in one repetition per vertex
instead of every copy; a scan whose blocks vary in more than a plane, a blade
turned while it is phase encoded, is its own envelope. The scan is then
designed and checked once, and the check of the scan stays the gate: a scan
still past a limit is designed again. The console's protocol is evaluated on
the plugin's representative sequence alone, which designs no scan. A plugin
that wants a design refused rather than adjusted is strict, and says so in its
files with the
`StrictLimits` definition ({doc}`../developer-guide/internals/sequence-definitions`).

### Limit sources

The interpreter sends a `[Limits]` block with every call
({doc}`../user-guide/running`): gradient limits and dead times, the nerve
model, the forbidden bands, the VOPs for pTx, and the acoustic transfer
function where the scanner has one. Called directly, as in the Course,
{func}`~pulserver.ir.check` leaves out a check whose model it is not given.

### Sound pressure and SAR

Sound pressure is that of the repetition with the most gradient energy,
filtered through the coil's acoustic transfer function and played back to
back. SAR ratios relate each subsequence's pulses to a 1 ms, 180° hard pulse
in the coil's default channel weights, the pulse the vendor's SAR routine is
calibrated for; {doc}`../developer-guide/internals/sar-ratios` derives them.

## Examples

- {doc}`../generated/gallery/01-course/03_scanner_representation`: a design
  refused on PNS, and the same design accepted when made at a lower slew rate.
- {doc}`../user-guide/running`: the `[Limits]` keys an interpreter sends.
- {doc}`architecture`: what is stored once a design passes.
- {doc}`../api/ir`: {func}`~pulserver.ir.check`, {class}`~pulserver.ir.CheckLimits`
  and {func}`~pulserver.ir.sar_ratios`.
- {doc}`../developer-guide/internals/sequence-definitions`: the definitions a
  design writes and the checks read.
