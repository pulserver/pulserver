# Safety checks

```{admonition} TL;DR
:class: tldr

- Before a design is stored, pulserver checks it as the scanner will play it:
  rotated to the prescription, on the physical gradient axes.
- Timing, gradient amplitude and slew rate are always checked. PNS, mechanical
  resonance and sound pressure are checked when the scanner sends a model for
  them.
- A sequence raising a safety check violation is refused: nothing is stored,
  and the operator reads the violations.
- SAR, RF coil heating and gradient heating are computed on the scanner by the
  vendor's routines. pulserver computes their inputs and passes them on, as a
  product sequence does; for pTx this includes SAR ratios from VOPs.
```

Your sequence function designs a scan. pulserver checks it before the scanner
plays it, because a design that respects the scanner's limits on paper can
still exceed them as it plays. On an oblique slice, for example, two logical
axes add up on one physical axis. The checks run on every design call, for
generated designs and imported Pulseq files alike, with the limits the scanner
sends with the call. Lesson 3 of the {doc}`Course <../examples/course>` runs
them by hand on a design that fails.

The checks compute estimates; passing them does not establish scanner or
patient safety.

## What pulserver does

:::{container} capabilities

- **Checks every file of a `NextSequence` chain before it is stored.** A sequence raising a safety check violation is refused, nothing is stored, and the reply lists the violations.

  Code: {func}`~pulserver.ir.check`, called by the design calls. Tests: *a design beyond the scanner limits is refused and stores nothing* (`test_design_service.py`); *an import beyond the scanner limits is refused* (`test_design_service.py`).
- **Rotates each file to the prescription before checking.** The rotation is composed after each block's own rotation.

  Code: `check(rotation=...)`. Test: *a gradient within the limit on each logical axis can exceed it on a physical one* (`test_ir_checks.py`).
- **Leaves blocks labelled `NOROT` unrotated, as the scanner plays them.**

  Code: {func}`~pulserver.ir.check`. Test: *a block labelled NOROT is checked as it plays unrotated* (`test_ir_checks.py`).
- **Checks a prescription that includes a reflection as it plays.** A rotation that is not orthonormal is refused.

  Code: {func}`~pulserver.ir.check`. Tests: *a reflected prescription is checked as it plays* (`test_ir_checks.py`); *a rotation that is not orthonormal is refused* (`test_ir_checks.py`).
- **Designs under per-logical-axis design limits and checks against the gradient coil's per-physical-axis limits.**

  Code: `design_max_grad`, `design_max_slew` in the call's limits. Test: *an oblique design is held under the design limits and checked against the scanner's* (`test_design_service.py`).
- **Checks that each raster of the file (RF, gradient, ADC, block duration) and the scanner's are integer multiples of each other.** The vendor's instruction generation then plays the file's waveforms on the scanner's rasters, by decimation or piecewise-constant interpolation.

  Code: `check_raster_times` in `src/cpp/ir/dedup.cpp`, run by {func}`~pulserver.ir.convert`. Test: *a file raster converts only as an integer multiple of the scanner's* (`test_ir.py`).
- **Checks timing, including gradient continuity, dead times and ringdown, on the rasters the file declares.**

  Code: `pypulseqpp.check_timing`. Test: *a dead time the file does not leave is a timing problem* (`test_ir.py`).
- **Checks gradient amplitude and slew rate on the physical axes.**

  Code: `pypulseqpp.safety.check_max_grad`, `check_max_slew`. Test: *limits without check limits check timing and gradients only* (`test_limits.py`).
- **Checks PNS under the scanner's nerve model (chronaxie or SAFE), up to a fraction of its threshold.**

  Code: `CheckLimits(pns=..., pns_limit=...)`, `pypulseqpp.safety.check_pns`. Test: *PNS is refused from the fraction of threshold the limits allow* (`test_ir_checks.py`).
- **Refuses gradient trains in the forbidden bands of mechanical resonance, on the physical axis they play on.**

  Code: `CheckLimits(bands=...)`, `pypulseqpp.safety.check_mech_resonance`. Tests: *a gradient train in a forbidden band is refused on the axis it plays* (`test_ir_checks.py`); *the prescription moves a train into the forbidden band of another axis* (`test_ir_checks.py`).
- **Computes the sound pressure level of each file's loudest repetition and holds it to 140 dB peak and 99 dB(A) average (IEC 60601-2-33).** The levels are written into the cache.

  Code: `CheckLimits(acoustic=...)`, {func}`~pulserver.ir.spl_levels`, {class}`~pulserver.ir.SplLevels`. Tests: *a repetition too loud on the physical axis it plays on is refused* (`test_ir_checks.py`); *the levels the cache carries are those the check holds to its limits* (`test_ir_checks.py`).
- **For pTx, computes each subsequence's SAR ratios from the VOPs against a reference pulse and writes them into the cache.** They are inputs to the vendor's SAR routine; nothing is refused on them here.

  Code: {func}`~pulserver.ir.sar_ratios`, {class}`~pulserver.ir.SarRatio`. Tests: *a repetition's SAR is measured against the same repetition of reference pulses* (`test_ir_checks.py`); *the checks apply no SAR limit* (`test_ir_checks.py`).
- **Plays the reference pulse in the coil's default channel weights, and applies the VOP file's safety factor to local SAR only.**

  Code: `CheckLimits(default_shim=...)`. Tests: *the reference pulse is played in the default shim* (`test_ir_checks.py`); *the file's safety factor raises the local ratio and not the global* (`test_ir_checks.py`).
- **Refuses a VOP file made for a transmit configuration other than the one the scanner reports.** The virtual scanner designs under the VOPs of the exam's transmit coil.

  Code: `CheckLimits(vop_coil=...)`. Tests: *the VOP file must name the transmit configuration the scanner reports* (`test_ir_checks.py`); *a design is made under the VOPs of the exam's transmit coil* (`test_console.py`).
- **Writes the sequence's request for SAR burst limits into the cache, where the scanner reads it.**

  Code: `EnableSarBurstMode` definition. Test: *a scan asking for SAR burst limits says so in its cache* (`test_ir_repetition.py`).
- **Makes the limits, check limits and VOP file contents included, part of the design's identity, so the same protocol checked under other limits is another design.**

  Code: {func}`~pulserver.host.design_identity`. Test: *a design is one of the VOP file contents its SAR ratios came from* (`test_design_service.py`).

:::

## What the scanner does

- **SAR, RF coil heating and gradient heating.** These are computed on the
  scanner by the vendor's proprietary routines, under the scanner's own
  calibration. The interpreter feeds them the inputs pulserver computes, as a
  product sequence does: the RF pulses and gradients of each subsequence, and
  for pTx the SAR ratios described below.
- **Resampling onto the scanner's rasters.** The vendor's instruction
  generation decimates a waveform on a finer raster and holds the samples of
  one on a coarser raster, which is why pulserver requires the rasters to be
  integer multiples of each other.

A sequence that passes has passed these estimates under the limits the scanner
sent, and nothing more.

## How it works

```{figure} ../_static/safety.svg
:figclass: only-light

A design made under the design limits is rotated to the prescription and
checked on the physical axes. A sequence raising a violation is refused; one
without is converted into the IR cache, which carries its sound pressure
levels and SAR ratios to the interpreter.
```

```{figure} ../_static/safety-dark.svg
:figclass: only-dark

A design made under the design limits is rotated to the prescription and
checked on the physical axes. A sequence raising a violation is refused; one
without is converted into the IR cache, which carries its sound pressure
levels and SAR ratios to the interpreter.
```

### The physical frame

Each gradient coil has its own amplitude and slew-rate limits, and the peak on
one coil depends on the orientation of the slice. Two logical axes at 0.8 of
the amplitude limit each put $0.8\sqrt{2}$ of it on one physical axis at 45°.
So the checks rotate the scan to the prescription first. The rotation $R$, from
logical to physical axes, arrives in the nine `fov_rotation_ij` entries of the
protocol, element $(i, j)$ of $R$.

Your sequence function never sees $R$. It is designed under the *design limits*
`design_max_grad` and `design_max_slew`, per logical axis, which the scanner
derates for the prescription. The checks then hold the physical axes to the
gradient coil's own `max_grad` and `max_slew`. A scanner that sends no design
limits gets the design made under its physical ones, and an oblique slice can
then be refused.

### Where the limits come from

The interpreter sends its limits with every design call, as a `[Limits]` block
({doc}`../user-guide/running`). The gradient limits, dead times and ringdown
time make up `system`. Keys that start with `pns_`, `forbidden_band_`, `vop_`
and `acoustic_` make up the {class}`~pulserver.ir.CheckLimits`. A check whose
model is not sent is left out: without a nerve model there is no PNS check.

### Sound pressure

The level is computed for the repetition with the most gradient energy, the
one {func}`~pulserver.ir.repetition_gradients` reads from the cache. It is
filtered through the gradient coil's acoustic transfer function on each
physical axis, as a periodic waveform, so the level is that of the steady state
the repetition reaches when it is played back to back. Both levels go into the
cache, where the interpreter reads them; a cache written without a transfer
function carries -1.

### SAR against a reference pulse

VOPs are used only for pTx: a multichannel transmit coil is connected, and
the file contains RF shim events or multichannel RF waveforms. The interpreter
then sends the coil's virtual observation points (VOPs). The vendor's SAR
routine is calibrated for a pulse played in the coil's default channel weights,
so pulserver relates every pulse of a subsequence to such a pulse, the
*reference pulse*: hard, 180° and 1 ms, in the default channel weights.

The energy a pulse deposits at VOP $v$ is
$\int \mathbf{b}(t)^H Q_v\, \mathbf{b}(t)\,dt$, with $\mathbf{b}$ the drive of
each transmit channel and $Q_v$ the VOP's matrix. In the head of body model $b$
it is the same integral with that model's head SAR matrix $G_b$. For each
repetition $w$ of a subsequence, {func}`~pulserver.ir.sar_ratios` computes

$$
r_{\mathrm{local}} = \frac{L_{\mathrm{head}}}{L_{\mathrm{local}}} \max_w
\frac{M \max_v E_{v,w}}{N_w \min_b E^{\mathrm{ref}}_{G_b}},
\qquad
r_{\mathrm{head}} = \max_w \max_b \frac{E_{G_b,w}}{N_w\, E^{\mathrm{ref}}_{G_b}},
$$

with $E_{v,w}$ and $E_{G_b,w}$ the energy of the repetition at VOP $v$ and in
the head of body model $b$, $N_w$ the number of pulses it plays,
$E^{\mathrm{ref}}_{G_b}$ the head energy of one reference pulse, $M$ the VOP
file's safety factor, and $L_{\mathrm{head}}$ and $L_{\mathrm{local}}$ the
scanner's head and local SAR limits, `vop_head_limit` and `vop_local_limit`.
The blocks before the first repetition and after the last are included, as
pypulseqpp's SAR check averages over them.

The interpreter gives the reference pulse the shortest time its calibration
allows, the time at which the reference's head SAR reaches $L_{\mathrm{head}}$.
This fixes the drive scale without a measurement of transmitted power: at that
scale, a pulse given $\max(r_{\mathrm{local}}, r_{\mathrm{head}})$ times the
reference pulse's time deposits at most $L_{\mathrm{local}}$ of peak local SAR
and $L_{\mathrm{head}}$ of head SAR. Two consequences are worth checking a
design against:

- a 1 ms hard pulse of 90° counts a quarter of the reference in both terms,
  and a repetition of reference pulses has a head term of 1;
- a scale common to every channel's drive and to the matrices cancels in both
  terms, but the relative channel gains do not.

The smallest reference energy over the body models sets the largest drive
scale, so the local term holds in every model; the head term is taken body
model by body model, since a subject's head SAR is that of one body. The bound
rests on the interpreter's head SAR for the reference not falling below the
true one, and on the VOPs and the safety factor bounding peak local SAR.

The cache carries the two terms of each subsequence in its
`pulseg_subseq_info`, zero without VOPs or without RF. The interpreter passes
each pulse of the subsequence to the vendor's SAR routine as a reference pulse
lasting the reference pulse's time times the larger of the two.

## See it run

- {doc}`../generated/gallery/01-course/03_scanner_representation`: a design
  refused on PNS, and the same design accepted when made at a lower slew rate.
- {doc}`../user-guide/running`: the `[Limits]` keys an interpreter sends.
- {doc}`designs`: what is stored once a design passes.
- {doc}`../api/ir`: {func}`~pulserver.ir.check`, {class}`~pulserver.ir.CheckLimits`
  and {func}`~pulserver.ir.sar_ratios`.
