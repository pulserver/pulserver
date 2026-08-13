# Peripheral nerve stimulation

## The physical problem

A changing magnetic field induces an electric field in tissue. Switch a
gradient fast enough and the induced field depolarises peripheral nerve
membranes, which the subject feels — a tap, a twitch, at worst pain. Unlike
[mechanical resonance](mechanical_resonance.md), this is a limit on the
subject, not the hardware, and it is regulatory rather than advisory.

The quantity that stimulates is not the gradient amplitude but its **rate of
change**, $\mathrm{d}G/\mathrm{d}t$, and not its instantaneous value either. A
nerve integrates: a short, intense slew and a longer, gentler one can be
equally stimulating. The threshold therefore depends on pulse *duration*, and
the classical description of that dependence is the strength–duration curve.

Two model families are in use, and Pulserver supports both because they are
the two you meet in practice.

**Irnich rheobase/chronaxie**, the form GE uses. The stimulation threshold
for a rectangular slew pulse of duration $\tau$ is

$$S_\text{thr}(\tau) = \frac{S_\text{min}}{1 - \bigl(\tfrac{c}{c+\tau}\bigr)}
\quad\text{with}\quad S_\text{min} = \frac{\text{rheobase}}{\alpha},$$

where $c$ is the chronaxie time, the rheobase is the asymptotic threshold for
an infinitely long pulse, and $\alpha$ is a coil attenuation factor. Realised
as a linear filter, this is a convolution of the slew waveform with the
kernel

$$k[i] = \frac{\Delta t}{S_\text{min}}\cdot\frac{c}{(c + i\,\Delta t)^2},$$

whose $1/\tau^2$ tail is truncated after 20 chronaxie constants. The result
is a fraction of threshold, per axis, at every instant.

**SAFE** (Szczepankiewicz and Witzel), the form Siemens `.asc` hardware files
describe, and what upstream PyPulseq implements. Pulserver delegates to
upstream for this one rather than reimplementing it.

Both are available from
{meth}`pulserver.pypulseq.Sequence.calculate_pns`, selected by the shape of
the `hardware` argument it is given.

## The naive algorithm

Render the whole scan to a gradient waveform. Differentiate. Convolve with
the kernel. Root-sum-square across axes. Compare the peak against 100 %.

Unlike the mechanical resonance case, **this one cannot be replaced by a
structural shortcut**, and it is worth being precise about why.

The mechanical resonance criterion asks about the spectrum, and the Fourier
transform is linear and shift-covariant, so a sum of time-shifted block
shapes transforms into a sum of phase-rotated block transforms. The verdict
is the *coherent sum* over the whole window, and that sum can be assembled
analytically from per-definition transforms.

PNS asks for the **pointwise maximum of a convolution**. Convolution is
linear too, so the response is likewise a sum of per-event responses — but
the verdict is $\max_t$ of the root-sum-square of that sum, and a maximum is
neither linear nor decomposable. You cannot bound the peak from
per-definition peaks without being either wrong or uselessly conservative:
two adjacent events each at 60 % of threshold may combine to 90 % or to 30 %
depending entirely on their relative timing and sign. The instant of the
peak is a property of the whole window, so the whole window has to be
evaluated.

So PNS genuinely needs a materialised time-domain waveform. What can still
be avoided is materialising the *whole scan*.

## What Pulserver computes

**One canonical TR, circularly padded.** The waveform is built for a single
canonical TR — the same window
[the mechanical resonance analysis](mechanical_resonance.md#stage-1--the-canonical-structural-window)
uses — and the slew rate computed on it. Because a TR is played back to back
with copies of itself, the convolution needs history from before the window
starts; the waveform is therefore **circularly padded** by however many
samples the nerve model declares it needs (`required_padding()` — for the
Irnich kernel, 20 chronaxie constants' worth). The model then sees a
fully warmed-up history, and the peak found inside one TR is the peak of the
steady-state scan.

That padding query is the whole reason the split below exists.

**The core computes slew; the vendor computes stimulation.** `pulseg_calc_pns`
returns $\mathrm{d}G/\mathrm{d}t$ per axis, in T/m/s, and nothing else. The
nerve model is injected as two function pointers — `required_padding(dt)` and
`evaluate(dgdt_x, dgdt_y, dgdt_z, n, dt)` — so the vendor-neutral library
carries no nerve-stimulation constants and no vendor's threshold curve. GE's
implementation lives in `pulserver_ge_pns.c` in the interpreter; the Python
counterpart in `pulserver.pypulseq._safety` is a line-for-line match of it,
so a plot drawn while authoring a sequence and a verdict returned on the
scanner cannot disagree by reimplementation.

This is a deliberate boundary, not an accident of layering. Nerve models and
their coefficients are the part of safety most likely to be vendor-specific,
proprietary, or revised; the padding query is exactly the minimum the core
needs to know about a model in order to hand it a correct waveform.

**Rasterisation.** The slew waveform is built on the gradient raster, from
the same block definitions the rest of the pipeline uses — trapezoid
vertices expanded exactly, arbitrary gradients taken as their raw samples.
No interpolation onto a finer grid, because the raster is where the
hardware's own slew limit is defined.

Cost tracks TR duration rather than sequence complexity, because building
the waveform — not evaluating the nerve model — is almost all of the work;
the measured figures and their scaling against
{doc}`mechanical resonance <mechanical_resonance>` are in {doc}`../benchmarks`.

## Inspecting PNS while authoring

{meth}`~pulserver.pypulseq.Sequence.calculate_pns` is a visualisation, not a
gate. It draws per-axis and combined stimulation against threshold lines so
the margin is visible while a sequence is being written. Which nerve model
runs is decided by the `hardware` argument — a mapping carrying `chronaxie`
and `rheobase` selects Irnich, a Siemens `.asc` path or a per-axis namespace
selects SAFE:

```python
import pulserver.pypulseq as pp

seq = build_my_sequence(pp.Opts())

irnich = {"chronaxie_us": 360.0, "rheobase": 4.25e8, "alpha": 0.333}
ok, norm, components, t = seq.calculate_pns(irnich, tr=0)          # Irnich
ok, norm, components, t = seq.calculate_pns("scanner.asc", tr=0)   # SAFE
```

The hardware description is always passed in, never read off `Opts`: a
sequence author's system limits describe the gradients they are designing
for, not the coil's nerve-response coefficients, and conflating the two lets
a plot silently use the wrong model.

Scoping it to one TR with `tr=` is the right default: it is the window the
scanner evaluates, and it avoids materialising a whole scan to look at a
quantity that is periodic anyway.

The verdict returned is the model's, not the scanner's. The authoritative
gate is `pulseg_check_safety` at predownload, plus the vendor's own checks —
and the reason to keep the inspection view non-authoritative is that a
sequence author's `Opts` need not match the scanner's configuration, so any
pass this printed would be a promise Pulserver is not in a position to make.

## The SAFE model

The chronaxie plots below all use one nerve model — Irnich, GE's — because
that is the arithmetic `pulseg_check_safety` actually runs. Passing a SAFE
hardware description instead covers the other half of the split described
above: it delegates the *entire* stimulation calculation, waveform included,
to upstream PyPulseq's own `calc_pns`, which implements the
Szczepankiewicz/Witzel SAFE model against a Siemens `.asc` file. Pulserver's
contribution here is only the plumbing — normalising the hardware
description, and scoping the call to one TR or segment the same way the
Irnich path is scoped.

Because SAFE needs a real hardware coefficient set and none ships with
Pulserver, the figure below uses upstream PyPulseq's own bundled
`safe_example_hw()` — explicitly documented there as *"EXAMPLE scanner
hardware (not a real scanner)"* — applied to the GRE fixture's first TR:

![GRE SAFE PNS](../assets/pns_safety/gre_2d_pns_safe.png)

The shape is the point, not the number: three per-axis stimulation traces
and a combined norm, drawn against upstream's own threshold line, from the
same slice-select/phase-encode/readout events the Irnich plot below draws.
Read the 66 %/81 % figures as "this is what the SAFE model's output looks
like", not as a statement about any scanner's real margin — that would need
that scanner's actual `.asc` file, passed as `hardware`.

## Reading the plots

`docs/_bench/waveform_plots.py` drives the plot above across the same
five-sequence corpus [the mechanical resonance
page](mechanical_resonance.md#reading-the-plots) uses (representative Irnich
constants, not any particular scanner's), scoped with `tr=` to the
first TR that actually plays out an ADC — every corpus fixture opens with a
handful of non-acquiring dummy TRs to reach steady state, so `tr=0`
alone would show an unrepresentative shot for several of them.

**GRE**, one isolated readout gradient per TR:

![GRE representative TR](../assets/representative_tr/gre_2d_tr.png)

![GRE PNS](../assets/pns_safety/gre_2d_pns.png)

The stimulation waveform mirrors the gradient waveform directly: one rise
for the slice-select lobe, one for the phase-encode/rewind pair, one for the
readout — three isolated spikes per TR, decaying between them because
nothing else is playing.

**EPI**, a long blipped echo train, is the sharpest contrast available in
the corpus and the reason this section exists at all — a single readout
gradient repeated dozens of times a few hundred microseconds apart is a
qualitatively different stimulation problem from GRE's three isolated
events:

![EPI representative TR](../assets/representative_tr/epi_2d_tr.png)

![EPI PNS](../assets/pns_safety/epi_2d_pns.png)

Every blip adds its own rising edge before the previous one has decayed, so
the per-blip peaks ride on a sustained, near-saturated plateau across the
whole train instead of returning to baseline — the nerve model's
convolution/memory genuinely matters here in a way it does not for GRE's
well-separated events. This is also why EPI dominates the cost comparison in
{doc}`../benchmarks`: the same $\mathrm{d}t$ over an order of magnitude more
samples.
