# Mechanical resonance

A gradient coil sits in a strong static field. Driving current through it
produces a Lorentz force, so every gradient waveform is also a mechanical
drive on a large, stiff, lightly damped structure — the coil former, the
cryostat, the bore. That structure has resonances, and a drive that lands on
one is amplified by the structure's Q factor rather than merely transmitted.
The consequences run from unpleasant acoustic noise, through image artefacts
from bore vibration, to fatigue damage of the gradient assembly.

Vendors therefore publish **forbidden bands**: frequency ranges in which the
oscillating gradient amplitude must stay below a stated limit, often zero.
The question a sequence has to answer before it is allowed to run is narrow
and specific:

> At the frequencies inside a forbidden band, how much oscillating gradient
> amplitude does this sequence actually deliver?

Note what the question is *not*. It is not "how much acoustic energy is
produced", and it is not a broadband noise figure. A forbidden band is a
statement about a resonance, and a resonance responds to *coherent, sustained
drive at its own frequency*. The quantity that matters is the amplitude of
the sinusoid the sequence effectively presents there — in mT/m, the same
units the vendor's limit is stated in.

## Why an amplitude threshold is not well posed

Render the whole sequence to a gradient waveform, one sample per raster tick,
per axis. Take a Fourier transform. Look inside each forbidden band. Compare
against the limit.

This is what any offline analysis would do, and it has problems beyond cost
(a scan is minutes of waveform at a microsecond-scale raster). A single
transform of a minutes-long record has microhertz resolution and no
meaningful notion of "the amplitude at 1150 Hz" — the answer depends on the
analysis window, and there is no principled window width, because the
sequence's own structure supplies the timescale, not the analyst. And the
record is the scan you happened to prescribe: half as many repetitions gives
a different spectrum for the same physical drive, so the verdict would depend
on scan length rather than on what the coil experiences per unit time.

What rescues the problem is that a scan is **periodic**. A periodic drive has
a line spectrum — energy at the harmonics $k/T_\text{TR}$ of the repetition
and nothing in between — so "the amplitude at 1150 Hz" is a question about
whether a *line* lands there, and how strong it is. The repetition period is
the principled window. This is why the {doc}`structural TR
<../sequence_model/tr_and_segmentation>` has to be right for the verdict to
mean anything.

## The spectrum of a nested-loop sequence

Seginer et al. ([arXiv 2508.03220](https://arxiv.org/abs/2508.03220)) work
this out in closed form for the sequence family that matters most
acoustically: a multi-echo, multi-slice EPI train. Their model treats the
gradient waveform as one echo train convolved with trains of impulses at the
echo spacing, the slice period and the TR, so its spectrum factorises into
the single-train envelope multiplied by one sine-ratio comb per nested
periodicity:

$$|g(\omega)| \approx |A_n(\omega)| \cdot
\left|\frac{\sin(\omega\,\Delta T_E N_\text{TE}/2)}{\sin(\omega\,\Delta T_E/2)}\right| \cdot
\left|\frac{\sin(\omega\,\Delta T_\text{sl} N_\text{sl}/2)}{\sin(\omega\,\Delta T_\text{sl}/2)}\right| \cdot (\cdots)$$

Each factor concentrates energy into a comb; where the echo, slice and TR
periods are commensurate the combs align and one dominant peak carries the
drive, and where a period slips off the common raster the energy spreads into
a comb of comparable peaks. The paper's Fig. 1 demonstrates exactly this on
EPI.

The model is exact for the structure it describes — but it has to be *told*
the structure: the echo spacing, the echo count, the slice period, the TR,
each an explicit factor. A general Pulseq file declares none of them.

## Combs derived from the block content

Pulserver computes the same physics without being told any inner period,
using two exact properties of the Fourier transform and the fact that a
Pulseq sequence is already written in the form they want. The transform of a
sum is the sum of the transforms (*linearity*), and delaying a waveform by
$t_k$ multiplies its transform by $e^{-j2\pi f t_k}$ and changes nothing else
(*the shift theorem*). A sequence is a sum of time-shifted,
amplitude-scaled copies of a small library of gradient shapes, so each unique
shape is transformed once — analytically for trapezoids and ramps, from the
raw samples for arbitrary waveforms — and occurrence $k$, with signed
amplitude $A_k$, starting at $t_k$, contributes

$$a_k(f) = A_k\, W_k(f)\, e^{-j2\pi f t_k},$$

with the per-axis spectrum the coherent complex sum over one canonical TR,
folded by the *actual, finite* number of repetitions of that TR in the scan.
This is not an approximation traded for speed: it is algebraically the same
number the rendered-waveform transform would give.

Where Seginer et al. write one closed-form comb factor per declared
periodicity, here every echo of the train and every slice repeat is simply
one more event inside the one period that is known — and the combs *emerge*,
because a sum of $N$ identical equally spaced phasors is the sine-ratio comb
whether you write the closed form or accumulate it term by term. The figures
below reproduce the paper's Fig. 1 — echo spacing 0.52 ms, 54 echoes, 3 TEs,
6 slices — through the same compiled engine the scanner-side check runs:

```{figure} ../assets/mechanical_resonance/epi_seginer_fig1_reproduction.png
Reproduction of Seginer et al. Fig. 1: on-raster TE and slice spacing keeps
one dominant peak; off-raster spacing spreads it into a comb.
```

As in the paper, spacings that stay on the common raster keep the energy in
one dominant peak no matter how many echoes or slices are added, while a
sub-millisecond off-raster shift spreads it into a comb of comparable peaks.
And because every instance is a real event rather than a factor in a
closed form, the engine can export the individual contributions that sum to
a line:

```{figure} ../assets/mechanical_resonance/epi_seginer_component_decomposition.png
Per-event decomposition of one comb peak: 18 event contributions adding in
phase.
```

At the comb peak, all 18 materialised events (3 TEs × 6 slices) add almost
perfectly in phase. There is no "TE factor" or "slice factor" anywhere in the
engine — only event contributions and a running complex sum.

## The verdict

For each frequency $f$ worth checking, the per-axis figure is the
**equivalent sustained amplitude**

$$A_\text{eq}(f) \;=\; \frac{2}{T_\text{TR}}\,\bigl|S_\text{ax}(f)\bigr|,$$

the amplitude of the pure sinusoidal gradient that would deliver the same
coherent drive at $f$ — a real gradient amplitude, directly comparable with
the vendor's limit. Two guards turn it into a verdict:

- **A frequency guard.** A resonance is not infinitely sharp, and neither is
  a harmonic once the scan is finite, so every band is widened by half the
  width of the narrowest band in the table — the narrowest band being the
  sharpest resonance the vendor identified. Without it, a strong line sitting
  40 Hz outside a band edge would pass a check it should not.
- **An amplitude floor.** A zero-tolerance band cannot be taken literally:
  every periodic gradient sprinkles weak harmonics into any band wider than
  its comb spacing, so a literal zero refuses every sequence ever written.
  When a band states zero, the threshold becomes a fixed equivalent
  amplitude, the scale of a drive that is actually sustained rather than
  incidental; where the threshold comes from, and why it cannot be derived,
  is on the {doc}`performance page <../performance/mechanical_resonance>`.
  A band that does state an amplitude is stating the plateau of a readout
  train, so it is converted into the same equivalent-sinusoid convention
  before the comparison — the two sides of a `>` must be the same quantity.

The criterion separates the two cases a per-harmonic zero-tolerance test gets
wrong in opposite directions. A small Cartesian GRE's phase-encode blip
produces a broadband transient every TR — flagged by a literal test, harmless
to the coil because nothing is *sustained*, and correctly passed here, far
below the floor. A bSSFP readout comb puts a genuine sustained line near
1.2 kHz at several mT/m — and is flagged, as it must be. Both verdicts are
regression-tested over every shipped plugin.

One more property comes free: forbidden bands carry no axis tag, and the
check runs every axis against every band. A rotation — a per-block rotation
extension, an oblique prescription — only redistributes a fixed vector of
gradient drive among the three physical axes; it cannot move energy to a
frequency, so no sequence can hide a dangerous line by rotating it onto an
axis the check is not watching.

This is the check the interpreter runs at **predownload** — once
`make_sequence` has written the finished `.seq` and the file comes back
in, not while an operator is still choosing a parameter. Nothing before that
point has built the actual gradient waveforms to check: a console's live
feasibility estimate is a duration computed from module lengths, and has
nothing to evaluate a forbidden band against. How the check is made cheap
enough to run at that point, at the size a real protocol reaches — and the
demonstration that the fast evaluation equals the rendered-waveform one — is
on the {doc}`performance page <../performance/index>`.

