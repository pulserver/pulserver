# Mechanical resonance

```{admonition} TL;DR
:class: tip

**Criterion.** At every guarded frequency, the equivalent sustained amplitude
$A_\text{eq}(f) = \tfrac{2}{T_\text{TR}}|S_\text{ax}(f)|$ — the amplitude of the
pure sinusoid delivering the same coherent drive — against the vendor's band
limit, in the same mT/m the limit is stated in.

**Window.** The {doc}`per-position spectral bound <canonical_tr>` over the
structural TR: positions identical in every repetition enter a coherent complex
sum, positions that vary contribute the largest magnitude over the combinations
they really take. This is a **proven** ceiling on the scan at every frequency,
not an estimate of it.

**Cost.** One transform per distinct gradient waveform. Every occurrence is
then an amplitude times a phase factor, so the whole analysis is a walk over
the amplitude table — no repetition is evaluated and no waveform is rendered.

**Guards.** Bands are widened by half the narrowest band's width; a
zero-tolerance band is compared against an empirical amplitude floor, because a
literal zero refuses every sequence ever written.
```

A gradient coil sits in a strong static field, so every gradient waveform is
also a Lorentz drive on a large, stiff, lightly damped structure — the coil
former, the cryostat, the bore. Drive one of that structure's resonances and it
is amplified by the Q factor rather than merely transmitted: acoustic noise,
image artefacts from bore vibration, fatigue damage of the gradient assembly.

Vendors therefore publish **forbidden bands**, and the question a sequence must
answer is narrow:

> At the frequencies inside a forbidden band, how much oscillating gradient
> amplitude does this sequence actually deliver?

Not how much acoustic energy is produced, and not a broadband noise figure. A
resonance responds to *coherent, sustained drive at its own frequency*, and
what matters is the amplitude of the sinusoid the sequence effectively presents
there.

## Why the period is the window

Render the whole scan, transform it, look inside each band. That is what an
offline analysis would do, and beyond costing minutes of waveform on a
microsecond raster it has two problems. A single transform of a minutes-long
record has microhertz resolution and no meaningful notion of "the amplitude at
1150 Hz" — the answer depends on an analysis window nobody has a principled
width for. And the record is the scan you happened to prescribe: half as many
repetitions gives a different spectrum for the same physical drive.

A scan is **periodic**, which rescues both. A periodic drive has a line
spectrum — energy at the harmonics $k/T_\text{TR}$ and nothing between — so
"the amplitude at 1150 Hz" is a question about whether a line lands there, and
the repetition period is the principled window. This is why the
{doc}`structural TR <../sequence_model/tr_and_segmentation>` has to be right
for the verdict to mean anything.

## Combs derived from the block content

Seginer et al. ([arXiv 2508.03220](https://arxiv.org/abs/2508.03220)) work this
out in closed form for a multi-echo, multi-slice EPI train: the waveform is one
echo train convolved with impulse trains at the echo spacing, the slice period
and the TR, so its spectrum factorises into the single-train envelope times one
sine-ratio comb per nested periodicity,

$$|g(\omega)| \approx |A_n(\omega)| \cdot
\left|\frac{\sin(\omega\,\Delta T_E N_\text{TE}/2)}{\sin(\omega\,\Delta T_E/2)}\right| \cdot
\left|\frac{\sin(\omega\,\Delta T_\text{sl} N_\text{sl}/2)}{\sin(\omega\,\Delta T_\text{sl}/2)}\right| \cdot (\cdots)$$

Each factor concentrates energy into a comb; where the periods are
commensurate the combs align on one dominant peak, and where one slips off the
common raster the energy spreads. The model is exact for the structure it
describes — but it has to be *told* that structure, and a general Pulseq file
declares none of it.

Pulserver computes the same physics from two exact properties of the transform.
Linearity, and the shift theorem: delaying a waveform by $t_k$ multiplies its
transform by $e^{-j2\pi f t_k}$ and changes nothing else. A sequence is already
a sum of time-shifted, amplitude-scaled copies of a small shape library, so
each unique shape is transformed once — analytically for trapezoids and ramps,
from the raw samples otherwise — and occurrence $k$ contributes

$$a_k(f) = A_k\, W_k(f)\, e^{-j2\pi f t_k}.$$

The per-axis spectrum is the sum over one canonical TR, folded by the actual
finite number of repetitions. This is not an approximation traded for speed: it
is algebraically the number a rendered-waveform transform would give.

Where Seginer et al. write one comb factor per declared periodicity, here every
echo and every slice repeat is one more event inside the one period that is
known — and the combs *emerge*, because a sum of $N$ identical equally spaced
phasors is the sine-ratio comb whether you write the closed form or accumulate
it term by term.

```{figure} ../assets/mechanical_resonance/epi_seginer_fig1_reproduction.png
Reproduction of Seginer et al. Fig. 1 through the compiled engine the
scanner-side check runs: on-raster TE and slice spacing keeps one dominant
peak; off-raster spacing spreads it into a comb.
```

```{figure} ../assets/mechanical_resonance/epi_seginer_component_decomposition.png
Per-event decomposition of one comb peak: 18 event contributions (3 TEs × 6
slices) adding almost perfectly in phase. There is no TE factor or slice factor
anywhere in the engine — only event contributions and a running complex sum.
```

## The verdict

Per axis, per guarded frequency,

$$A_\text{eq}(f) \;=\; \frac{2}{T_\text{TR}}\,\bigl|S_\text{ax}(f)\bigr|,$$

a real gradient amplitude, directly comparable with the vendor's limit. Two
guards turn it into a verdict:

**A frequency guard.** A resonance is not infinitely sharp, and neither is a
harmonic once the scan is finite, so every band is widened by half the width of
the narrowest band in the table — the narrowest band being the sharpest
resonance the vendor identified. Without it, a strong line 40 Hz outside a band
edge would pass a check it should not.

**An amplitude floor.** Every periodic gradient sprinkles weak harmonics into
any band wider than its comb spacing, so a literal zero-tolerance band refuses
every sequence ever written. When a band states zero, the threshold becomes a
fixed equivalent amplitude — the scale of a drive that is actually sustained
rather than incidental. Where that number comes from, and why it cannot be
derived, is on the {doc}`performance page <../performance/mechanical_resonance>`.
A band that does state an amplitude is stating the plateau of a readout train,
so it is converted into the same equivalent-sinusoid convention first; the two
sides of a `>` must be the same quantity.

The criterion separates the two cases a literal per-harmonic test gets wrong in
opposite directions. A small Cartesian GRE's phase-encode blip produces a
broadband transient every TR — flagged by a literal test, harmless because
nothing is sustained, and correctly passed here far below the floor. A bSSFP
readout comb puts a genuine sustained line near 1.2 kHz at several mT/m, and is
flagged. Both verdicts are regression-tested over every shipped plugin.

One property comes free: forbidden bands carry no axis tag, and every axis is
run against every band. A rotation only redistributes a fixed vector of drive
among the physical axes and cannot move energy to a frequency, so no sequence
can hide a dangerous line by rotating it onto an axis the check is not
watching.

The interpreter runs this at **predownload** — once `make_sequence` has written
the finished `.seq` and the file comes back in, not while an operator is still
choosing a parameter. Nothing before that point has built the gradient
waveforms: a console's live feasibility estimate is a duration computed from
module lengths, with nothing to evaluate a band against.

## Scans with a different waveform in every repetition

The period is the window only while there is a period. A scan that plays
more distinct waveform sets than the grouping holds — a distinct optimised
readout in every repetition — has no line spectrum and nothing between lines
for the Dirichlet kernel to attenuate. For such a scan the check prices what
a mode sees: the amplitude sustained inside its band over its memory,
$1/\Delta f$, as a window slid over the whole scan,

$$
A_W(f) = \frac{2}{\text{span}}\,\Bigl|\sum_{t_m \in W} a_m\, T_m(f)\, e^{-i 2\pi f t_m}\Bigr|,
$$

the sum running over the gradient events that start inside the window,
$T_m$ each one's transform, $t_m$ its start, and span the run those events
cover (never less than the window). This is the same quantity as the
periodic rule wherever a period exists — repeat one TR and it reduces to
$\tfrac{2}{T_{TR}}|S_{TR}(f)|$ — and where none does it keeps the cancellation
between repetitions that a bound over instances would discard. The verdict
compares $A_W$ against the band's threshold exactly as above, on a grid
0.25 Hz apart across each band, raised by the guards that make it an upper
bound; the refusal names the frequency, the amplitude and the axis. The
spectrum a sequence plot draws for such a scan is this one. How it is
evaluated and why it is a bound is on the
{doc}`performance page <../performance/mechanical_resonance>`.
