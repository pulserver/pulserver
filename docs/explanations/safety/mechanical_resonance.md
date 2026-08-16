# Mechanical resonance

A gradient coil is a spring. Driven at a frequency it resonates at, it
converts a modest gradient into a large mechanical amplitude — noise, image
artefact, and in the worst case damage. Systems therefore ship a table of
**forbidden bands**: frequency ranges in which the gradient drive must stay
below a stated amplitude, often zero.

The check is easy to state and easy to get wrong: *is this sequence driving
the coil inside a forbidden band?* Getting it wrong in one direction refuses
sequences that are fine; in the other it passes sequences that hammer the
magnet.

## The drive is a line spectrum, not a broadband one

A scan is periodic. Its gradient waveform is therefore not a smooth spectrum
but a **comb**: energy at the harmonics $k/T_{TR}$ of the repetition, and
nothing in between. Whether a coil is being driven at 1180 Hz is a question
about whether a *line* lands there, not about how much broadband energy sits
nearby.

That is why the criterion is evaluated at the TR harmonics that fall inside a
band, and why the structural TR has to be right for the answer to mean
anything. Inner periodicities — an echo train, a slice loop — are not
declared anywhere; they emerge from the coherent sum of the individual event
instances inside the one period that is known.

## The equivalent sustained amplitude

For each in-band harmonic $f_L$, the per-axis figure is

$$
A_{eq}(f_L) \;=\; \frac{2}{T_{TR}} \left| \sum_{\text{events}} a\,
\hat{g}(f_L)\, e^{-j 2\pi f_L t_{\text{event}}} \, D(f_L) \right|
$$

— the amplitude of a *pure sinusoidal gradient* that would deliver the same
sustained on-resonance drive as the sequence does. Each event contributes its
own transform $\hat{g}$ scaled by its instance amplitude $a$ and phased by
when it plays; $D$ accounts for the finite number of repetitions.

Reducing the question to "what constant sinusoid is this equivalent to"
is what makes the verdict comparable with a vendor limit, which is stated in
exactly those terms: milli-tesla per metre, not spectral density.

## Two guards, and why each exists

**A frequency guard.** A resonance is not infinitely sharp, and neither is a
harmonic once the acquisition is finite. A line counts against a band if it
lies within `guard` of it, where `guard` is the half-width at half-maximum
implied by the narrowest band the vendor listed — the narrowest band is the
sharpest resonance they identified; wide bands are keep-out ranges rather
than resonances. Without this, a strong line 40 Hz outside a band edge would
pass a check it should not.

**An amplitude floor.** A zero-tolerance band cannot be taken literally.
Every periodic gradient sprinkles weak harmonics into any band wider than its
comb spacing, so $\varepsilon = 0$ refuses every sequence ever written. When
a band states literal zero, the floor becomes hardware-anchored,

$$\varepsilon = 0.08 \, G_{max}$$

— about 4 mT/m on a 50 mT/m system: the scale of a *readout*, i.e. of a drive
that is actually sustained rather than incidental. Any **nonzero** vendor
limit is trusted as stated, even below that floor; the substitution only
generalises "zero" into something a real sequence can be measured against.

## What this rejects, and what it keeps

The criterion was chosen against two cases that a naive per-harmonic test
gets wrong in opposite directions:

- A **32×32 GRE with a phase-encode blip** produces a broadband transient
  every repetition. A per-harmonic zero-tolerance test flags it; the coil
  does not care, because nothing is *sustained*. Its $A_{eq}$ in band is far
  below the readout floor, and it passes.
- A **bSSFP readout comb** puts a genuine sustained line at ~1176 Hz in the
  corpus protocol, at ~7.9 mT/m. It is flagged, as it must be. So is an EPI
  train at ~1158 Hz.

Both verdicts are corpus regression tests — see
{doc}`../validation/sequence_zoo` — and both are $G_{max}$-dependent by
design: a coil with a smaller maximum gradient has a smaller floor and would
flag a weaker line, which is the correct behaviour rather than an
inconsistency.

## Cost

The evaluation is a Fourier transform of the canonical TR at a set of chosen
frequencies, not a full FFT of the scan. Because only in-band harmonics are
ever evaluated, the cost is set by the band table rather than by the scan
length; a chirp-z transform computes the display comb, which is what took the
analysis from 5.5 s to 265 ms on a Cartesian protocol.

## Looking at it

```python
freqs, spectrum, *_ , lines = seq.calculate_gradient_spectrum(
    plot=True, tr="worst_case", resonance_lines=True, bands=bands,
)
lines.line_freqs      # the harmonics inside a guarded band
lines.line_a_eq       # their equivalent sustained amplitudes, per axis
lines.ok              # the verdict the predownload gate will reach
```

The overlay draws the bands, the comb, and the floor, so a sequence that
fails shows *which* line failed it. `lines.ok` is the same verdict
`check_safety` reaches — a plot that disagreed with the gate would be worse
than no plot.
