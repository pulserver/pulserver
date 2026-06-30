# Mechanical Resonance Safety: Structural Acoustic Analysis

## Overview

MRI gradient systems produce acoustic noise whose spectral content is
determined by the temporal pattern of gradient pulses within and across
repetition periods.  Certain frequencies excite mechanical resonances of
the scanner bore, cryostat, or gradient coil assembly, potentially causing
hardware damage or exceeding acoustic safety limits.  The structural
acoustic analysis module computes a physics-informed spectral model of the
gradient waveform, identifies candidate resonance frequencies, and checks
each candidate against user-defined forbidden frequency bands.

The analysis is performed **per gradient axis independently** (Gx, Gy, Gz)
and **per subsequence** in a sequence collection.  Forbidden bands from
all axes are collected into a single union set: a violation on *any* axis
flags the candidate.

## Analytical spectral model


The spectrum is built from the *structural timing* of gradient events
within one canonical repetition period, without relying on a windowed FFT.
The pipeline consists of the following stages:


### Stage 1 — event extraction and canonical TR structure

For each axis, every block in the canonical TR window is inspected. If a block contains a gradient event, an *occurrence* is recorded:

- **def_id** — gradient definition index.
- **start_time** — cumulative time from the start of the TR (µs).
- **amplitude** — maximum positional gradient amplitude (Hz/m), signed.
   For each gradient definition at each block position, the amplitude is the TR instance whose absolute amplitude is largest across all instances (e.g. if a phase-encode gradient takes amplitudes {−5, 1, 0, 1} across TR instances, the value stored is −5). Together, these per-position maximum amplitudes define a *virtual* canonical TR that represents the worst-case spectral excitation. The sign is preserved so that opposite-polarity instances of the same gradient definition (e.g. bipolar phase-encode steps) contribute the correct phase to the coherent spectral sum.

If the sequence contains non-degenerate prep/cooldown blocks (i.e. where the prep or cooldown varies between passes), the canonical TR is the full *pass* (prep + imaging + cooldown). When `num_averages > 1`, the imaging section repeats `num_averages` times within each pass. Rather than explicitly expanding all copies, each imaging event is tagged with:

- **num_reps** = `num_averages`
- **rep_period_us** = total imaging section duration

Cooldown events are shifted in time to their position in the expanded pass. Prep events are left at their original position with `num_reps = 1`.

This tagging allows the Dirichlet kernel (see below) to handle repetition analytically in $O(1)$ per event per frequency, instead of enumerating all copies.


### Stage 2 — waveform response model

Each gradient definition is classified into one of three response models.

**Trapezoid gradients** are represented as a 4-vertex piecewise-linear
(PWL) waveform:

$$(0, 0) \;\to\; (\tau_r, 1) \;\to\; (\tau_r + \tau_f, 1) \;\to\; (\tau_r + \tau_f + \tau_d, 0)$$

The normalised Fourier transform is computed analytically from the
vertices.  Because the waveform is continuous and starts and ends at
zero, the response decays as $\sim 1/f^2$.

**Arbitrary waveforms with many samples** (≥ 10) are tested for internal
sub-periodicity via normalised autocorrelation.  If a repeating sub-period
is detected (autocorrelation > 0.5, minimum 3 repetitions), the waveform
is *decomposed*: each sub-period becomes a separate event whose PWL model
is the sub-period shape, and consecutive sub-periods emit separate events
at spaced start times.  If no sub-period is detected, the normalised
waveform shape is transformed via a stored per-event FFT:

1. If a time shape is present (non-uniform sampling), the waveform is
   interpolated onto a uniform grid at `grad_raster_us` spacing.
2. The uniform samples are zero-padded to the next power of 2.
3. A real-to-complex FFT (`kiss_fftr`) produces the magnitude spectrum
   $|F[k]|$ for $k = 0, \ldots, N/2$.
4. Magnitudes are normalised by $|F[0]|$ (DC).  If DC $\approx 0$
   (zero-mean waveform), the peak magnitude is used instead.

At query time, $W_k(f)$ is obtained by linearly interpolating between
the two nearest FFT bins, giving accurate spectral response for
waveforms that lack detectable sub-periodicity (e.g., rosette with
incommensurate frequencies, SPARKLING trajectories).  If the FFT
computation fails (allocation), the code falls back to a coarse 16-vertex
uniformly-sampled PWL approximation.

**Arbitrary waveforms with few samples** (< 10) are treated as PWL
directly from their sample values.

### Stage 3 — spectrum assembly (phase-shifted sum)

For each event, the waveform response (from Stage 2) is phase-shifted according to the event's start time within the TR. The complex contribution of a single event $k$ at frequency $f$ is:

$$a_k(f) = A_k \; W_k(f) \; e^{-j\,2\pi f\, t_k}$$

where $A_k$ is the signed amplitude (Hz/m), $W_k(f)$ the normalised waveform response (PWL or FFT-interpolated), and $t_k$ the start time within the TR.

If an event has `num_reps = N > 1`, its contribution is multiplied by the Dirichlet kernel:

$$D_N(f, T) = \frac{\sin(N\,\pi\, f\, T)}{\sin(\pi\, f\, T)} \cdot e^{-j\,(N-1)\,\pi\, f\, T}$$

where $T$ is `rep_period_us` in seconds. This preserves complex phase and produces sharp peaks at multiples of $1/T$.

The **coherent magnitude** on one axis is the magnitude of the complex sum over all events:

$$M_\text{coh}(f) = \Bigl|\,\sum_k a_k(f)\,D_{N_k}(f,T_k)\,\Bigr|$$

The **incoherent magnitude** is the root-sum-of-powers:

$$M_\text{inc}(f) = \sqrt{\sum_k N_k \,|a_k(f)|^2}$$

The coherence ratio $\rho = M_\text{coh} / M_\text{inc}$ measures how strongly events interfere constructively at $f$.


## Evaluation frequency grid

The coherent and incoherent magnitudes are evaluated on a composite
frequency grid:

1. **TR harmonics**: $f_m = m \cdot f_1$ for
   $m = 1, \ldots, \lfloor f_\text{max}/f_1 \rfloor$, where
   $f_1 = 1/T_\text{TR}$ is the fundamental of the canonical TR.
2. **FFT-promoted peaks**: for each axis independently, local maxima of the
   dense FFT whose magnitude exceeds 15% of that axis's own peak are
   identified.  If *any* axis has a qualifying local max at a given FFT bin
   and that frequency falls more than 25% of the harmonic spacing away from
   the nearest TR harmonic, it is added as an extra evaluation frequency.
   This catches spectral features that land between adjacent TR harmonics
   (e.g. bipolar-readout fundamentals in bSSFP).

## Candidate selection

After computing magnitudes at every evaluation frequency, candidates are
selected through a two-stage filter.

### Per-axis analytical power gate

Each axis is gated independently.  The per-axis peak coherent magnitude
squared is

$$P_\text{ax} \;=\; \max_f\, M_{\text{coh,ax}}^2(f)$$

At each evaluation frequency, axis `ax` is rejected if
$M_{\text{coh,ax}}^2(f) < 0.05 \times P_\text{ax}$, unless the axis has
a corresponding entry in its **per-axis FFT bypass set** (see below).
Axes are never mixed: a weak axis cannot be rescued by a strong axis on
a different gradient channel.

### Per-axis FFT bypass set

The analytical waveform response $W_k(f)$ — whether PWL or FFT-based —
can underestimate power at high-order harmonics when using a coarse
approximation (e.g., few PWL vertices for a complex shape).  To
compensate, for each axis independently the dense FFT is scanned for
local maxima whose magnitude exceeds 15% of that axis's own FFT peak.
Each such peak is snapped to its nearest TR harmonic, and that harmonic
is marked to bypass the power gate on that axis only.  FFT-bypassed TR
harmonics use a relaxed tier check (amplitude floor only, no coherence
ratio requirement), since the FFT already confirms a real spectral peak.

### Per-axis tier checks

Frequencies that survive the per-axis power gate are evaluated per axis through
three tiers:

| Tier | Condition |
|------|-----------|
| **Tier 1** — multi-event coherence | ≥ 2 events on the axis, coherence ratio $\rho > 2$, and $M_\text{coh} > 1$ Hz/m |
| **Tier 2** — single-event amplitude | Exactly 1 event and $M_\text{coh} > 10^6$ Hz/m |
| **Tier 3** — FFT-promoted peaks | Between TR harmonics; $M_\text{coh} > 1$ Hz/m |

A frequency becomes a **candidate** if any axis qualifies through any
tier.  The candidate list is shared across axes.

## Forbidden-band check and effective gradient amplitude

For each candidate frequency $f_c$, the effective gradient amplitude
$G_\text{eff}$ is computed only on axes that qualified as candidates at
that frequency.  For non-candidate axes,
$G_{\text{eff},\text{ax}}(f_c) = 0$.

On a qualifying axis:

$$G_{\text{eff},\text{ax}}(f_c) \;=\; \frac{\displaystyle\sum_k |A_k|\;|a_k(f_c)|\;|D_{N_k}(f_c,T_k)|}{\displaystyle\sum_k |a_k(f_c)|\;|D_{N_k}(f_c,T_k)|}$$

This is a spectral-weight–averaged amplitude: events that contribute more
to the spectral peak at $f_c$ dominate the average.  The result is in the
same units as the time-domain gradient amplitude (Hz/m).

Each candidate is then checked against the union set of forbidden bands:

$$f_\text{lo} \;\le\; f_c \;\le\; f_\text{hi} \quad\text{and}\quad G_{\text{eff},\text{ax}}(f_c) \;>\; A_\text{limit}$$

A **violation** is flagged if *any single axis* exceeds the band limit.
Because non-candidate axes have $G_\text{eff} = 0$, they never trigger a
violation.  This ensures that only axes with genuine spectral activity at
$f_c$ participate in the forbidden-band comparison.

## Rotation invariance

The analysis is invariant to both per-block rotation events and global FOV
rotation (oblique prescriptions).  These operations redistribute gradient
amplitude among the Gx, Gy, and Gz axes but do not change the total energy
at any frequency.  Because:

- Forbidden bands carry no axis tag — each band is defined solely by
  $(f_\text{lo}, f_\text{hi}, A_\text{limit})$.
- Every axis is checked against every band: the violation loop iterates
  over all three axes for each (candidate, band) pair.
- Event extraction, spectral evaluation, $G_\text{eff}$, and the tier
  checks use identical logic and thresholds for all axes.

energy cannot migrate to an unchecked axis.  A candidate that violates a
band on any single axis is flagged regardless of which physical gradient
channel carries it.

## Per-subsequence processing

A sequence collection may contain multiple subsequences (e.g. a localiser
followed by a scan).  The analysis iterates over each subsequence
independently:

1. Identify the canonical TR window (or pass window for non-degenerate
   sequences) and enumerate unique TR variants by shot ID.
2. For each unique variant, extract the gradient waveforms, run the
   structural analysis, and check candidates against the forbidden bands.
3. In the safety-check path, the first violation triggers an immediate
   failure with a diagnostic message identifying the subsequence, TR
   variant, axis, frequency, amplitude, and band that was exceeded.

## Computational efficiency

The analysis is designed so that its cost depends on the *complexity* of a
single canonical TR — not on the number of TRs in the sequence.

- **One-time base-definition work.**  The expensive per-definition
  operations — sub-period detection, PWL vertex construction, and
  FFT-based response computation — are performed once per unique gradient
  definition.  Because Pulseq encodes waveforms as reusable definitions,
  a sequence with hundreds of TRs typically contains only a handful of
  distinct gradient shapes.
- **Reuse across blocks.**  Each block in the canonical TR references a
  gradient definition that has already been processed.  Event extraction
  simply records the definition ID, timing offset, and amplitude — no
  waveform resampling or per-block FFT is needed.
- **Analytical spectral evaluation.**  The inner-TR spectrum is evaluated
  by summing closed-form phasor contributions from each event, not by
  constructing and transforming a full time-domain waveform.  Evaluating
  at $N_f$ frequencies with $K$ events costs $O(N_f \cdot K)$.
- **Independence from sequence length.**  Because the method analyses the
  canonical TR analytically rather than simulating the full time series,
  a 10 000-TR scan costs the same as a 10-TR scan with the same TR
  structure.  The only input that scales with sequence length is the
  dense FFT spectrum, which is computed once by the caller and passed in.
