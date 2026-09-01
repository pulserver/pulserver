# The canonical TR

```{admonition} TL;DR
:class: tip

- Every windowed check evaluates **one** repetition and applies the verdict to
  the scan. That repetition is the **canonical TR**.
- The window is always the {doc}`structural TR
  <../sequence_model/tr_and_segmentation>`. What differs between checks is what
  is *put in* it, and there are three constructions:
  **positional-maximum envelope**, **per-position spectral bound**, and
  **worst real instance**.
- Which one a check uses follows from its criterion. A peak criterion takes an
  envelope; a magnitude-at-a-frequency criterion takes the spectral bound; an
  average-over-a-window criterion takes a real instance, because an envelope
  would refuse sequences the hardware can play.
- **No check is ever evaluated per repetition.** The envelope and the spectral
  bound are read off the amplitude table; a real instance is chosen by a
  closed-form score — amplitude² times a per-definition integral computed once.
  One check evaluation follows, not $N_\text{TR}$ of them.
```

A scan is $M$ repetitions of one structural TR. Checking all of them is out of
the question — $M$ is thousands, and every windowed check costs at least the
length of the window. So each check builds one repetition, evaluates that, and
the verdict stands for the scan.

The construction has to earn that. Repetitions are not identical: a phase
encode steps, a spoiler's phase advances, a shot is turned by a rotation, a
multishot readout plays a different arm each time. Position by position inside
the TR, three situations arise:

| | what varies at that position | example |
|---|---|---|
| **A** | nothing | excitation, slice select, spoiler, an unrotated readout |
| **B** | the amplitude, or the rotation | a phase encode; a radial spoke turned by a `ROTATIONS` extension |
| **C** | the waveform itself | a multishot readout written out arm by arm |

The three constructions below differ in how they handle B and C.

## Construction 1 — the positional-maximum envelope

At each block position, take the largest amplitude that position reaches over
the repetitions, keeping its sign, and render that position's own shape at it.
The result is a waveform that drives at least as hard as any repetition, at
every sample, on every axis. No repetition looks like it: it is built, not
sampled.

Case C defeats a single envelope — no amplitude makes one spiral arm's shape
cover another's. So the repetitions are first **grouped by the waveforms they
play**: an instance's signature is the sequence of $(g_x, g_y, g_z)$ definition
and shape ids over its positions, instances sharing a signature form a group,
one envelope is built per group, and the worst group is the verdict. A scan
whose repetitions differ only in amplitude has one group and one envelope. A
four-arm spiral written out has four. Past `PULSEG__MAX_SHAPE_GROUPS` (64) the
sweep **fails closed**, and the diagnostic asks for the repeated waveform to be
written once and turned with a `ROTATIONS` extension.

**What it guarantees, and what it does not.** The envelope dominates every
repetition sample by sample, which is exact and is asserted directly:
`tests/ctests/test_acoustic_window.c` renders every repetition of seven shipped
families and compares. It does *not* follow that the envelope's **peak** bounds
every repetition's peak. PNS response and sound pressure are both maxima over
time of a signed sum, and a pointwise-larger drive can produce a smaller peak
where terms that reinforced now cancel. Those verdicts are held against every
repetition by test rather than by proof — see the tables on the
{doc}`PNS <../performance/pns>` and {doc}`acoustic <../performance/mechanical_resonance>`
pages, where the window's peak lands on the worst repetition's to seven digits.

Used by: amplitude, slew and continuity; peripheral nerve stimulation; sound
pressure; and the `peakB1` envelope of the RF check.

## Construction 2 — the per-position spectral bound

Mechanical resonance reads its verdict off a magnitude at one frequency, and
that changes what can be bounded. Positions of kind A enter a **coherent
complex sum** — value and phase, which is where a comb gets its sharpness.
Positions of kind B or C contribute the largest magnitude over the
$(\text{waveform}, \text{amplitude}, \text{rotation})$ tuples they really take,
which needs no envelope at all because a spectrum at one frequency is a single
number.

That construction is a **proven** ceiling, not a verified one. Writing $C$ for
the coherent sum and $b$ for the summed per-position bounds,

$$\Bigl|\sum_m S_m(f)\,e^{-2\pi i f m T}\Bigr| \le \sum_m |S_m(f)|
  \le M \max_m |S_m(f)| \le M\bigl(|C(f)| + b(f)\bigr),$$

so after dividing by $M T_\text{TR}$ the scan's equivalent sustained amplitude
is at most the window's, at **every** frequency and not only at the harmonics.
A sum of magnitudes bounds a magnitude of sums; the same inequality does not
survive an inverse transform, which is why construction 1 cannot promise the
same thing for a peak.

Used by: mechanical resonance.

## Construction 3 — the worst real instance

SAR, RF amplifier duty and gradient heating integrate power over a window. The
state that stresses them is a repetition the scan really plays — hand them an
envelope and they refuse sequences the hardware can play, because no such
repetition occurs. So the canonical TR here is **selected**, not built, and
every block in it carries its own amplitudes and its own rotation.

Selection is a ranking, and the ranking is closed-form. Both scores are an
amplitude table read against an integral computed once per event definition:

| | score for instance $u$ | the per-definition integral |
|---|---|---|
| RF (SAR, `minseqrfamp`) | $\sum_p A_{u,p}^2\, P_p$ | $P_p=\int \lvert h_p^\text{norm}(t)\rvert^2 dt$ of the unit-peak envelope |
| gradient heating (`minseq`) | $\sum_b \sum_\text{axes} A_{u,b,a}^2\, E_{b,a}$ | $E=\int \lvert g^\text{norm}(t)\rvert^2 dt$ of the gradient shape |

The TR duration is common to every instance, so it drops out of the ranking and
B1rms is ordered by the sum alone. Both integrals are computed at deduplication
time, once per distinct definition. Ranking is then $O(N_\text{TR} \times
\text{positions})$ multiply-adds over a table that is already in memory, and no
waveform is materialised and no vendor routine is called until the winner is
known. Gradient heating ranks per *segment* rather than per TR, and assembles
the winners into one timeline; pure-delay positions take their shortest
instance instead, so a variable TI shrinks to the shortest TI.

## Which check takes which

| Check | Construction | Criterion it serves |
|---|---|---|
| {doc}`amplitude, slew, continuity <gradient_slew>` | 1 — envelope | per-sample limit |
| {doc}`peripheral nerve stimulation <pns>` | 1 — envelope | peak of the combined response |
| sound pressure | 1 — envelope | peak and A-weighted average pressure |
| {doc}`mechanical resonance <mechanical_resonance>` | 2 — spectral bound | equivalent sustained amplitude in a band |
| SAR, RF amplifier duty | 3 — worst instance | power averaged over a window |
| `peakB1` | 1 — envelope | instantaneous amplitude |
| gradient heating | 3 — worst instance, per segment | energy over a duty cycle |

The RF check builds both: `rfPulseInfo` from the worst-B1rms instance for the
averaged limits, `peakPulseInfo` from the envelope for the instantaneous one.
They answer different questions and neither array can do the other's job.

## What the envelope does not carry

Both constructed windows are rendered in the sequence's **logical** frame. A
`ROTATIONS` extension and the prescribed FOV rotation redistribute a fixed
vector of drive among the three physical axes.

For a forbidden band that costs nothing — a band carries no axis tag, and a
rotation cannot move energy to a different frequency. It costs nothing for the
Irnich nerve model either, which applies one kernel to every axis and combines
by root-sum-square, so it commutes with a rotation. It does cost something for
a model with per-axis coefficients (SAFE) and for sound pressure, whose coil
transfer function is tabulated per axis: an oblique prescription can move drive
from a quieter axis onto a louder one. The prescription matrix is not part of
the sequence — the PSD programs it at `startseq` — so no evaluation at this
layer accounts for it, and the hardware monitor is what stands behind that.
