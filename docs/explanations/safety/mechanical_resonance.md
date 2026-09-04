# Mechanical resonance

```{admonition} TL;DR
:class: tip

**Criterion.** Per physical gradient axis, at every frequency inside a
forbidden band, the amplitude of the sinusoid the scan sustains over the
resonance memory $W$ of the gradient coil:

$$A_W(f) = \max_{t_0}\;\frac{2}{W}\Bigl|\int_{t_0}^{t_0+W} g_\text{ax}(t)\,e^{-2\pi i f t}\,dt\Bigr|,$$

compared with the band's tolerance in the same mT/m. $W$ is a property of
the coil, not of the band or the sequence: `pulseg_opts.mech_memory_us`,
20 ms unless the scanner configuration says otherwise.

**Tolerance.** A band that states an amplitude states the plateau of a
readout train; it is converted to a sinusoid through the shape of the train
that drives the band. A band that states none is held to
`SA_ZERO_BAND_SINUSOID_MT_PER_M`, the sustained sinusoid the vendor's own
refusals and allowances bracket.

**What is judged.** The composite waveform on each physical axis: every
gradient the scan plays, `ROTATIONS` applied, the prescription rotation
composed in at predownload. Nothing is told which event is a readout or an
echo train; a spiral, an EPI train and a SPARKLING arm are read by the same
integral.

**Refusal.** Names the band, the axis, the frequency, the amplitude against
its tolerance, and the gradient definitions that carry the reading with
their share of it.
```

A gradient coil sits in a strong static field, so every gradient waveform is
also a Lorentz force on a large, stiff, lightly damped structure. Drive one
of its mechanical resonances and the response is amplified by the mode's
quality factor rather than merely transmitted: acoustic noise, vibration of
the bore, ghosting where the vibration reaches the field. Vendors therefore
publish **forbidden bands**, and a sequence has one question to answer inside
each of them:

> How much oscillating gradient amplitude does this scan sustain at the
> frequencies of the band?

Not how much acoustic energy it emits and not a broadband figure: a
resonance answers to coherent drive at its own frequency, kept up for as long
as the mode remembers.

## What the vendor measures

The vendor checks three product families, each through the parameter that
sets the family's drive frequency, and nothing else:

| family | the parameter it locks out | the frequency it guards |
|---|---|---|
| echo-planar imaging (`epiesp*.dat`, one section per physical axis) | the echo spacing, in ranges per axis with a tolerance in G/cm | the train's fundamental, $f = 1/(2\,\mathrm{ESP})$ |
| FIESTA / bSSFP (`greAcousticLimit.<coil>.dat`) | the repetition time, in ranges | the harmonic of the readout comb, $k/T_R$, that falls in the coil's band |
| multi-echo gradient echo (`greAcousticLimitEsp.dat`) | the echo spacing | the echo period, $1/\mathrm{ESP}$ |

The three tables of a coil carry the same resonances, each written in its
family's own parameter: a locked TR range is the range that puts a harmonic
of the readout comb into the band the EPI table states, and a locked
multi-echo spacing puts the echo period there. Three checks, one resonance,
read at the harmonic order each family happens to drive it at. A bSSFP at a
locked TR sustains that harmonic inside the band at tens of mT/m where it is
the second; at a TR whose *third* harmonic falls in the band it sustains a
third of that and is not locked out. What decides is the amplitude at the
frequency, not the order of the harmonic or the name of the family. The tolerance column of the EPI tables is almost always zero, and a
zero cannot be read literally: every gradient puts *some* drive into every
band.

## The criterion

A mode of linewidth $\Delta f$ rings up and decays in about $1/\Delta f$;
it answers to the drive sustained over that long and forgets what came
before. That memory belongs to the coil, so the check reads every band over
one window $W$. Per physical axis and per frequency,

$$A_W(f) = \max_{t_0}\;\frac{2}{W}\Bigl|\int_{t_0}^{t_0+W} g_\text{ax}(t)\,e^{-2\pi i f t}\,dt\Bigr|,$$

the amplitude of the sinusoid at $f$ that would carry the same Fourier
content over the window, in mT/m. What this reading does on the waveforms a
sequence is made of:

- **A train longer than the memory** — an EPI readout, a bSSFP readout comb —
  reads at its own sustained amplitude at its fundamental and its harmonics,
  whatever the repetition time. A 20 ms memory is shorter than every echo
  train the vendor locks out, so the reading is the train's.
- **A burst shorter than the memory** reads in proportion to the memory it
  fills: a phase-encode blip a few hundred microseconds long, a spoiler, an
  overtone of a ramp-sampled train, all read at a fraction of a mT/m where a
  refused train reads at ten or more.
- **A sweep** — a spiral, a chirp — crosses a band once and reads what it
  sustained while inside it, scaled by how much of the memory that took.
- **A periodic drive** read over the memory is the classical line amplitude
  $\tfrac{2}{T_R}\lvert S(f)\rvert$, up to the events that start inside the
  window and run past its end.
- Below one period of the window, $f < 1/W$, the integral is the gradient
  moment over the window, which no mode answers to, and the reading is zero.

Nothing in this asks what kind of event it is looking at. There is no
echo-train detector, no "equivalent echo spacing" for a spiral, no
acquisition-based clause: the criterion is the integral, on the composite
waveform of each physical axis, and a gradient echo whose readouts are made
bipolar and numerous enough is refused on exactly the arithmetic that
refuses an EPI.

### The tolerance

A band that **states an amplitude** states it in G/cm as the plateau of the
readout train the family plays, so the two sides of the comparison are put in
the same units first. The reading is a sinusoid; a train of lobes repeated
without end has a fundamental whose amplitude is a fixed fraction of the
plateau — $8/\pi^2$ for triangular lobes, $4/\pi$ for square ones, about 0.92
for a ramp-sampled EPI lobe — and that fraction is read off the lobe's own
transform: when a fused train's fundamental lies in the band, the stated
plateau is converted through *that* train's ratio; otherwise through
$8/\pi^2$, the smallest ratio a train the vendor plays can have, which is the
conservative side.

A band that **states zero** is held to `SA_ZERO_BAND_SINUSOID_MT_PER_M`. The
number is an estimate of the harm threshold read off the vendor's own
decisions, and the {doc}`performance page <../performance/mechanical_resonance>`
shows how: read inside the vendor's bands on the axis they drive, what the
vendor locks out — an echo train whose fundamental lands in a band, a FIESTA
at a locked TR on either coil that locks one — reads a few tenths above the
floor, and every family the vendor runs unchecked reads below it. The one
tolerance a table states converts to a larger number in the same units. It
is a policy constant with a provenance, not a physical constant, and it is
set in one place.

### The frame

Vendor tables are stated **per physical axis**: a band read from an EPI
table carries its axis tag from `read_esp_bands` through `bands_to_hz_per_m`
into the safety core, which judges it against that axis alone, and the axes
differ on every coil in the table. The drive on a physical axis is the
composite of everything the scan plays there: the sequence's own `ROTATIONS`
are folded into the amplitudes each axis receives, and the interpreter hands
the check the prescription rotation at predownload — slice 0's host matrix,
the one every segment's `setrotate` carries — composed left of every
`ROTATIONS` matrix. An oblique prescription that carries a readout from x
onto z is judged against z's bands. A block flagged `NOROT` plays in the
logical frame and is judged there. From Python the frame is the design frame
unless `set_prescription_rotation` is given the matrix. Only this check is
judged in the prescribed frame; stimulation and the gradient limits are
frame-free and stay in the logical frame.

## Where the criterion and the product part

The vendor locks a family out by its parameter alone, whatever its
amplitude; the criterion reads amplitude. Two kinds of train the product
refuses read below the floor and pass:

- **A train whose plateau is under the floor.** A low-bandwidth echo train
  sustains a sinusoid smaller than its plateau, and a plateau under the floor
  cannot reach it.
- **A short packet.** A multi-echo gradient echo of a few echoes at a locked
  spacing fills a fraction of the memory and reads well under the floor,
  monopolar or bipolar.

Both are the criterion doing what it says; an acquisition-based clause that
refused them would refuse a spiral of the same plateau too. The
{doc}`performance page <../performance/mechanical_resonance>` lists them with
their readings.

## What a refusal says, and what to do with it

The refusal names the band, the physical axis, the frequency and the reading
against its tolerance, then the gradient definitions behind the loudest
refused reading with their share of it. The reading is linear in the events,
so the shares are exact: a share above one means the named lobe is partly
cancelled by the others.

```text
mech-res exceeded (f=558.00Hz,a=602543.25>553488.00Hz/m,ax=2,ss=0,tr=0,def=1,share=54,def=0,share=46)
```

From Python the same terms are on the object `calculate_gradient_spectrum`
returns under `resonance_lines=True`: `tolerance` per candidate,
`contributors` as `(definition, share)` pairs, `contributor_freq` and
`contributor_axis`; `_get_segment_blocks` maps a definition to the block and
axis that play it. To move a line, change the parameter that sets the lobe
carrying it. A readout train's line is set by its bandwidth and echo spacing,
a repetition comb's by the TR, and every other lobe by the gradient limits it
was built against: build the sequence on a derated system,
`pp.apply_system_derates(system, grad_derate=...)`, and every trapezoid keeps
its area and lengthens while the readout, set by bandwidth and field of view,
stays where it is. On the 3D gradient echo the loudest in-band line belongs
to the spoiler and the prewinder, not to the readout, and halving the system's
gradient amplitude moves it.

```python
derated = pp.apply_system_derates(system, grad_derate=0.5, slew_derate=1.0)
seq = gre3D_sequence.main(system=derated)
lines = seq.calculate_gradient_spectrum(
    plot=False, tr="worst_case", resonance_lines=True, bands=bands
)[4]
lines.ok, lines.contributors
```

## When it runs

The interpreter runs the check at **predownload**, once `make_sequence` has
written the finished `.seq` and the file comes back in. Nothing before that
point has built the gradient waveforms: a console's live feasibility estimate
is a duration computed from module lengths, with nothing to evaluate a band
against. The verdict is an estimate that runs before the scanner's own gate
and its hardware monitor, which stand behind every verdict here.

## One criterion, computed two ways

While the repetitions of a scan play the same set of waveforms — at most
`PULSEG__MAX_SHAPE_GROUPS` (64) distinct sets — the reading is computed from
the {doc}`structural TR <canonical_tr>` alone: its events laid out over as
many repetitions as one window reaches, every position whose amplitude or
rotation varies entering as the largest magnitude it takes, which bounds the
scan from above. A scan that plays more distinct waveform sets than that — a
distinct optimised readout in every repetition — is read from every event it
plays, exactly:

$$A_W(f) = \max_{t_0}\;\frac{2}{W}\Bigl|\sum_{t_m \in [t_0,\,t_0+W)} a_m\,T_m(f)\,e^{-2\pi i f t_m}\Bigr|,$$

the sum over the gradient events that start inside the window, $T_m$ each
one's transform and $t_m$ its start, events longer than the memory read in
pieces of an eighth of it. The two forms are one quantity: a bound that
refuses is settled by the exact reading, a repetition whose varying positions
outlast the memory is read exactly from the start, and the same events
written as $N$ repetitions of $K$ blocks or as one repetition of $NK$ blocks
read the same. How both are evaluated in well under a second, and the
measurements behind every number on this page, are on the
{doc}`performance page <../performance/mechanical_resonance>`.
