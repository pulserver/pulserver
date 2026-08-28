# Peripheral nerve stimulation

A changing magnetic field induces an electric field in tissue. Switch a
gradient fast enough and the induced field depolarises peripheral nerve
membranes, which the subject feels — a tap, a twitch, at worst pain. Unlike
{doc}`mechanical resonance <mechanical_resonance>`, this is a limit on the
subject, not the hardware, and it is regulatory rather than advisory.

The quantity that stimulates is not the gradient amplitude but its **rate of
change**, $\mathrm{d}G/\mathrm{d}t$ — and not its instantaneous value either.
A nerve integrates: a short, intense slew and a longer, gentler one can be
equally stimulating. The threshold therefore depends on pulse *duration*, and
the classical description of that dependence is the strength–duration curve.

## The two nerve models

Two model families describe that curve in practice, and Pulserver supports
both on equal footing — which one runs is decided by the hardware description
you pass, because the model and its coefficients belong to the *gradient
coil*, not to the sequence.

**Irnich rheobase/chronaxie**, the form GE systems use. The stimulation
threshold for a rectangular slew pulse of duration $\tau$ rises as the pulse
shortens:

$$S_\text{thr}(\tau) = \frac{S_\text{min}}{1 - \bigl(\tfrac{c}{c+\tau}\bigr)}
\quad\text{with}\quad S_\text{min} = \frac{\text{rheobase}}{\alpha},$$

where $c$ is the **chronaxie** time, the **rheobase** is the asymptotic
threshold for an infinitely long pulse, and $\alpha$ is a coil attenuation
factor. Realised as a filter, this is a convolution of the slew waveform with
the kernel

$$k[i] = \frac{\Delta t}{S_\text{min}}\cdot\frac{c}{(c + i\,\Delta t)^2},$$

whose tail decays as $1/\tau^2$ and is truncated after 20 chronaxie
constants. The result is a fraction of threshold, per axis, at every instant.

**SAFE** (Hebrank and Gebhardt), the form Siemens `.asc` hardware files
describe. It is not a single convolution: each axis passes through a small
cascade of nerve-response stages with coil-specific coefficients, and the
combination is nonlinear. Pulserver's compiled engine implements it alongside
Irnich, and its answer is held equal — sequence by sequence, in the test
suite — to the open Szczepankiewicz–Witzel implementation that upstream
PyPulseq ships.

Both are available from
{meth}`pulserver.pypulseq.Sequence.calculate_pns`, selected by the shape of
the `hardware` argument: a mapping carrying `chronaxie` and `rheobase`
selects Irnich, a Siemens `.asc` path or a per-axis description selects SAFE.

```python
irnich = {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333}
ok, norm, components, t = seq.calculate_pns(irnich, tr="worst_case")         # Irnich
ok, norm, components, t = seq.calculate_pns("scanner.asc", tr="worst_case")  # SAFE
```

The rheobase is in T/m/s, the unit of the slew waveform the model is handed;
the response is normalised against `rheobase / alpha`. Whichever model runs,
the picture is drawn the same way — upstream PyPulseq's `safe_plot`, with the
100 % threshold and the 80 % margin marked over it — so two figures can be
read against each other without first working out which is scaled how.

The hardware description is always passed in, never read off the system
limits: a sequence author's `Opts` describe the gradients they are designing
for, not the coil's nerve-response coefficients, and conflating the two would
let a plot silently use the wrong model. Because the coefficients are the
coil's, no sequence file carries them — which is why the check is opt-in:
supply the model and it runs, omit it and Pulserver does not guess.

## The check

The evaluation itself is the direct one. Build the gradient waveform on the
gradient raster, differentiate it to get the slew, run the nerve model on
each axis, combine the axes, and compare the peak of the combined response
against threshold. There is no shortcut around evaluating the whole window:
the verdict is the *peak of a running response with memory*, and two adjacent
events each at 60 % of threshold may combine to 90 % or to 30 % depending
entirely on their relative timing and sign. The instant of the peak is a
property of the whole window, so the whole window is evaluated.

The window is the {doc}`structural TR
<../sequence_model/tr_and_segmentation>` — the repeating unit the scan is
made of — taken at its worst case: where an amplitude varies across
repetitions, the check sees the largest one, so the answer bounds every
repetition of the real scan. Because a TR plays back to back with copies of
itself, the response at the start of one TR depends on the end of the
previous one; the evaluation accounts for that history, so the peak found
inside one TR is the peak of the steady-state scan, boundaries included.
How this — and the caching that makes the evaluation interactive — is
implemented, and the tests that hold the fast path exactly equal to the
plain one, are on the {doc}`performance page <../performance/index>`.

```{note}
This estimate is not the regulatory verdict. The scanner's own predownload
check is, and the hardware monitor stands behind that. What the design-time
estimate buys is that a sequence which will be refused is refused *while it
is being written* — computed by the same compiled code the interpreter links,
so the two answers cannot disagree by reimplementation.
```

## The two models on the corpus

The corpus figures below pair each sequence's worst-case TR with its Irnich
stimulation trace; the SAFE figure uses upstream PyPulseq's own example
coefficients (explicitly *not* a real scanner's) on the same GRE fixture.

Each trace runs past the end of its TR. That tail is the wrapped history the
model needed — the opening events of the TR replayed with the previous
repetition's memory behind them — and comparing it against the same events at
$t = 0$, where the model starts cold, shows directly how much of the response
comes from the repetition rather than from the events themselves. How far it
runs is the model's own answer to `required_padding()`, which is why the SAFE
figure extends further than the Irnich one over the same sequence.

**GRE**, one isolated readout gradient per TR. Every gradient edge raises a
spike of its own and each has time to decay before the next: the slice-select
ramp at $t = 0$, its rewinder at 3.3 ms, then the prewinder pair and the
readout ramps clustered around 6–7 ms, and nothing at all across the 22 ms of
dead time that fills out the TR.

```{figure} ../assets/representative_tr/gre_2d_tr.png
The GRE worst-case TR: one slice-select and rewinder, one prewinder pair, one
readout, and 22 ms of dead time.
```

```{figure} ../assets/pns_safety/gre_2d_pns.png
The same TR under the Irnich model. Each gradient edge raises a spike that
decays before the next arrives; the verdict is the 3.3 ms slice-select
rewinder.
```

```{figure} ../assets/pns_safety/gre_2d_pns_safe.png
The same TR under SAFE, with upstream PyPulseq's example coefficients — not a
real scanner's. Per-axis coefficients move the verdict onto the 6.8 ms
prewinder pair.
```

The shape is the point, not the number. The two models see the same events at
the same instants, but they do not agree on which one is worst. Irnich reads
122 % on the slice-select rewinder at 3.3 ms — a single large
excursion on Z, well clear of anything else — because one chronaxie and one
rheobase serve all three axes, so the verdict follows whichever axis slews
hardest. SAFE carries a separate coefficient set per axis, and in this
example table Y is twice as sensitive as X; the prewinder pair at 6.8 ms
therefore comes out at 57 % and 56 % *together*, and their combination, not
the Z rewinder, is what sets the 80 %. Which of the two is right for a given
scanner is a question about that scanner's gradient coil, so read the SAFE
figure as "this is what the model's output looks like", not as a statement
about any real margin; that would need that scanner's own `.asc` file passed
as `hardware`.

**EPI**, a long blipped echo train, is the sharpest contrast available — a
readout gradient reversed dozens of times a few hundred microseconds apart
asks a question GRE's well-separated events never do: whether the responses
pile up.

```{figure} ../assets/representative_tr/epi_2d_tr.png
The EPI worst-case TR: a blipped echo train reversing the readout gradient
every 400 µs.
```

```{figure} ../assets/pns_safety/epi_2d_pns.png
The train under Irnich: one tooth per reversal, none of them stacking,
and the verdict still set by the slice-select rewinder at 8.7 ms.
```

The train shows up as a run of near-identical ~59 % teeth, one per reversal,
400 µs apart. They do not stack: the echo spacing is just longer than the
360 µs chronaxie, so each response has decayed to a couple of per cent before
the next edge arrives, and the verdict — 124 % — is still set by the
slice-select rewinder at 8.7 ms, exactly as in GRE. Shorten the echo spacing,
or move to a coil with a longer chronaxie, and those teeth start landing on
each other's tails instead, at which point the train sets the verdict and
nothing about any single reversal predicts it. That is why the check
evaluates a whole TR with its history rather than scoring events one at a
time: whether a repetitive train accumulates is a property of the interval
between its edges against the nerve's own time constant, and neither is
visible in the events alone.
