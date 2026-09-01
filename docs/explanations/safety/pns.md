# Peripheral nerve stimulation

```{admonition} TL;DR
:class: tip

**Criterion.** The peak, over the window, of the axis-combined nerve response
to $\mathrm{d}G/\mathrm{d}t$, as a percentage of the model's own threshold.
100 % is the limit; 80 % is the margin a console leaves.

**Window.** The {doc}`positional-maximum envelope <canonical_tr>` over the
structural TR, one per group of repetitions playing the same waveforms, worst
group wins. Played back to back with itself, so the response carries the
previous repetition's history.

**Cost.** No repetition is ever evaluated. The envelope is read off the
amplitude table; inside it, each distinct gradient shape is convolved with the
kernel once and every occurrence is a scaled, shifted add.

**Model.** Irnich rheobase/chronaxie (GE) or SAFE (Siemens `.asc`), chosen by
the `hardware` argument. Coefficients belong to the gradient coil, so no
sequence file carries them and the check is opt-in.
```

A changing magnetic field induces an electric field in tissue. Switch a
gradient fast enough and the induced field depolarises peripheral nerve
membranes, which the subject feels — a tap, a twitch, at worst pain. Unlike
{doc}`mechanical resonance <mechanical_resonance>`, this is a limit on the
subject, and it is regulatory rather than advisory.

What stimulates is not the gradient amplitude but its rate of change, and not
its instantaneous value either: a nerve integrates, so a short intense slew and
a longer gentler one can be equally stimulating. The threshold depends on pulse
*duration*, and the classical description of that dependence is the
strength–duration curve.

## The two nerve models

Both are supported on equal footing, and which one runs is decided by the
hardware description you pass — the model and its coefficients belong to the
gradient coil, not to the sequence.

**Irnich rheobase/chronaxie**, the form GE systems use. The threshold for a
rectangular slew pulse of duration $\tau$ rises as the pulse shortens:

$$S_\text{thr}(\tau) = \frac{S_\text{min}}{1 - \bigl(\tfrac{c}{c+\tau}\bigr)}
\quad\text{with}\quad S_\text{min} = \frac{\text{rheobase}}{\alpha},$$

with $c$ the chronaxie time, the rheobase the asymptotic threshold for an
infinitely long pulse, and $\alpha$ a coil attenuation factor. As a filter this
is a convolution of the slew waveform with the kernel $c/(c+\tau)^2$, each tap
integrated across its own sample interval,

$$k[i] = \frac{c\,\Delta t}{S_\text{min}\,(c + i\,\Delta t)\,(c + (i{+}1)\,\Delta t)},$$

whose $1/\tau^2$ tail is truncated after 20 chronaxie constants. The taps are
bin integrals because the slew samples they filter are bin integrals — the
gradient differenced on its raster — and matching the two is what makes the
reported response a property of the waveform rather than of the raster it is
evaluated on.

**SAFE** (Hebrank and Gebhardt), the form Siemens `.asc` files describe. Each
axis passes through a cascade of nerve-response stages with coil-specific
coefficients, and the combination is nonlinear — so it publishes no kernel.
Pulserver's compiled engine holds its answer equal, sequence by sequence in the
test suite, to the open Szczepankiewicz–Witzel implementation upstream PyPulseq
ships (8×10⁻⁷ relative).

```python
irnich = {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333}
ok, norm, components, t = seq.calculate_pns(irnich, tr="worst_case")         # Irnich
ok, norm, components, t = seq.calculate_pns("scanner.asc", tr="worst_case")  # SAFE
```

The rheobase is in T/m/s, the unit of the slew waveform; the response is
normalised against `rheobase / alpha`. Both are drawn the same way — upstream
PyPulseq's `safe_plot`, with the 100 % threshold and the 80 % margin marked —
so two figures can be read against each other.

## The check

Build the gradient waveform on the gradient raster, differentiate it, run the
nerve model on each axis, combine by root-sum-square, and take the peak.

There is no shortcut around evaluating a whole window. The verdict is the peak
of a running response *with memory*, so two adjacent events each at 60 % of
threshold may combine to 90 % or to 30 % depending entirely on their relative
timing and sign. And because a TR plays back to back with copies of itself, the
response at the start of one depends on the end of the one before: the
evaluation carries that history, so the peak found inside the window is the
peak of the steady-state scan.

The window itself is {doc}`construction 1 <canonical_tr>` — the
positional-maximum envelope, one per shape group. How it is made interactive,
and the tests holding the fast path equal to the plain one, are on the
{doc}`performance page <../performance/pns>`.

```{note}
This estimate is not the regulatory verdict. The scanner's predownload check
is, and the hardware monitor stands behind that. What the design-time estimate
buys is that a sequence which will be refused is refused *while it is being
written* — by the same compiled code the interpreter links, so the two answers
cannot disagree by reimplementation.
```

## The two models on the corpus

Each figure pairs a sequence's canonical TR with its stimulation trace. The
SAFE figure uses upstream PyPulseq's own example coefficients — explicitly not
a real scanner's.

Every trace runs past the end of its TR. That tail is the wrapped history: the
opening events replayed with the previous repetition's memory behind them.
Comparing it against the same events at $t=0$, where the model starts cold,
shows how much of the response comes from the repetition rather than from the
events. How far it runs is the model's own `required_padding()`, which is why
the SAFE figure extends further than the Irnich one.

**GRE**, one isolated readout gradient per TR.

```{figure} ../assets/representative_tr/gre_2d_tr.png
The GRE canonical TR: one slice-select and rewinder, one prewinder pair, one
readout, and 22 ms of dead time.
```

```{figure} ../assets/pns_safety/gre_2d_pns.png
The same TR under Irnich. Each gradient edge raises a spike that decays before
the next arrives; the verdict is the 3.3 ms slice-select rewinder.
```

```{figure} ../assets/pns_safety/gre_2d_pns_safe.png
The same TR under SAFE. Per-axis coefficients move the verdict onto the 6.8 ms
prewinder pair.
```

The shape is the point, not the number. The two models see the same events at
the same instants and disagree on which is worst. Irnich reads 122 % on the
slice-select rewinder — one chronaxie and one rheobase serve all three axes, so
the verdict follows whichever axis slews hardest. SAFE carries a coefficient
set per axis, and in this example table Y is twice as sensitive as X; the
prewinder pair comes out at 57 % and 56 % *together*, and their combination
sets the 80 %. Which is right for a given scanner is a question about that
scanner's coil, so read the SAFE figure as what the model's output looks like,
not as a margin.

**EPI**, a long blipped echo train, asks the question GRE's well-separated
events never do: whether the responses pile up.

```{figure} ../assets/representative_tr/epi_2d_tr.png
The EPI canonical TR: a blipped echo train reversing the readout gradient every
400 µs.
```

```{figure} ../assets/pns_safety/epi_2d_pns.png
The train under Irnich: one tooth per reversal, none of them stacking.
```

They do not stack: the echo spacing is just longer than the 360 µs chronaxie,
so each response decays to a couple of per cent before the next edge arrives,
and the verdict — 124 % — is still the slice-select rewinder at 8.7 ms.
Shorten the echo spacing, or move to a coil with a longer chronaxie, and the
teeth land on each other's tails; at that point the train sets the verdict and
nothing about any single reversal predicts it. Whether a repetitive train
accumulates is a property of the interval between its edges against the nerve's
own time constant, and neither is visible in the events alone.
