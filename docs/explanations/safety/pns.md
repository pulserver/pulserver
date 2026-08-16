# Peripheral nerve stimulation

Switching a gradient induces an electric field in the patient. Above a
threshold that field stimulates peripheral nerves — a twitch, then
discomfort, then pain. The limit is regulatory, and unlike the gradient
amplitude it is not a property of any single event: it depends on how the
field is being switched *over time*.

## The model

The estimate is a filter and a threshold. The stimulus is the gradient
derivative $dG/dt$; the nerve responds not to its instantaneous value but to
its history, weighted by a kernel that decays over the nerve's **chronaxie**
time. Pulserver implements the Irnich rheobase/chronaxie form,

$$
h(\tau) \;\propto\; \frac{c}{(c + \tau)^2},
$$

with $c$ the chronaxie constant — the form GE uses, as distinct from the SAFE
model upstream PyPulseq implements. The response on each axis is the
convolution $h * dG/dt$; the axes are combined and the result is compared
against the coil's saturation level, expressed as a percentage of threshold.

Three numbers make the model, and all three belong to the *coil*, not to the
sequence: the chronaxie time, the saturation (rheobase) level, and an
effective length that scales the field the coil produces per unit gradient.
No sequence file carries them, which is why the check is opt-in — supply the
model and it runs, omit it and Pulserver does not guess.

```{note}
This estimate is not the regulatory verdict. The scanner's own predownload
check is, and the hardware monitor stands behind that. What Pulserver's
estimate buys is that a sequence which will be refused is refused *while it
is being written*, computed by the same code the interpreter links.
```

## Why it is expensive, and why it is not

The naive evaluation is a convolution over every gradient sample of the whole
scan: a 30-minute protocol at a 10 µs raster is 180 million samples per axis,
convolved with a kernel tens of milliseconds long. That is not a check you
run while an operator edits a parameter.

Two properties make it affordable:

**The scan is periodic.** The stimulus is the same in every repetition except
for the amplitudes, so the response can be built from per-shape templates
computed once and reused, rather than convolving the timeline. This is the
same structural fact everything else in Pulserver leans on — see
{doc}`../sequence_model/tr_and_segmentation`.

**The kernel is short.** It decays, so it is truncated at a stated number of
chronaxie constants, and the convolution becomes an FFT with a plan that is
reused across shapes.

Together these took the evaluation from about 2 s to 57 ms for a Cartesian
protocol. The memoized path is checked against the exact one, sequence by
sequence, in the test suite: `test_pns_memo_matches_exact_*` asserts the peak
agrees to a relative tolerance on GRE, EPI, FSE, MPRAGE and a non-Cartesian
scan — because a fast estimate that disagrees with the slow one is not an
optimization, it is a different check.

A model that does not publish a kernel is never memoized. It may not be a
linear filter at all — the SAFE model is not — so the exact path runs and the
answer is whatever the model says.

## Looking at it

```python
# `hardware` is the coil's own model: chronaxie, saturation, effective
# length -- a pypulseq SAFE-style object, or Pulserver's Irnich form.
ok, percent, *_ = seq.calculate_pns(hardware, tr="worst_case", do_plots=True)
```

`tr=` is the same selector the rest of the analysis takes: the worst-case
envelope, the zero-variable canonical TR, or an actual repetition by index.
Asking about the timeline instead is a different question and a much slower
one; ask it when you are debugging a specific shot, not when you are checking
a protocol.
