# Safety

```{toctree}
:hidden:
:maxdepth: 1

gradient_slew
pns
mechanical_resonance
```

A Pulseq file cannot be checked against a scanner, because it does not
describe one. Whether a sequence is playable is a statement about *this*
system: its gradient amplifier, its coil geometry, its acoustic response, the
patient in it. So somebody has to run those checks, and where they run
decides what the operator experiences.

Pulserver runs them twice, on purpose, with the same code:

- **At design time**, in Python, so a protocol the operator is editing can be
  refused at the console with a reason and a duration rather than accepted
  and then rejected at the magnet.
- **Before download**, in the interpreter, because the design-time answer was
  computed by a program the scanner did not run, and the scanner is what is
  responsible.

Those are bindings and a link against one C library, not two
implementations. A design-time verdict that the scanner later contradicts
would be worse than no check at all, so there is nothing to keep in sync.

## What is checked

| Check | Needs | Refuses |
|---|---|---|
| {doc}`gradient_slew` | `max_grad`, `max_slew` | an axis or the vector above the limit; a slew above it, including across block boundaries |
| gradient continuity | — | a waveform that starts or ends where its neighbour does not meet it |
| RF consistency | — | a repetition whose RF the safety model cannot bound |
| {doc}`pns` | a coil response model | a stimulation estimate above threshold |
| {doc}`mechanical_resonance` | a forbidden-band table | a sustained drive inside a band the system must not be driven in |

The first three need nothing a scanner does not already know, so they always
run. The last two need site data that no sequence file carries — a coil's
chronaxie and saturation, a magnet's acoustic bands — and are opt-in: supply
the data and the check runs, omit it and it is skipped rather than guessed.

## Everything is checked over the TR

Three of these are not properties of a block. SAR and coil heating are
defined per unit time over a repetition; gradient heating over a duty cycle;
the acoustic response over a periodic drive, which needs a period before it
can be a spectrum at all. That is what the structural TR is for — see
{doc}`../sequence_model/tr_and_segmentation` — and it is why detection is a
safety-critical part of the interpreter rather than a convenience.

Where a quantity varies across repetitions, the check takes the **worst**
one, not the first and not the mean: an RF train whose flip angle ramps has a
worst B1rms somewhere in the middle, and a gradient envelope is evaluated at
the positional maximum across instances.

## The order, and why it matters

The checks run cheapest-first and stop at the first failure: amplitude, then
continuity, then slew, then the opt-in analyses. This is not only about time.
A sequence that violates the slew limit will also, usually, look alarming to
the acoustic analysis — reporting the slew violation is reporting the cause,
while reporting a resonance would be reporting a symptom.

## Running them

```python
ok, message = seq.check_hardware_limits()      # amplitude, slew, continuity
seq.calculate_pns(hardware, tr="worst_case")   # the stimulation estimate
seq.calculate_gradient_spectrum(              # the acoustic drive
    tr="worst_case", bands=bands, resonance_lines=True,
)
```

and from an interpreter, the same three in one call —
{doc}`../../examples/cpp/safety_only`.
