# Safety

```{toctree}
:hidden:
:maxdepth: 1

gradient_slew
canonical_tr
pns
mechanical_resonance
```

```{admonition} TL;DR
:class: tip

- A `.seq` file carries the system it was *designed* against, not the system it
  will be *played* on. The scanner's verdict is the one that decides.
- Pulserver runs the same checks the reference toolboxes offer, and turns the
  two design-time *views* — PNS and gradient spectrum — into verdicts.
- What it moves is **when**: the checks are deferred from every `add_block` to
  `write()`, and a file written for the scanner skips them entirely because the
  interpreter runs the same compiled code at predownload.
- Every check is defined over a repetition, so every check needs a
  {doc}`canonical TR <canonical_tr>`.
- SAR and heating stay with the vendor. What Pulserver owns is their input.
```

`system` in a Pulseq file carries the `Opts` the sequence was designed
against — `max_grad`, `max_slew`, the raster times. It does not carry the
system it will be played on. The two coincide when the designer typed in the
numbers of the scanner in front of them, and part when the file travels, when a
site derates its limits, or when a raster is finer in the script than on the
amplifier.

```{figure} ../assets/safety/where_checks_run.png
The four moments a gradient or RF constraint can be tested at, and what runs at
each on the two paths a file can take.
```

## What the reference toolboxes check

Gradient amplitude and slew as each block is added, continuity along with them,
so an offending `add_block` raises where it was written.

The rest is left where it is evaluated anyway. SAR, RF coil heating and
gradient heating are the scanner's to compute at predownload, from coil and
thermal models a sequence file has no access to — no design-time toolbox offers
a public function for them, Pulserver included.

Peripheral nerve stimulation and mechanical resonance *are* available at design
time, as views rather than gates: `calc_pns` (the SAFE model, hence Siemens
hardware) and `calc_gradient_spectrum` draw a picture for you to read.

## Where each check runs

Amplitude, slew, continuity and timing are deferred to `write()` under two
flags, `check_gradients` and `check_timing`. The pass runs over the gradient
library, so it costs one evaluation per distinct waveform however many times
the scan plays it.

**A file written for the scanner runs none of them.** That is the recommended
pipeline for a plugin: `write_sequence(seq, path, offline=False)` takes no
check flags, and the interpreter checks amplitude, slew, continuity and timing
at predownload through the same C safety core the Python bindings call. Design
time and parse time are one implementation invoked from two places, so running
it twice buys nothing — while running it only at the scanner buys a pass saved
on every prescription change and an answer computed against the raster times
and limits the scanner actually has.

**A file written for anywhere else keeps them on.** A bench, a foreign toolbox,
a colleague's directory: nothing downstream will run the checks, so
`write_sequence(seq, path, offline=True)` writes `.seq` text through `write()`
with both flags on and a signature appended.

## Every check runs over a canonical TR

SAR is defined per unit time over a repetition, gradient heating over a duty
cycle, the acoustic response over a periodic drive. So every check needs one
repetition to stand for the scan — and there is more than one way to build it.
{doc}`canonical_tr` covers all of them: the pass criterion each check serves,
how its window is defined, why it bounds the scan, and how it is found without
evaluating every repetition.

## The three checks Pulserver owns

| Check | Needs | Refuses |
|---|---|---|
| {doc}`gradient_slew` | `max_grad`, `max_slew`, the rasters | an axis or the vector above the limit; a slew above it, including across block boundaries; a discontinuity; an event time the hardware cannot address |
| {doc}`pns` | a coil response model | a stimulation estimate above threshold |
| {doc}`mechanical_resonance` | a forbidden-band table | a sustained drive inside a band the system must not be driven in |

The first needs nothing a scanner does not already know, so it always runs. The
other two need site data no sequence file carries — a coil's chronaxie and
saturation, a magnet's acoustic bands — and are opt-in: supply the data and the
check runs, omit it and it is skipped rather than guessed.

## SAR and heating

The vendor's routines own these. What Pulserver supplies is their input, and
those routines are only as right as the repetition they are handed: the
worst-B1rms repetition for the RF chain, a positional-maximum envelope and its
duty cycle for the amplifiers. Which repetition, and why that one, is
{doc}`canonical_tr`.

## The Python surface

```python
ok, message = seq.check_hardware_limits()      # amplitude and slew
ok, message = seq.check_gradient_continuity()  # joins across block boundaries
ok, report = seq.check_timing()                # rasters and dead times
seq.calculate_pns(hardware, tr="worst_case")   # the stimulation estimate
seq.calculate_gradient_spectrum(               # the acoustic drive
    tr="worst_case", bands=bands, resonance_lines=True,
)
```

`write()` runs the first two under `check_gradients=True` and the third under
`check_timing=True`; both default to on. A plugin writing for the scanner asks
for neither. From an interpreter, the two analyses come in one call —
{doc}`../../examples/c/safety_gate`.
