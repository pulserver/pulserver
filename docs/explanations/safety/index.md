# Safety

```{toctree}
:hidden:
:maxdepth: 1

gradient_slew
pns
mechanical_resonance
```

A Pulseq file does describe a system: `system` carries the `Opts` the
sequence was designed against — `max_grad`, `max_slew`, the raster times.
What it does not carry is the system it will be *played* on. The two coincide
when the designer typed in the numbers of the scanner in front of them, and
they part when the file travels, when a site derates its limits, or when a
raster is finer in the script than on the amplifier. So a design-time verdict
is a statement about the declared system and the scanner's verdict is a
statement about the real one, which is why the scanner's is the one that
decides.

## What Pulseq checks, and what it leaves out

The reference toolboxes check gradient amplitude and slew as each block is
added — gradient continuity along with them, since a step across a block
boundary is infinite slew — so an offending `add_block` raises where it was
written.

The rest is left where it is evaluated anyway. SAR, RF coil heating and
gradient heating are the scanner's to compute at predownload, from coil and
thermal models a sequence file has no access to, so no design-time toolbox
offers a public function for them — Pulserver included.

Peripheral nerve stimulation and mechanical resonance *are* available at
design time, as views rather than gates: `calc_pns` (the SAFE model, hence
Siemens hardware) and `calc_gradient_spectrum` draw a picture for you to
read.

## Where the checks run

Pulserver runs the same checks, and turns those two views into verdicts.
What it moves is *when*.

Amplitude, slew and continuity are deferred to `write()` rather than paid at
every `add_block`, and they go together under one flag. The pass runs over
the gradient library, so it costs one evaluation per distinct waveform however
many times the scan plays it, which is what makes a protocol-scale sequence
affordable to check at all.

They can also be switched off, and on the scanner path they are. The
interpreter checks amplitude, slew, continuity and timing at predownload
using the same C safety core the Python bindings call — so design time and
parse time are not two implementations to keep in agreement, they are one
implementation invoked from two places, and the answers cannot differ.
Running it twice therefore buys nothing, while running it only at the scanner
buys two things: a pass saved, and an answer computed against the raster
times and limits the scanner actually has rather than the ones the script
declared.

A file written for anywhere else — a bench, a foreign toolbox, a colleague —
gets every check at design time instead, because nothing downstream will run
them.

## What is checked

| Check | Needs | Refuses |
|---|---|---|
| {doc}`gradient_slew` | `max_grad`, `max_slew` | an axis or the vector above the limit; a slew above it, including across block boundaries, where a discontinuity is also caught as itself |
| {doc}`pns` | a coil response model | a stimulation estimate above threshold |
| {doc}`mechanical_resonance` | a forbidden-band table | a sustained drive inside a band the system must not be driven in |

The first needs nothing a scanner does not already know, so it always runs.
The other two need site data that no sequence file carries — a coil's
chronaxie and saturation, a magnet's acoustic bands — and are opt-in: supply
the data and the check runs, omit it and it is skipped rather than guessed.

## SAR and heating stay with the vendor

The vendor's routines own these, which is why this repository has no section
on them. What Pulserver supplies is their input, and those routines are only
as right as the repetition they are handed: Pulserver hands them the worst
one — a worst-case B1rms for the RF chain, a positional-maximum gradient
envelope and its duty cycle for the amplifiers.

## Everything is checked over the TR

That is the same requirement the checks here have. SAR and coil heating are
defined per unit time over a repetition; gradient heating over a duty cycle;
the acoustic response over a periodic drive, which needs a period before it
can be a spectrum at all. That is what the structural TR is for — see
{doc}`../sequence_model/tr_and_segmentation` — and it is why detection is a
safety-critical part of the interpreter rather than a convenience.

Where a quantity varies across repetitions, the check takes the **worst**
one, not the first and not the mean: an RF train whose flip angle ramps has a
worst B1rms somewhere in the middle, and a gradient envelope is evaluated at
the positional maximum across instances.

## Running them

```python
ok, message = seq.check_hardware_limits()      # amplitude and slew
ok, message = seq.check_gradient_continuity()  # joins across block boundaries
seq.calculate_pns(hardware, tr="worst_case")   # the stimulation estimate
seq.calculate_gradient_spectrum(               # the acoustic drive
    tr="worst_case", bands=bands, resonance_lines=True,
)
```

`write()` runs the first two itself under `check_gradients=True`, and the
timing check under `check_timing=True`. A plugin writing for the scanner asks
for neither: `write_sequence(seq, path, offline=False)` writes the binary form
unchecked and lets predownload decide. From an interpreter, the two analyses
come in one call — {doc}`../../examples/cpp/safety_only`.
