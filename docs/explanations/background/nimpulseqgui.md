# Nimpulseq and Nimpulseqgui

[Nimpulseq](https://github.com/nimpulseq/nimpulseq) is a Pulseq writer in
[Nim](https://nim-lang.org), a compiled language with Python-like syntax.
[Nimpulseqgui](https://github.com/nimpulseq/nimpulseqgui) sits on top of it:
a small desktop application that reads a sequence program's declared
parameters, renders them as a property editor, validates what the user typed,
and writes the `.seq`.

Together they address two of the
{doc}`three gaps <pulseq>` between a Pulseq file and a product sequence, and
they are worth knowing as the closest published relative of Pulserver's
design side.

## What they improve

**Prescription-time adjustment.** A Nimpulseqgui sequence is not a file
someone generated in advance — it is a *program*, kept next to the scanner,
that declares which parameters it can be asked for and builds the file on
demand. The operator edits TE, matrix, FOV in a property editor and presses
build; the sequence is regenerated for exactly that prescription. This is the
workflow of a product sequence, applied to Pulseq.

**Design throughput.** Nim compiles to native code, so the block loop that
dominates sequence building runs at compiled speed rather than interpreted
speed. A scan that takes minutes to write from MATLAB or Python builds in
seconds.

## How a sequence talks to the GUI

The contract is plain text. A sequence program declares its parameters as a
preamble — a `[NimPulseqGUI Protocol]` block of `key: value` lines — and the
GUI turns that into widgets, hands the edited values back, and asks the
program for a file. Validation answers a boolean, which is enough for the GUI
to grey out its Build button.

## The trade

Each sequence is written in Nim and compiled — its own program, built per
sequence. That buys the speed, and it costs interoperability: a Nim sequence
cannot call into the Python ecosystem the MR community designs in — pulse
design toolboxes, optimization, simulation, PyTorch — and a Python sequence
cannot be driven by the GUI without something in between.

Pulserver keeps the workflow — the sequence as a parameter-declaring program
an operator or a console drives — and moves the sequence itself back into
Python, with the speed recovered by a compiled core underneath rather than by
compiling each sequence. How, including the bridge that lets a Python
sequence be driven by Nimpulseqgui unmodified, is the subject of
{doc}`../sequence_model/design_service`.
