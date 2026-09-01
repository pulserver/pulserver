# The design side

```{admonition} TL;DR
:class: tip

- A sequence is a **service**, not an artefact: the program that builds it stays
  available and the console asks it for a protocol.
- The sequence **declares its own parameter space** — kinds, bounds, defaults,
  and `UIParam` slots a console already has fields for. A console carries no
  table of what a GRE needs; it asks.
- Three clocks: *what parameters?* once, *is this protocol feasible and how
  long?* on every edit in tens of ms, *give me the file* in seconds.
  `validate_protocol` answers the middle one from module durations — there is no
  waveform yet to check.
- Two console entry points, one plugin: **headless** (the interpreter calls the
  program directly) and **over the bridge** to Nimpulseqgui, with a warm
  `--persistent` interpreter for revalidation on every keystroke.
- The speed comes from **one compiled core shared by every plugin**. No sequence
  is ever individually compiled, and the Python stays plain Python.
```

A Pulseq file is usually treated as an artefact: someone runs a script, gets a
`.seq`, copies it to the scanner. That works for a method being developed, and
stops working the moment a sequence is meant to be *used* — a clinical protocol
is a sequence with the parameters an operator picks, at the console, minutes
before the exam.

{doc}`Nimpulseq <../background/nimpulseqgui>` reached the same conclusion and
pays for it by writing every sequence in a compiled language. Pulserver keeps
the sequence in Python — where the MR community's design, optimisation and
simulation tools live — and recovers the speed underneath it, once, in a
compiled sequence core that every sequence shares.

## What the console needs, and when

Three questions, on three different clocks:

| Question | When | Cost budget |
|---|---|---|
| What parameters do you have, and what are their limits? | once, when the sequence is selected | anything |
| Is *this* protocol valid, and how long would it take? | on every edit | interactive — tens of ms |
| Give me the file. | on Download | seconds |

`validate_protocol` estimates *feasibility and duration*: whether the parameters
make a physically realisable timing given the modules' own durations, and how
long the resulting scan would take. It builds no waveform, so there is nothing
yet to check the gradients, the PNS model or the acoustic response against —
"valid" in the full sense only means something once a sequence has been built.

So the full verdict is not on the console's clock either. Writing straight to a
scanner, `write_sequence(seq, path, offline=False)` leaves the binary unchecked,
because the interpreter checks timing and gradients at predownload against its
*real* rasters and limits. Writing anywhere else,
`write_sequence(..., offline=True)` runs every check inline — see
{doc}`../safety/index`.

What design time buys is the *same compiled engine* available before a sequence
ever reaches a scanner — to a plugin author, to a CI run, to anyone without a
magnet to ask. {doc}`../performance/sequence_creation` shows where each check
sits in the write path.

## The protocol declaration

A sequence program does not just build a sequence; it states what it can be
asked for. Parameters are declared with their kind, bounds and defaults, and
the console renders that:

```python
from pulserver.design import (
    DropdownFloatParam,
    TypeinIntParam,
    UIParam,
    protocol_to_dict,
)


class Gre2D(SequencePlugin):
    def get_default_protocol(self, system):
        return protocol_to_dict({
            UIParam.TE: DropdownFloatParam(
                value=8.0, min=1.0, max=80.0, incr=0.1, unit="ms",
                options=[TEPreset.MINIMUM, 5.0, 8.0, 15.0, 30.0],
            ),
            UIParam.NY: TypeinIntParam(value=128, min=16, max=512, incr=1),
        })

    def validate_protocol(self, system, protocol): ...
    def make_sequence(self, system, protocol, output_path): ...
```

The parameter *kinds* are what a console needs to render a widget: a type-in
with bounds and an increment, a dropdown with presets, a switch that can be
off, a list. `UIParam` names the slot — `TE`, `TR`, `FLIP`, `FOV`, `NX`,
`NY`, `NSLICES` — so a console that already has a field for TE puts the
sequence's TE in it rather than in a generic property table.

Not everything in a protocol is a control. A `ConfigParam`, keyed by
`ConfigKey`, is something the sequence states about itself for the
interpreter's benefit — the SAR regime it should be costed against, say. It
has no widget, no range, and the console never sends it back; it goes out
once with the default protocol and is read while the scan is being set up.

The *sequence* owns its parameter space. Adding a parameter is a change in one
place, and a protocol saved against one version can be validated against another
rather than silently meaning something else.

## The two console entry points

**Headless.** The interpreter calls the sequence program directly and gets a
structured answer back: valid or not, the acquisition time, a message. This
is the path a modern console takes, and the one
{doc}`../../examples/c/index` builds against. Because the answer is
structured, the console can show the operator the acquisition time next to
the parameter that changed it, and a refusal can name the limit it hit.

**Through Nimpulseqgui, over the bridge.** When the interpreter has no way to
call out — an older platform, or a workflow where the protocol is prepared
before the exam — the parameters are edited in
{doc}`Nimpulseqgui <../background/nimpulseqgui>`'s property editor instead.
Pulserver ships two host executables — `pypulseq_host` and `matlab_host` —
that sit between the GUI and a sequence written in Python or MATLAB. A host
embeds the interpreter, loads the sequence plugin at run time, and exposes it
in four modes:

| Mode | Who calls it | Answer |
|---|---|---|
| *(default)* | a person, through the GUI | the `.seq` file |
| `--no-gui` | a script | the `.seq` file |
| `--validate-only` | a console, once | JSON: `{valid, duration, info}` |
| `--persistent` | a console, repeatedly | the same, per command, over stdin/stdout |

The bridge adds two things the stock GUI contract does not have, both of
which matter to a scanner rather than to a person. It answers with more than
a boolean — valid or not, the total acquisition time, and a message, so a
console can put "TA = 5:32" next to the Build button and a refusal can say
which limit it hit. And it keeps a **warm interpreter**: starting Python and
importing NumPy costs a second or two, which a UI that revalidates on every
keystroke cannot pay each time, so `--persistent` keeps one process alive and
answers commands on stdin. The loop is stateless by design — every command
starts from the plugin's default protocol and applies the edits it was given
— so a dropped connection loses nothing and two callers cannot interfere.

Both paths go through the same declaration and the same plugin, so a sequence
written once works either way.

## The compiled core

The reason a Python sequence can answer on the console's clock is that the
Python is only the *loop*; everything the loop calls is compiled.

`pulserver.pypulseq` is a drop-in replacement for
[PyPulseq](https://github.com/imr-framework/pypulseq): the same functions,
the same signatures, the same `Sequence` surface, so an existing PyPulseq
script runs against it unchanged — and stays an ordinary Python program,
free to call NumPy, PyTorch, a pulse-design toolbox, or anything else in the
ecosystem. Underneath, the `Sequence` is a C++ object. Events are compact
compiled objects whose fields read at native speed; `add_block` registers a
block's events in one compiled call; finding the distinct events in a
million-block scan, and writing the file, are compiled passes. A
protocol-scale MPRAGE — close to a million blocks — designs in about three
seconds, at a steady couple of microseconds per block, from plain Python.

The wrapper stays interoperable in both directions: any event or sequence can be
handed back to upstream PyPulseq objects on demand, which is how plotting and
the rest of the upstream tooling keep working. The
{doc}`performance pages <../performance/index>` measure all of this.

## Consequences for the rest of the system

Once a sequence is a service, three requirements follow that a
write-a-file-and-copy-it workflow never imposes:

- **Design has to be fast.** A million-block MPRAGE cannot take a minute to
  rebuild after a parameter edit — see {doc}`../performance/index`.
- **Checks have to be the real ones.** A design-time "valid" that the scanner
  later contradicts is worse than no check at all, so the safety engine is
  the same compiled code on both sides — Python calls it through bindings,
  the interpreter links it. See {doc}`../safety/index`.
- **The structure has to be derivable.** The console asks for a file and the
  interpreter must play it without a human annotating segments or TRs in
  between — see {doc}`tr_and_segmentation`.
