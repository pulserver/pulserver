# The design side: a Python service on a compiled core

This is where Pulserver started. A Pulseq file is usually treated as an
artefact: someone runs a script, gets a `.seq`, copies it to the scanner.
That works for a method being developed, and stops working the moment a
sequence is meant to be *used* — because a clinical protocol is not one
sequence, it is a sequence with the parameters an operator picks, at the
console, minutes before the exam.

The alternative is to treat the sequence as a **service**: the program that
builds it stays available, the console asks it for a protocol, and it
answers. {doc}`Nimpulseq <../background/nimpulseqgui>` reached the same
conclusion and pays for it by writing every sequence in a compiled language.
Pulserver keeps the sequence in Python — where the MR community's design,
optimization and simulation tools live — and recovers the speed underneath
it, once, in a compiled sequence core that every sequence shares. No sequence
is ever individually compiled.

## What the console needs, and when

Three questions, on three different clocks:

| Question | When | Cost budget |
|---|---|---|
| What parameters do you have, and what are their limits? | once, when the sequence is selected | anything |
| Is *this* protocol valid, and how long would it take? | on every edit | interactive — tens of ms |
| Give me the file. | on Download | seconds |

Answering the second one honestly is the hard part, and it is why the safety
engine exists at design time rather than only on the scanner. "Valid" cannot
mean "the script ran": it has to mean the gradients fit the system, the slew
fits, the PNS model stays under threshold, the acoustic response avoids the
forbidden bands — the same checks the scanner will run before it downloads,
run early enough that the operator can change a parameter instead of being
refused at the magnet.

## The declaration is part of the sequence

A sequence program does not just build a sequence; it states what it can be
asked for. Parameters are declared with their kind, bounds and defaults, and
the console renders that:

```python
from pulserver import DropdownFloatParam, TypeinIntParam, UIParam, protocol_to_dict


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

The point is that the *sequence* owns its parameter space. A console does not
carry a table of what a GRE needs; it asks. Adding a parameter is a change in
one place, and a protocol saved against one version can be validated against
another rather than silently meaning something else.

## Two ways the console asks

**Headless.** The interpreter calls the sequence program directly and gets a
structured answer back: valid or not, the acquisition time, a message. This
is the path a modern console takes, and the one
{doc}`../../examples/c/interpreter` builds against. Because the answer is
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
written once works either way. See {doc}`../../examples/python/bridge_gui`
for a plugin driven both ways.

## The compiled core: a drop-in `pypulseq`

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

The wrapper stays interoperable in both directions: any event or sequence can
be handed back to upstream PyPulseq objects on demand, which is how plotting
and the rest of the upstream tooling keep working. The
{doc}`performance pages <../performance/index>` measure all of this; the
point here is only that *no individual sequence pays for it* — one compiled
core, shared by every plugin, with the sequences themselves staying plain
Python.

## Why this shapes the rest

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
