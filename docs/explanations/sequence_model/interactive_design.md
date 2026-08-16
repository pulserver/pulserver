# Interactive design: the sequence as a service

This is where Pulserver started. A Pulseq file is usually treated as an
artefact: someone runs a script, gets a `.seq`, copies it to the scanner. That
works for a method being developed, and stops working the moment a sequence is
meant to be *used* — because a clinical protocol is not one sequence, it is a
sequence with the parameters an operator picks, at the console, minutes before
the exam.

The alternative is to treat the sequence as a **service**: the program that
builds it stays available, the console asks it for a protocol, and it answers.
Everything else in Pulserver follows from taking that seriously.

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

## Two ways the console asks

**Headless.** The interpreter calls the sequence program directly and gets
JSON back: valid or not, the acquisition time, a message. This is the path a
modern console takes, and the one {doc}`../../examples/c/interpreter` builds
against. Because the answer is structured, the console can show the operator
the acquisition time next to the parameter that changed it, and a refusal can
name the limit it hit.

**Through a GUI.** When the interpreter has no way to call out — an older
platform, or a workflow where the protocol is prepared before the exam — the
parameters are edited in a property editor and the file is written from
there. Pulserver uses [nimpulseqgui](../background/nimpulseqgui.md) for this,
unmodified, and improves the path underneath it with a host that embeds
Python or MATLAB and answers with more than a boolean.

Both go through the same declaration and the same plugin, so a sequence
written once works either way.

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

The parameter *kinds* are what a console needs to render a widget: a
type-in with bounds and an increment, a dropdown with presets, a switch that
can be off, a list. `UIParam` names the slot — `TE`, `TR`, `FLIP`, `FOV`,
`NX`, `NY`, `NSLICES` — so a console that already has a field for TE puts the
sequence's TE in it rather than in a generic property table.

The point is that the *sequence* owns its parameter space. A console does not
carry a table of what a GRE needs; it asks. Adding a parameter is a change in
one place, and a protocol saved against one version can be validated against
another rather than silently meaning something else. See
{doc}`protocol_ui` for the full vocabulary.

## Why this shapes the rest

Once a sequence is a service, three requirements follow that a
write-a-file-and-copy-it workflow never imposes, and they are why Pulserver
looks the way it does:

- **Design has to be fast.** A million-block MPRAGE cannot take a minute to
  rebuild after a parameter edit, so the build path is optimized rather than
  merely correct — see {doc}`../performance/index`.
- **Checks have to be the real ones.** A design-time "valid" that the scanner
  later contradicts is worse than no check at all, so the safety engine is the
  same compiled code on both sides — Python calls it through bindings, the
  interpreter links it. See {doc}`../safety/index`.
- **The structure has to be derivable.** The console asks for a file and
  the interpreter must play it without a human annotating segments or TRs
  in between, which is what {doc}`tr_and_segmentation` is for.
