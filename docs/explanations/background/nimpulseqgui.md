# nimpulseqgui, and the bridge

A Pulseq sequence is a program, and someone has to run it with the
parameters an operator chose. Two situations, both real:

- The scanner's own interpreter asks for a sequence directly — it knows the
  protocol, it wants a `.seq` back, and no human is in the loop. This is
  **headless** operation, and it is what
  {doc}`../sequence_model/interactive_design` describes.
- The interpreter cannot do that — an older platform, a research console, a
  workflow where the sequence is prepared before the exam. Then someone needs
  a **user interface** to set the parameters and press build.

[nimpulseqgui](https://github.com/nimpulseq/nimpulseqgui) is that interface.
It is a small Nim application that reads a sequence program's declared
parameters, renders them as a property editor, validates what the user
typed, and writes the `.seq`. It is the fallback path, and Pulserver uses it
unmodified — as a Nimble dependency, not a fork.

## What the GUI expects

nimpulseqgui talks to a sequence program through a plain text protocol. A
program declares its parameters as a **preamble** — a `[NimPulseqGUI Protocol]`
block of `key: value` lines — and the GUI turns that into widgets, hands the
edited values back, and asks for a file.

The stock contract is deliberately minimal: the program is an executable, and
validation answers a boolean. That is enough for a GUI to grey out a Build
button, and not enough for a console that wants to tell the operator *why* a
protocol is unplayable and how long it would take.

## The bridge

Pulserver ships two host executables — `pypulseq_host` and `matlab_host` —
that sit between the GUI and a sequence written in Python or MATLAB. A host
embeds the interpreter (CPython through `nimpy`, or the MATLAB runtime),
loads a user plugin at run time, and exposes it in four modes:

| Mode | Who calls it | Answer |
|---|---|---|
| *(default)* | a person, through the GUI | the `.seq` file |
| `--no-gui` | a script | the `.seq` file |
| `--validate-only` | a console, once | JSON: `{valid, duration, info}` |
| `--persistent` | a console, repeatedly | the same, per command, over stdin/stdout |

Two things the bridge adds over the stock path, both of which matter to a
scanner rather than to a person:

**A richer answer.** The headless modes are handled by the bridge itself
rather than through the GUI's `makeSequenceExe`, so a caller gets a
`ValidationResult` — valid or not, the total acquisition time, and a message
— instead of a boolean. A console can put "TA = 5:32" next to the Build
button, and a refusal can say which limit it hit.

**A warm interpreter.** Starting CPython, importing NumPy and importing the
sequence package costs a second or two; a UI that revalidates on every
keystroke cannot pay that each time. `--persistent` keeps one process alive
and answers commands on stdin:

```
→ VALIDATE                          → GENERATE /tmp/scan.seq
→ [NimPulseqGUI Protocol]           → [NimPulseqGUI Protocol]
→ TE: 5.0                           → TE: 5.0
→ [NimPulseqGUI Protocol End]       → [NimPulseqGUI Protocol End]
← VALID 5.32 TA = 5.32 s            ← GENERATED /tmp/scan.seq
```

The loop is **stateless by design**: every command starts from the plugin's
default protocol and applies the block it was given. Persistence buys
interpreter warmth, not session state — so a dropped connection loses
nothing, and two callers cannot interfere.

The plugin contract is equally plain. A Python plugin exposes

```python
def make_sequence(opts, protocol, output_path):
    ...
```

and receives a real `pulserver.pypulseq.Opts` for the hardware and the
operator's parameter values for the protocol, and writes the file itself. No
temporary files, no sequence returned as a string, no GUI preamble written in
headless mode.

See {doc}`../../examples/python/bridge_gui` for a plugin driven both ways —
through the GUI and through the persistent loop.
