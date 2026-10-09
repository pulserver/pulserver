# Protocol resolution

```{admonition} TL;DR
:class: tldr

- Every edit is answered by your plugin's `evaluate`; the scanner shows the value the design achieves.
- The shipped plugins design one repetition and extrapolate the scan time, so a reply takes tens of milliseconds.
- A resolved protocol sent back resolves to itself, and so to the same design.
```

A requested value is not always achievable: an echo time may be shorter than
the readout permits, a bandwidth is realized on the ADC raster, and *Minimum*
has no number until the sequence is designed. So every edit in the scanner UI
is answered by your plugin. Lesson 2 of the {doc}`Course <../examples/course>`
writes such a plugin.

## What pulserver does

:::{container} capabilities

- **Lists the protocol a plugin declares, times in integer microseconds and presets as negative codes.**

  Code: `pulserver design list`, {mod}`pulserver.protocol`. Tests: *every listed line matches the interpreter grammar* (`test_wire.py`); *a time is listed in integer microseconds* (`test_resolution.py`).
- **Completes a request with the initial values of the entries it omits.**

  Code: {meth}`~pulserver.design.SequencePlugin.validate`. Test: *an entry's default is the protocol's initial value in place of the app's* (`test_resolution.py`).
- **Resolves a request through your `evaluate` and replies with the values the design achieves.**

  Code: {meth}`~pulserver.design.SequencePlugin.evaluate`, {class}`~pulserver.design.Evaluation`. Tests: *a minimum request resolves to the designed value* (`test_resolution.py`); *a valid reply carries the protocol duration and note of the evaluation* (`test_sequence_plugin.py`).
- **Refuses an infeasible request with your sequence's error message, and an undeclared entry by name.**

  Code: {meth}`~pulserver.design.SequencePlugin.validate`. Tests: *an infeasible protocol is invalid with the design error as info*; *a request for an undeclared entry is invalid and names it* (`test_resolution.py`); *a pypulseq feasibility assertion is an expected rejection* (`test_sequence_plugin.py`).
- **Resolves a resolved protocol to itself.**

  Code: {class}`~pulserver.design.Protocol`. Tests: *resolving a resolved protocol changes nothing*; *a resolved protocol survives cv storage* (`test_resolution.py`); *a float is carried to six significant digits and a time to a microsecond* (`test_protocol.py`).
- **Converts between wire units and the sequence function's SI arguments in one place.**

  Code: {meth}`~pulserver.design.Protocol.from_wire`, {meth}`~pulserver.design.Protocol.to_wire`. Tests: *a protocol holds the values in the units of the application arguments*; *a protocol converts back to the wire values it was made from* (`test_protocol.py`).
- **Carries the prescription (offset and rotation) through unchanged and binds it to no argument.**

  Code: {mod}`pulserver.protocol`. Tests: *the prescription travels through resolution unchanged*; *a sequence plugin may not bind a prescription entry*; *a request whose rotation is not orthonormal is invalid* (`test_resolution.py`).
- **Adds averages to every plugin, repeating the main sequence but not its prescans.**

  Code: {class}`~pulserver.design.SequencePlugin`. Tests: *every plugin offers its averages unless it declares them*; *the averages repeat the main sequence and not its prescans*; *the scan time counts every average* (`test_sequence_plugin.py`).
- **Replies, when asked, with the RF layout of one TR.**

  Code: {class}`~pulserver.design.RfLayout`. Tests: *a gre layout is one excitation and lists one definition whatever the matrix*; *a validated amplitude times the listed peak is the peak the instance plays*; *the rf layout is sent only when asked* (`test_rf_layout.py`).
- **Ships evaluations that design at most two TRs, however large the prescription.**

  Code: the shipped plugins, `src/pulserver/_zoo/sequences/`. Tests: *a zoo evaluation designs two trs at most however large the prescription*; *an evaluation states the values and the scan time of the design* (`test_zoo_evaluation.py`).

:::

## What the scanner and your plugin do

- **Your plugin decides what `evaluate` reports**: one designed TR, times from block durations, or the request accepted as it is. The default accepts the protocol and builds nothing.
- **The interpreter draws the UI**, converts microseconds to the milliseconds the operator types, and fills the prescription entries.

## How it works

### Resolution

1. Every omitted entry takes its initial value, so a request is a complete prescription.
2. The wire values become the sequence function's SI arguments, a {class}`~pulserver.design.Protocol`, which your plugin evaluates under the design limits ({doc}`../user-guide/running`). A sequence function that cannot realize the prescription raises.
3. The reply carries the achieved protocol in wire values, the scan time, a note and, if asked, the RF layout. The shipped plugins read `TE`, `TR` and `SliceThickness` from the definitions of the sequence they built, and the bandwidth from its dwell time.

A scan repeats one TR, so the shipped plugins design one line, partition,
spoke, interleaf, echo train or inversion shot of one slice, and multiply its
duration by the number of repetitions. The prescription entries
(`fov_offset_*` in mm, `fov_rotation_ij`) are filled by the interpreter,
returned unchanged, and applied by the host when it builds the IR.

### RF layout

An evaluation may also state the RF instances of one TR, each with the
protocol entry its amplitude follows, so that the scanner estimates the RF of
another prescription without a design call. Its size does not depend on the
matrix; {ref}`rf-layout` states its rules.

### Presets and precision

A preset is a negative time shown as a word, such as *Minimum*
({class}`~pulserver.protocol.TEPreset`, {class}`~pulserver.protocol.TRPreset`).
Your plugin maps it to `None` (shortest achievable), a time, or a function of
the limits, and the reply carries the time the design achieved.

Times travel as integer microseconds and other floats at the six significant
digits a float32 parameter keeps, each with the scale between its UI unit and
the SI argument (`1e-3` for a field of view in mm). Because replies carry
exactly what the scanner stores, sending a resolved protocol back gives the
same protocol, and the same design ({doc}`architecture`).

## See it run

* {doc}`../generated/gallery/01-course/02_sequence_plugin` — a plugin written one entry at a time, and its `evaluate`.
* {doc}`../user-guide/scanner-sequences` — writing a scanner sequence.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
* {doc}`../api/protocol` — protocol entries and wire blocks.
* {doc}`/generated/gallery/02-tours/01_protocol_resolution` — bandwidth quantization and minimum echo time, executed.
