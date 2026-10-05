# Protocol resolution

```{admonition} TL;DR
:class: tldr

- The value the scanner shows after an edit is the value the design achieves,
  not the value requested: a request is completed with initial values,
  evaluated under the scanner limits and returned as resolved.
- An evaluation may state an RF layout of one TR, from which a scanner
  estimates the RF of a prescription without designing it.
- Times travel as integer microseconds and other floats at six significant
  digits, so a resolved protocol sent back resolves to itself.
```

An operator prescribes an acquisition by editing protocol entries in the
scanner UI: echo time, repetition time, field of view, matrix size, receiver
bandwidth. A requested value is not always achievable exactly. An echo time
shorter than the readout permits is infeasible, a receiver bandwidth is
realized on the ADC raster, and a request for the shortest echo time has no
numeric value until the sequence is designed. The value the scanner shows after
an edit is therefore the value the design achieves, not the value requested.

## Resolution

A scanner sequence ({class}`~pulserver.design.SequencePlugin`) maps each
interpreter parameter name to a keyword argument of its sequence function, a function that
returns sequences. Resolving a request proceeds in three steps.

1. Every entry the request omits takes its initial value, the sequence function's default
   unless the entry declares another, so a request is always a complete
   prescription.
2. The wire values are converted to the sequence function's arguments, held as a
   {class}`~pulserver.design.Protocol`, and the plugin evaluates the protocol
   under the scanner limits, capped by the design limits the scanner derates
   for the prescription ({doc}`../user-guide/running`). An evaluation that
   calls the sequence function designs the events and their timing against those limits
   without playing the scan, and the sequence function raises an error for a prescription it
   cannot realize. The shipped plugins evaluate this way. The default
   evaluation of a sequence function accepts the protocol unchanged and builds
   nothing, and a plugin overrides
   {meth}`~pulserver.design.SequencePlugin.evaluate` to check or complete it.
3. The evaluation returns the protocol the design achieves, converted back to
   wire values, with the scan time, a note and, optionally, the RF layout
   ({ref}`rf-layout`). An evaluation that calls the sequence function reads the values from
   the sequences it designed: the shipped plugins return the values the main
   sequence states for the prescription, the echo time and the repetition time
   in its `TE` and `TR` definitions, the receiver bandwidth as the inverse of the
   dwell time of its ADC event and the slice thickness in its `SliceThickness`
   definition, and an argument they do not read keeps its requested value.

An evaluation answers every edit of an entry, so its duration adds to the
latency of the UI, and a design of the whole scan grows with the matrix and the
slices. A scan repeats one TR, so the shipped plugins design one repetition of
it, one line, partition, spoke, interleaf, echo train or inversion shot of one
slice without dummy repetitions, and extrapolate: the scan time is the duration of a repetition
times the number of repetitions the prescription plays, and slices that share a
TR are dealt into the packets the sequence plays them in, from which the TR and
the scan time follow.

The prescription is not a design argument. Every listing ends with its
entries, not editable in the UI, which the interpreter fills from the
scanner's prescription: the field-of-view offset in `fov_offset_x`,
`fov_offset_y` and `fov_offset_z`, in mm, and the rotation from the logical to
the physical axes in the nine `fov_rotation_ij`, identity by default.
Resolution returns them unchanged, and a rotation that is not orthonormal is
invalid. The host applies the offset to the designed sequence when it builds
the IR, and checks the design in the physical frame of the rotation
({doc}`designs`). A scanner sequence cannot bind them to an argument.

A request the evaluation rejects is invalid, and the reply carries the request
unchanged. A `ValueError` or an `AssertionError`, which pypulseqpp and PyPulseq
raise for an event or a timing they cannot realize, is reported by its
message, which the interpreter shows to the operator. Any other exception also
makes the request invalid, and the message names only its type. A request that
names an entry the protocol does not declare is invalid, and the message names
the entry.

A valid reply carries the resolved values, a note, and the scan time in
seconds the evaluation states. An evaluation that states no scan time, `0.0`, is
valid, and the reply reports the scan time as unknown.

(rf-layout)=
## RF layout

The RF a protocol plays depends on the protocol. An evaluation may state it as
an {class}`~pulserver.design.RfLayout`, from which a scanner estimates the RF of
a prescription without designing it: the RF definitions of a sequence, the RF
instances of one TR in play order, and for each instance the protocol entry its
amplitude follows.

A definition is one RF pulse: a complex waveform at unit peak magnitude, its
timing and its use. Instances of a definition differ in amplitude and in
frequency and phase offset. In the sequence an evaluation builds, the amplitude
of an instance is relative to the first instance of its definition with a
nonzero amplitude, so a refocusing train with a flip angle schedule is one
definition with one instance per pulse, and RF spoiling, which steps the phase
offset, adds no definition ({meth}`pypulseqpp.Sequence.rf_instances`).

A control of an instance is the flip angle, or a float user entry, of the
plugin's protocol. It states that the amplitude of the instance is proportional
to the entry: a scanner that plays a protocol with another value of the entry
multiplies the amplitude by the ratio of that value to the value in the
evaluated protocol, which is positive. An instance without a control is played
at the amplitude it was evaluated at.

A validation states the amplitude of an instance over the `peak_hz` the listing
states for its definition, not over the first instance of the definition in the
evaluation of the validation. A protocol that changes a flip angle changes the
peak of the definition in its own evaluation, and the amplitude stated includes
that change, so that the product of the amplitude and the listed `peak_hz` is the
peak RF amplitude the instance plays.

A layout is one TR: the instances one TR plays, and the TR as its `period`,
over which a scanner averages their RF. Its size does not depend on the matrix.
The shipped plugins state the first TR of the repetition they design, its shot
repeated once per slice of the largest packet, and a balanced steady state the
TR after its half-angle pulse. A definition is named by its number in the
sequence the evaluation builds, in the order of first play. The listing carries
the definitions of the evaluation at the default protocol, and a layout names
the pulses of the listing only where the protocol does not change which pulses
are played first: the length of a train, a flip angle schedule and the matrix do
not, and a protocol that adds a preparation pulse does.

The layout is an estimate for the prescription. An evaluation that states none
states no estimate and is valid, and what the scanner checks before the scan is
the stored design ({doc}`designs`).

## Keys and values

The keys of a protocol are members of the key enums of
{mod}`pulserver.protocol`, of which {data}`~pulserver.protocol.ProtocolKey` is
the union. A key is a `str` that equals and hashes as its wire name, so a plain
string indexes a mapping of keys, and a scanner sequence stores a plain string
in `protocol` as the member it names. A {class}`~pulserver.design.Protocol`
maps each key to its value in the units of the sequence function's argument: seconds for a
time, the argument's unit for a float, a member of the enum for a
{class}`~pulserver.design.ChoiceParam`. The prescription entries bind no
argument and keep the units of the wire.

The wire carries the same protocol as integer microseconds, values in the
entry's UI unit and, for a choice, the index of the chosen option.
{meth}`~pulserver.design.Protocol.from_wire` and
{meth}`~pulserver.design.Protocol.to_wire` are the only conversions between
the two, and the formatting and parsing functions of
{mod}`pulserver.protocol` are the only conversions between keys and values and
the text of a block. A value block that names no option of a choice is refused
when it is parsed, and a request that names none is invalid when it is given
to {meth}`~pulserver.design.SequencePlugin.validate` as a mapping.
{func}`~pulserver.design.StringListParam`, which builds the enum from option
strings, is deprecated: declare the enum and use `ChoiceParam`.

## Presets

A preset is a negative value of a time entry that the UI shows as a word, such
as *Minimum* for the echo time ({class}`~pulserver.protocol.TEPreset`,
{class}`~pulserver.protocol.TRPreset`). A scanner sequence maps each preset it
offers to the argument value it requests: `None`, which asks the sequence function for its
shortest achievable time; a time in seconds; or a function of the
scanner limits. A preset is resolved like any other request: where the
evaluation records the time the design achieved, the reply carries it in place
of the preset. A time showing a preset holds what the preset requests in a
protocol, and {meth}`~pulserver.design.Protocol.preset` returns the preset.

## Units and precision

The interpreter stores protocol values in scanner parameters. Time parameters
hold integer microseconds, so time entries are exchanged in integer
microseconds and converted to seconds for the sequence function, rounding to the nearest
microsecond. Other float entries are held in float32 parameters, whose
round trip preserves six significant decimal digits, so they
are exchanged at that precision. A float entry carries a scale between its UI
unit and the sequence function's SI argument, such as `1e-3` for a field of view
shown in mm and designed in m.

Resolved values are reported at the precision in which they are stored.
Consequently, sending a resolved protocol back unchanged resolves to the same
protocol. This property is what allows a design to be identified by its
resolved protocol ({doc}`designs`): an operator who reopens a protocol and
generates it again obtains the same design.

## See also

* {doc}`../user-guide/scanner-sequences` — writing a scanner sequence.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
* {doc}`../api/protocol` — protocol entries and wire blocks.
* {doc}`/generated/gallery/02-tours/01_protocol_resolution` — bandwidth quantization and minimum echo time, executed.
