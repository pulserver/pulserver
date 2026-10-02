# Protocol resolution

An operator prescribes an acquisition by editing protocol entries in the
scanner UI: echo time, repetition time, field of view, matrix size, receiver
bandwidth. A requested value is not always achievable exactly. An echo time
shorter than the readout permits is infeasible, a receiver bandwidth is
realized on the ADC raster, and a request for the shortest echo time has no
numeric value until the sequence is designed. The value the scanner shows after
an edit is therefore the value the design achieves, not the value requested.

## Resolution

A scanner sequence ({class}`~pulserver.design.ScannerSequence`) maps each
interpreter parameter name to an argument of a pypulseqpp sequence
application's `init_sequence`. Resolving a request proceeds in three steps.

1. Every entry the request omits takes the application's default, so a request
   is always a complete prescription.
2. The wire values are converted to the application's arguments, held as a
   {class}`~pulserver.design.Protocol`, and the application is constructed
   under the scanner limits, capped by the design limits the scanner derates
   for the prescription ({doc}`../user-guide/running`). Construction designs
   the events and their timing against those limits, without playing the scan,
   and raises an error for a prescription it cannot realize.
3. The value each argument took in the design, as the application records it
   ({attr}`~pypulseqpp.sequences.SequenceApp.resolved`), replaces the
   requested one in the protocol, which is converted back to wire values. An
   argument the application does not record keeps its requested value.

The prescription is not a design argument. Every listing ends with its
entries, not editable in the UI, which the interpreter fills from the
scanner's prescription: the field-of-view offset in `fov_offset_x`,
`fov_offset_y` and `fov_offset_z`, in mm, and the rotation from the logical to
the physical axes in the nine `fov_rotation_ij`, identity by default.
Resolution returns them unchanged, refusing a rotation that is not orthonormal;
the host applies the offset to the designed sequence when it builds the IR, and
checks the design in the physical frame of the rotation ({doc}`ir-cache`). A
scanner sequence cannot bind them to an argument.

A request the design refuses is invalid. The reply carries the request
unchanged and the error message the application raised, which the interpreter
shows to the operator. A valid reply carries the resolved values and the scan
time in seconds ({meth}`~pypulseqpp.sequences.SequenceApp.scan_time`).

## Keys and values

The keys of a protocol are members of the key enums of
{mod}`pulserver.protocol`, of which {data}`~pulserver.protocol.ProtocolKey` is
the union. A key is a `str` that equals and hashes as its wire name, so a plain
string indexes a mapping of keys, and a scanner sequence stores a plain string
in `ui` as the member it names. A {class}`~pulserver.design.Protocol` maps each
key to its value in the units of the application's argument: seconds for a
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
to {meth}`~pulserver.design.ScannerSequence.validate` as a mapping.
{func}`~pulserver.design.StringListParam`, which builds the enum from option
strings, is deprecated: declare the enum and use `ChoiceParam`.

## Presets

A preset is a negative value of a time entry that the UI shows as a word, such
as *Minimum* for the echo time ({class}`~pulserver.protocol.TEPreset`,
{class}`~pulserver.protocol.TRPreset`). A scanner sequence maps each preset it
offers to the argument value it requests: `None`, which asks the application
for its shortest achievable time; a time in seconds; or a function of the
scanner limits. A preset is resolved like any other request, so the reply
carries the time the design achieved in place of the preset. A time showing a
preset holds what the preset requests in a protocol, and
{meth}`~pulserver.design.Protocol.preset` returns the preset.

## Units and precision

The interpreter stores protocol values in scanner parameters. Time parameters
hold integer microseconds, so time entries are exchanged in integer
microseconds and converted to seconds for the application, rounding to the
nearest microsecond. Other float entries are held in float32 parameters, whose
round trip preserves six significant decimal digits, so they
are exchanged at that precision. A float entry carries a scale between its UI
unit and the application's SI argument, such as `1e-3` for a field of view
shown in mm and designed in m.

Resolved values are reported at the precision in which they are stored.
Consequently, sending a resolved protocol back unchanged resolves to the same
protocol. This property is what allows a design to be identified by its
resolved protocol ({doc}`designs`): an operator who reopens a protocol and
generates it again obtains the same design.

## Wire format

Protocols are exchanged as text blocks delimited by `[Protocol]` and
`[Protocol End]`. A listing block carries each entry with its schema, one line
per entry:

```text
TE: int|dropdown|8000|1000|80000|10|us|-2
```

that is, kind, input mode, current value, minimum, maximum, increment, unit and
the dropdown options. The line of a string list is its kind, the index of the
chosen option and the options:

```text
imaging_mode: stringlist|1|2d|3d
```

A value block carries `name: value` lines only, a string list as the index of
its chosen option. The grammar is implemented in {mod}`pulserver.protocol` for
the host and in `pulseg_protocol.h` for the interpreter.

## See also

* {doc}`../user-guide/scanner-sequences` — writing a scanner sequence.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
* {doc}`../api/protocol` — protocol entries and wire blocks.
* {doc}`/generated/gallery/01-protocol/01_protocol_resolution` — bandwidth quantization and minimum echo time, executed.
