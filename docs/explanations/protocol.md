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
2. The UI values are converted to the application's arguments, and the
   application is constructed under the scanner limits. Construction designs
   the sequence: it solves the gradient waveforms and timing against the
   system limits and raises an error for a prescription it cannot realize.
3. The value each argument took in the constructed application is read back
   and converted to UI units ({meth}`~pulserver.design.ScannerSequence.resolved`).

The field-of-view offset is not a design argument. Every listing ends with the
entries `fov_offset_x`, `fov_offset_y` and `fov_offset_z`, in mm and not
editable in the UI, which the interpreter fills from the scanner's
prescription; resolution returns them unchanged, and the host applies the
offset to the designed sequence when it builds the IR ({doc}`ir-cache`). A
scanner sequence cannot bind them to an argument.

A request the design refuses is invalid. The reply carries the request
unchanged and the error message the application raised, which the interpreter
shows to the operator. A valid reply carries the resolved values and the scan
time in seconds.

## Presets

A preset is a negative value of a time entry that the UI shows as a word, such
as *Minimum* for the echo time ({class}`~pulserver.protocol.TEPreset`,
{class}`~pulserver.protocol.TRPreset`). A scanner sequence maps each preset it
offers to the argument value it requests: `None`, which asks the application
for its shortest achievable time; a time in seconds; or a function of the
scanner limits. A preset is resolved like any other request, so the reply
carries the time the design achieved in place of the preset.

## Units and precision

The interpreter stores protocol values in scanner control variables. Time
control variables hold integer microseconds, so time entries are exchanged in
integer microseconds and converted to seconds for the application, rounding to
the nearest microsecond. Other float entries are held in float32 control
variables, whose round trip preserves six significant decimal digits, so they
are exchanged at that precision. A float entry carries a scale between its UI
unit and the application's SI argument, such as `1e-3` for a field of view
shown in mm and designed in m.

Resolved values are reported at the precision in which they are stored.
Consequently, sending a resolved protocol back unchanged resolves to the same
protocol. This property is what allows a design to be identified by its
resolved protocol ({doc}`sessions`): an operator who reopens a protocol and
generates it again obtains the same revision.

## Wire format

Protocols are exchanged as text blocks delimited by
`[NimPulseqGUI Protocol]` and `[NimPulseqGUI Protocol End]`. A listing block
carries each entry with its schema, one line per entry:

```text
TE: int|dropdown|8000|1000|80000|10|us|-2
```

that is, kind, input mode, current value, minimum, maximum, increment, unit and
the dropdown options. A value block carries `name: value` lines only. The
grammar is implemented in {mod}`pulserver.protocol` for the host and in
`pulseg_protocol.h` for the interpreter.

## See also

* {doc}`../user-guide/scanner-sequences` — writing a scanner sequence.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
* {doc}`../api/protocol` — protocol entries and wire blocks.
* {doc}`/generated/gallery/01-protocol/01_protocol_resolution` — bandwidth quantization and minimum echo time, executed.
