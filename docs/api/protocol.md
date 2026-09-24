# Protocol

The entries of a scanner protocol, and the text blocks that carry them between
the host daemon and the scanner interpreter.

```{eval-rst}
.. currentmodule:: pulserver.protocol
```

A protocol entry has a kind, a current value and the schema the scanner UI
presents it with: a range and increment for a numeric entry, and options for a
dropdown. Times are integer microseconds, the unit of the scanner's time
control variables. A preset is a negative value that the UI shows as a word and
that a scanner sequence maps to a design choice, such as the shortest echo
time. The grammar of the blocks is the one `pulseg_protocol_parse` reads on the
interpreter side.

## Entries

| Object | Description |
| --- | --- |
| {obj}`~pulserver.protocol.Parameter` | One protocol entry: kind, current value and UI schema. |
| {obj}`~pulserver.protocol.Kind` | Type tag of an entry on the wire. |
| {obj}`~pulserver.protocol.InputMode` | Presentation of an entry in the scanner UI: typed in, chosen from a dropdown, or not editable. |
| {obj}`~pulserver.protocol.TEPreset` | Echo-time presets. |
| {obj}`~pulserver.protocol.TRPreset` | Repetition-time presets. |

## Parameter names

The names of the interpreter's parameter table, grouped by the value type the
table declares, and the options of its string-list parameters. A scanner
sequence names its entries with them.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.protocol.UIParam` | The keys of every UI control, and the user-entry keys. |
| {obj}`~pulserver.protocol.FloatKey` | Keys declared as float. |
| {obj}`~pulserver.protocol.IntKey` | Keys declared as integer. |
| {obj}`~pulserver.protocol.BoolKey` | Keys declared as boolean. |
| {obj}`~pulserver.protocol.EnumKey` | Keys declared as string lists. |
| {obj}`~pulserver.protocol.ConfigKey` | Keys the sequence declares to the interpreter rather than shows. |
| {obj}`~pulserver.protocol.SequenceType` | Options of `sequence_type`. |
| {obj}`~pulserver.protocol.ImagingMode` | Options of `imaging_mode`. |
| {obj}`~pulserver.protocol.PreparationType` | Options of `preparation_type`. |
| {obj}`~pulserver.protocol.TriggerType` | Options of `trigger_type`: Pulseq trigger channel names. |

## Prescription

The field-of-view offset the interpreter fills from the scanner's prescription.
The entries close every listing and are not bound to a design argument; the
host applies the offset when it builds the IR.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.protocol.PRESCRIPTION` | Names of the field-of-view offset entries, in mm along the logical readout, phase and slice axes. |
| {obj}`~pulserver.protocol.prescribed_offset` | Field-of-view offset that protocol values carry, in metres. |

## Wire blocks

A listing carries every entry with its schema, as `LIST_PROTOCOL` returns it; a
value block carries values only, as `VALIDATE` and `GENERATE` requests and
replies do.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.protocol.PROTOCOL_BEGIN` | First line of every protocol block. |
| {obj}`~pulserver.protocol.PROTOCOL_END` | Last line of every protocol block. |
| {obj}`~pulserver.protocol.format_listing` | Format a protocol with its schema. |
| {obj}`~pulserver.protocol.parse_listing` | Read a protocol with its schema from a listing block. |
| {obj}`~pulserver.protocol.format_values` | Format a value block. |
| {obj}`~pulserver.protocol.parse_values` | Read a value block against the listing it was edited from. |
| {obj}`~pulserver.protocol.Validation` | Reply to `VALIDATE`: validity, scan time, information line and values. |
| {obj}`~pulserver.protocol.format_validation` | Format a `VALIDATE` reply. |
| {obj}`~pulserver.protocol.parse_validation` | Read a `VALIDATE` reply. |
