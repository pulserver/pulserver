# Protocol

The entries of a scanner protocol, and the text blocks that carry them between
the design calls and the scanner interpreter.

```{eval-rst}
.. currentmodule:: pulserver.protocol
```

A protocol entry has a kind, a current value and the schema the scanner UI
presents it with: a range and increment for a numeric entry, and options for a
dropdown. Times are integer microseconds, the unit of the scanner's time
parameters. A preset is a negative value that the UI shows as a word and
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
sequence names its entries with them. Each name is a member of a `StrEnum`, so
a key is a `str` that equals and hashes as its wire name; the keys of a
protocol stay members until a wire block formats them.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.protocol.ProtocolKey` | A key of the protocol: a member of any of the key enums below. |
| {obj}`~pulserver.protocol.UIParam` | Collects the keys of every UI control, and returns the user-entry keys. |
| {obj}`~pulserver.protocol.FloatKey` | Keys declared as float. |
| {obj}`~pulserver.protocol.IntKey` | Keys declared as integer. |
| {obj}`~pulserver.protocol.BoolKey` | Keys declared as boolean. |
| {obj}`~pulserver.protocol.EnumKey` | Keys declared as string lists. |
| {obj}`~pulserver.protocol.ConfigKey` | Keys the sequence declares to the interpreter rather than shows. |
| {obj}`~pulserver.protocol.UserKey` | Keys of the float user entries, `user0_value` to `user18_value`. |
| {obj}`~pulserver.protocol.UserNameKey` | Keys of the descriptions naming the user entries, `user0_name` to `user18_name`. |
| {obj}`~pulserver.protocol.SequenceType` | Options of `sequence_type`. |
| {obj}`~pulserver.protocol.ImagingMode` | Options of `imaging_mode`. |
| {obj}`~pulserver.protocol.PreparationType` | Options of `preparation_type`. |
| {obj}`~pulserver.protocol.TriggerType` | Options of `trigger_type`: Pulseq trigger channel names. |

## Prescription

The field-of-view offset and orientation the interpreter fills from the
scanner's prescription. The entries close every listing and are not bound to a
design argument; the host applies the offset when it builds the IR, and checks
the design in the physical frame the orientation gives.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.protocol.PRESCRIPTION` | Keys of every prescription entry: the offset's, then the rotation's. |
| {obj}`~pulserver.protocol.FOV_OFFSET` | Keys of the field-of-view offset entries, in mm along the logical readout, phase and slice axes. |
| {obj}`~pulserver.protocol.FOV_ROTATION` | Keys of the rotation entries, row-major: element `(i, j)` of `R` in physical = R logical. |
| {obj}`~pulserver.protocol.prescribed_offset` | Field-of-view offset that protocol values carry, in metres. |
| {obj}`~pulserver.protocol.prescribed_rotation` | Rotation from logical to physical axes that protocol values carry, made orthonormal. |

## Wire blocks

A listing carries every entry with its schema, as the `list` design call replies
it; a value block carries values only, as the `validate` and `generate` calls
read and reply them. These functions are the only place where keys and values
become text and text becomes them. The parsers return the key members for the
names they read, and a value block parsed against a listing returns each
string list as one of the listing's options.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.protocol.PROTOCOL_BEGIN` | First line of every protocol block. |
| {obj}`~pulserver.protocol.PROTOCOL_END` | Last line of every protocol block. |
| {obj}`~pulserver.protocol.format_listing` | Format a protocol with its schema. |
| {obj}`~pulserver.protocol.RfPulse` | One pulse a sequence plays, and the protocol parameter its flip angle follows. |
| {obj}`~pulserver.protocol.format_pulses` | Format the RF a design plays, as the `list` design call replies it. |
| {obj}`~pulserver.protocol.parse_pulses` | Read the design and the RF it plays from a listing. |
| {obj}`~pulserver.protocol.parse_listing` | Read a protocol with its schema from a listing block. |
| {obj}`~pulserver.protocol.format_values` | Format a value block. |
| {obj}`~pulserver.protocol.parse_values` | Read a value block against the listing it was edited from. |
| {obj}`~pulserver.protocol.format_prescription` | Format prescription entries as the `name: value` lines of a block. |
| {obj}`~pulserver.protocol.parse_prescription` | Read the prescription entries out of the `name: value` lines of a block. |
| {obj}`~pulserver.protocol.Validation` | Reply to `VALIDATE`: validity, scan time, information line and values. |
| {obj}`~pulserver.protocol.format_validation` | Format a `VALIDATE` reply. |
| {obj}`~pulserver.protocol.parse_validation` | Read a `VALIDATE` reply. |
