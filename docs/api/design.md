# Scanner sequences

A sequence app bound to the entries of the scanner protocol.

```{eval-rst}
.. currentmodule:: pulserver.design
```

A plugin file in the design calls' plugin directory defines one
{class}`SequencePlugin` subclass. Its `app` is a pypulseqpp sequence application
or a function returning sequences, and its `protocol` maps each key of the
interpreter's parameter table, a {data}`~pulserver.protocol.ProtocolKey`, to an
entry naming the argument of the app it sets. Its methods evaluate a request
into the protocol the design plays and write the design. Conversion between the
UI's units and the app's SI arguments, and the rounding of resolved values to
the precision of a float32 control variable, are described in
{doc}`../explanations/protocol`.

## Sequence

| Object | Description |
| --- | --- |
| {obj}`~pulserver.design.SequencePlugin` | A sequence app bound to the scanner protocol. |
| {obj}`~pulserver.design.Evaluation` | The outcome of evaluating a valid protocol: the protocol the design achieves, its scan time and a note. |
| {obj}`~pulserver.design.Protocol` | The values of a protocol in the units of the app's arguments, convertible from and to the wire values. |
| {obj}`~pulserver.design.ScannerSequence` | Deprecated name of {class}`~pulserver.design.SequencePlugin`. |
| {obj}`~pulserver.design.load_plugin` | Import a plugin file and instantiate the sequence plugin it defines. |

## UI entries

| Object | Description |
| --- | --- |
| {obj}`~pulserver.design.TimeParam` | Time entry in integer microseconds, bound to an argument in seconds, with presets. |
| {obj}`~pulserver.design.FloatParam` | Float entry, bound to an argument scaled from the UI unit. |
| {obj}`~pulserver.design.IntParam` | Integer entry. |
| {obj}`~pulserver.design.BoolParam` | Checkbox. |
| {obj}`~pulserver.design.ChoiceParam` | Choice among the members of a `StrEnum`; the argument receives the member. |
| {obj}`~pulserver.design.StringListParam` | Deprecated: a `ChoiceParam` over an enum built from option strings. |
| {obj}`~pulserver.design.ConfigParam` | Value declared to the interpreter; not shown or edited. |
| {obj}`~pulserver.design.Description` | Read-only text row. |
