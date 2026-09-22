# Scanner sequences

A pypulseqpp sequence application bound to the entries of the scanner protocol.

```{eval-rst}
.. currentmodule:: pulserver.design
```

A plugin file in the host daemon's plugin directory defines one
{class}`ScannerSequence` subclass. Its `ui` maps each interpreter parameter
name to an entry naming the `init_sequence` argument it sets, and its methods
resolve a request into the protocol the application will play and write the
design. Conversion between the UI's units and the application's SI arguments,
and the rounding of resolved values to the precision of a float32 control
variable, are described in {doc}`../explanations/protocol`.

## Sequence

| Object | Description |
| --- | --- |
| {obj}`~pulserver.design.ScannerSequence` | A pypulseqpp application exposed to the scanner UI. |
| {obj}`~pulserver.design.load_plugin` | Import a plugin file and instantiate the scanner sequence it defines. |

## UI entries

| Object | Description |
| --- | --- |
| {obj}`~pulserver.design.TimeParam` | Time entry in integer microseconds, bound to an argument in seconds, with presets. |
| {obj}`~pulserver.design.FloatParam` | Float entry, bound to an argument scaled from the UI unit. |
| {obj}`~pulserver.design.IntParam` | Integer entry. |
| {obj}`~pulserver.design.BoolParam` | Checkbox. |
| {obj}`~pulserver.design.StringListParam` | Choice among option strings; the argument receives the string. |
| {obj}`~pulserver.design.ConfigParam` | Value declared to the interpreter; not shown or edited. |
| {obj}`~pulserver.design.Description` | Read-only text row. |
