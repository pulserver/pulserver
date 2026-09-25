# Scanner sequences

A scanner sequence is a plugin file in the `--plugins` directory of the design calls.
It defines one {class}`~pulserver.design.ScannerSequence` subclass, which binds
a pypulseqpp {class}`~pypulseqpp.sequences.SequenceApp` to the entries of the
scanner protocol. A design call loads the file by its stem: `gre2d.py` is the
plugin a PSD host process names with `--plugin gre2d`.

## Binding the protocol

`app` is the sequence application and `ui` maps parameters of the interpreter's
table, named by {class}`~pulserver.protocol.UIParam` members, to entries. A name
outside that table is refused when the class is defined, because the
interpreter's parser would drop it. An entry names the `init_sequence` argument
it sets and how the scanner UI shows it; the application's own defaults are the
protocol's initial values.

```pycon
>>> from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp
>>> from pulserver.design import FloatParam, IntParam, ScannerSequence, TimeParam
>>> from pulserver.protocol import TEPreset, TRPreset, UIParam
>>> class Gre2D(ScannerSequence):
...     app = Gre2DApp
...     recon = "gre2d"
...     ui = {
...         UIParam.TE: TimeParam("te", range_min=1000, range_max=80000, range_incr=10,
...                         presets={TEPreset.MINIMUM: None}),
...         UIParam.TR: TimeParam("tr", range_min=1000, range_max=5_000_000,
...                         presets={TRPreset.MINIMUM: None}),
...         UIParam.FOV: FloatParam("fov_x", unit="mm", scale=1e-3,
...                           range_min=50.0, range_max=500.0),
...         UIParam.NX: IntParam("n_x", range_min=32, range_max=512, range_incr=2),
...     }

```

| Entry | UI | Argument |
| --- | --- | --- |
| {class}`~pulserver.design.TimeParam` | integer microseconds, with presets | seconds |
| {class}`~pulserver.design.FloatParam` | the argument divided by `scale`, in `unit` | as the application takes it |
| {class}`~pulserver.design.IntParam` | integer | integer |
| {class}`~pulserver.design.BoolParam` | checkbox | boolean |
| {class}`~pulserver.design.StringListParam` | dropdown | the chosen string |
| {class}`~pulserver.design.ConfigParam` | not shown | none; a value declared to the interpreter |
| {class}`~pulserver.design.Description` | read-only text | none |

The options of the four string-list parameters are
{class}`~pulserver.protocol.SequenceType`,
{class}`~pulserver.protocol.ImagingMode`,
{class}`~pulserver.protocol.PreparationType` and
{class}`~pulserver.protocol.TriggerType`, which
{class}`~pulserver.design.StringListParam` takes as they are.

A preset is a negative value the interpreter sends in place of a time.
`{TEPreset.MINIMUM: None}` passes `None` to the application, which designs its
shortest echo time; a preset can also map to a time in seconds or to a function
of the scanner limits.

`recon` names the reconstruction plugin the data of this sequence is
reconstructed with (see {doc}`reconstruction-plugins`).

## Resolving a protocol

{meth}`~pulserver.design.ScannerSequence.listing` is the protocol with its
schema, as the `list` design call replies it:

```pycon
>>> from pulserver.protocol import format_listing
>>> print(format_listing(Gre2D().listing()), end="")
[NimPulseqGUI Protocol]
TE: int|dropdown|8000|1000|80000|10|us|-2
TR: int|dropdown|250000|1000|5000000|1|us|-1
fov: float|typein|220.0|50.0|500.0|1.0|mm
nx: int|typein|128|32|512|2|
fov_offset_x: float|off|0.0|-1000.0|1000.0|0.1|mm
fov_offset_y: float|off|0.0|-1000.0|1000.0|0.1|mm
fov_offset_z: float|off|0.0|-1000.0|1000.0|0.1|mm
fov_rotation_11: float|off|1.0|-1.0|1.0|1e-06|
fov_rotation_12: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_13: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_21: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_22: float|off|1.0|-1.0|1.0|1e-06|
fov_rotation_23: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_31: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_32: float|off|0.0|-1.0|1.0|1e-06|
fov_rotation_33: float|off|1.0|-1.0|1.0|1e-06|
[NimPulseqGUI Protocol End]

```

The last twelve entries are the prescription, which the interpreter fills from
the scanner's: the field-of-view offset, which the host applies when it builds
the IR, and the rotation from the logical to the physical axes, in whose frame
the host checks the design ({doc}`../explanations/ir-cache`).

{meth}`~pulserver.design.ScannerSequence.validate` constructs the application
under the scanner limits and returns the protocol it will play. Entries the
request omits keep the application's defaults:

```pycon
>>> import pypulseqpp as pp
>>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
>>> reply = Gre2D().validate(system, {"TE": TEPreset.MINIMUM, "nx": 96})
>>> reply.valid, reply.duration, reply.info
(True, 36.0, '')
>>> {name: reply.values[name] for name in ("TE", "TR", "fov", "nx")}
{'TE': 3080, 'TR': 250000, 'fov': 220.0, 'nx': 96}

```

The values also carry the prescription entries as they were requested.

The resolved `TE` is the echo time the design achieved: the value the
application records with {meth}`~pypulseqpp.sequences.SequenceApp.resolve` in
`init_sequence`, as {attr}`~pypulseqpp.sequences.SequenceApp.resolved` reports
it. An entry whose argument the application does not record keeps the
requested value. Resolved values are reported at the precision a scanner
control variable stores, so a resolved protocol sent back resolves to itself.

A protocol the design refuses is invalid, and the error the application raised
is the reply's `info`:

```pycon
>>> reply = Gre2D().validate(system, {"TE": 1000})
>>> reply.valid, reply.info
(False, 'the requested TE of 1.000 ms is shorter than the 3.400 ms this readout can achieve')

```

`duration` is the scan time in seconds,
{meth}`~pypulseqpp.sequences.SequenceApp.scan_time`: the `duration` the
application states in `init_sequence`, which is reported without playing the
scan, and otherwise the summed duration of the designed prescans and main
sequence. An error the design raises while its scan time is computed makes the
reply invalid, as an error raised by construction does.

{meth}`~pulserver.design.ScannerSequence.generate` writes the design as signed
binary Pulseq, prescans first, and the `generate` call converts it to the IR
cache the scanner loads and stores it as a design, described in
{doc}`../explanations/designs`.

## See also

* {doc}`../explanations/protocol` — resolution, presets, units and precision.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
