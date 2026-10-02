# Scanner sequences

A scanner sequence is a plugin file, `<name>.py` in a `--plugins` directory,
that binds an app to the scanner protocol with one
{class}`~pulserver.design.SequencePlugin` subclass. The app is a pypulseqpp
{class}`~pypulseqpp.sequences.SequenceApp`, as here, or a function that returns
the designed sequences ({ref}`function-apps`):

```python
# sequences/gre.py
from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp
from pulserver.design import SequencePlugin

class Gre(SequencePlugin):
    app = Gre2DApp
```

The interpreter names it `--plugin gre`. Without `protocol`, the protocol is the
app's defaults. A PyPulseq script becomes a `SequenceApp` by moving
what precedes its loop into `init_sequence` and the loop body into `kernel`
([from a PyPulseq script](https://pulserver.github.io/pypulseqpp/latest/user-guide/from-pypulseq.html)).
`MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s) are required on every application:
they cap the scanner's limits, which the design call passes in.

## Binding the protocol

`protocol` maps the parameters of the interpreter's table, the members of
{data}`~pulserver.protocol.ProtocolKey`, to entries;
{class}`~pulserver.protocol.UIParam` collects those of the UI controls. A plain
string naming a parameter is stored as its member, and a name outside the table
is refused when the class is defined. An entry names the argument of the app it
sets: an `init_sequence` argument of a sequence application, a keyword argument
of a function. The class does not name a reconstruction: the console's scan
or the reconstruction client does ({doc}`reconstruction-plugins`). A `recon`
attribute has no effect, and defining one raises a `DeprecationWarning`.
`ui` and `ScannerSequence` are deprecated names of `protocol` and
`SequencePlugin`, and a class using either warns with a `DeprecationWarning`.

```pycon
>>> from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp
>>> from pulserver.design import (FloatParam, IntParam, SequencePlugin, TEPreset,
...                               TimeParam, TRPreset, UIParam)
>>> class Gre2D(SequencePlugin):
...     app = Gre2DApp
...     protocol = {
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
| {class}`~pulserver.design.ChoiceParam` | dropdown | the chosen member of a `StrEnum` |
| {class}`~pulserver.design.ConfigParam` | not shown | none; a value declared to the interpreter |
| {class}`~pulserver.design.Description` | read-only text | none |

The options of the four string-list parameters are
{class}`~pulserver.protocol.SequenceType`,
{class}`~pulserver.protocol.ImagingMode`,
{class}`~pulserver.protocol.PreparationType` and
{class}`~pulserver.protocol.TriggerType`, which
{class}`~pulserver.design.ChoiceParam` takes as its `choices`:
`ChoiceParam("mode", ImagingMode)`. The argument receives the member, a `str`
equal to the option. {func}`~pulserver.design.StringListParam`, which builds
the enum from option strings, is deprecated.

A preset is a negative value the interpreter sends in place of a time;
`{TEPreset.MINIMUM: None}` passes `None`, for which the application designs its
shortest echo time. An entry's `default` replaces the application's default as
the initial value, in UI units: `default=TEPreset.MINIMUM` offers a protocol a
scanner with weaker gradients than the application assumes can play.

(function-apps)=
## Function apps

A function app takes the scanner limits, a {class}`pypulseqpp.Opts`, and one
keyword argument for each entry of `protocol`, and returns the designed
{class}`pypulseqpp.Sequence`. A list of sequences is a chain, the prescans first
and the main sequence last; each file names the next as its `NextSequence`
definition. The initial value of an entry is the default of its argument in the
signature, so a {func}`functools.partial` is an app too. The function is called
when the design is generated, not when a protocol is evaluated
({ref}`evaluating-a-protocol`).

```pycon
>>> import numpy as np
>>> import pypulseqpp as pp
>>> def pulse_acquire(system, flip=90.0, tr=100e-3, averages=8):
...     seq = pp.Sequence(system)
...     rf = pp.make_block_pulse(np.deg2rad(flip), duration=1e-3, system=system)
...     adc = pp.make_adc(256, duration=10e-3, system=system)
...     for _ in range(averages):
...         seq.add_block(rf)
...         seq.add_block(adc)
...         seq.add_block(pp.make_delay(tr - 11e-3))
...     return seq
>>> class PulseAcquire(SequencePlugin):
...     app = pulse_acquire
...     protocol = {
...         UIParam.FLIP: FloatParam("flip", unit="deg", range_min=1.0, range_max=180.0),
...         UIParam.TR: TimeParam("tr", range_min=20000, range_max=5_000_000),
...         UIParam.NEX: IntParam("averages", range_min=1, range_max=64),
...     }
>>> listing = PulseAcquire().listing()
>>> {key.value: listing[key].value for key in PulseAcquire.protocol}
{'flip': 90.0, 'TR': 100000, 'nex': 8}

```

(shipped-sequences)=
## Shipped sequences

pulserver ships a plugin for each of these pypulseqpp sequences, searched after
every `--plugins` directory, so a file of the same name there replaces it:

| Plugin | Paired reconstruction |
| --- | --- |
| `gre2d`, `se2d`, `bssfp2d`, `gre_multiecho2d`, `gre3d`, `se3d` | `pics` |
| `gre_radial2d`, `gre_spiral2d`, `se_radial2d`, `se_spiral2d` | `nufft` |
| `gre_stack_of_stars3d`, `gre_stack_of_spirals3d`, `se_stack_of_stars3d`, `se_stack_of_spirals3d` | `nufft` |
| `epi2d` | `epi` |

The pair is what a console reconstructs a shipped sequence with when the scan
names no reconstruction ({doc}`virtual-scanner`); a scan can name another, and
the reconstruction client of a scanner names one in its config
({doc}`reconstruction-client`).

A 3D sequence takes its number of partitions from the number of slices. The
Cartesian ones take `Ry`, and the 3D ones `Rz`, as their undersampling.

## Resolving a protocol

{meth}`~pulserver.design.SequencePlugin.listing` is the protocol with its
schema, as the `list` design call replies it:

```pycon
>>> from pulserver.protocol import format_listing
>>> print(format_listing(Gre2D().listing()), end="")
[Protocol]
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
[Protocol End]

```

The last twelve entries are the prescription, which the interpreter fills from
the scanner's: the field-of-view offset, which the host applies when it builds
the IR, and the rotation from the logical to the physical axes, in whose frame
the host checks the design ({doc}`../explanations/ir-cache`).

{meth}`~pulserver.design.SequencePlugin.validate` evaluates a request under the
scanner limits and returns the protocol the design plays. Entries the request
omits keep their initial values:

```pycon
>>> import pypulseqpp as pp
>>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
>>> reply = Gre2D().validate(system, {"TE": TEPreset.MINIMUM, "nx": 96})
>>> reply.valid, reply.duration, reply.info
(True, 36.0, '')
>>> {name: reply.values[name] for name in ("TE", "TR", "fov", "nx")}
{'TE': 3080, 'TR': 250000, 'fov': 220.0, 'nx': 96}

```

For a sequence application, the resolved `TE` is the echo time the design
achieved, as `init_sequence` records it with
{meth}`~pypulseqpp.sequences.SequenceApp.resolve`; an argument the application
does not record keeps the requested value. Values are rounded to the precision
a scanner parameter stores, so a resolved protocol sent back resolves to
itself.

A protocol the evaluation rejects is invalid, and the message of the error it
raised is the reply's `info`:

```pycon
>>> reply = Gre2D().validate(system, {"TE": 1000})
>>> reply.valid, reply.info
(False, 'the requested TE of 1.000 ms is shorter than the 3.400 ms this readout can achieve')

```

`duration` is the scan time in seconds. For a sequence application it is
{meth}`~pypulseqpp.sequences.SequenceApp.scan_time`: the `duration`
`init_sequence` states, otherwise the summed duration of the designed prescans
and main sequence. A duration of `0.0` states no estimate: the protocol is
valid, `duration` is `None`, and the reply reports the scan time as unknown.

{meth}`~pulserver.design.SequencePlugin.design` generates the design, writes it
as signed binary Pulseq, prescans first, and returns the written paths. The first file
is `sequence.seq`, and each file names the next as its `NextSequence`
definition. The `generate` design call converts the design to the IR cache and
stores it ({doc}`../explanations/designs`).

(evaluating-a-protocol)=
## Evaluating a protocol

A plugin overrides {meth}`~pulserver.design.SequencePlugin.evaluate` to check a
protocol and to state what the console shows with it. The hook takes the scanner
limits and the requested {class}`~pulserver.design.Protocol`, in the units of the
app's arguments, and returns an {class}`~pulserver.design.Evaluation`: the
protocol holding the values the design achieves, the scan time in seconds, a
note shown with the valid protocol and, optionally, the RF layout
({ref}`stating-the-rf-layout`). Returning `None` accepts the protocol
unchanged. Raising an exception makes the protocol invalid:

```pycon
>>> from pulserver.design import Evaluation
>>> class Timed(PulseAcquire):
...     def evaluate(self, system, protocol):
...         if protocol[UIParam.TR] < 12e-3:
...             raise ValueError("the TR is shorter than the pulse and the readout")
...         scan_time = protocol[UIParam.NEX] * protocol[UIParam.TR]
...         return Evaluation(protocol, scan_time, "256 samples per average")
>>> reply = Timed().validate(system, {"TR": 50000, "nex": 4})
>>> reply.valid, reply.duration, reply.info
(True, 0.2, '256 samples per average')
>>> reply = Timed().validate(system, {"TR": 10000})
>>> reply.valid, reply.info
(False, 'the TR is shorter than the pulse and the readout')

```

The default for a sequence application constructs it and returns its scan time
and the values it recorded. The default for a function app accepts the protocol
unchanged, states no scan time and does not call the app. A protocol the
function cannot realize is then found when the sequence is generated, as an
error of that call, and not reported as invalid while the operator edits it. An
`evaluate` that builds or checks the design rejects the protocol when it is
evaluated.

`ValueError` and `AssertionError`, which pypulseqpp and PyPulseq raise for an
event or a timing they cannot realize, reject the protocol with the message of
the error as `info`, and are logged as a warning with their traceback. Any
other exception also makes the protocol invalid, but the reply names only its
type, as in `TypeError in Timed.evaluate`, and the exception is logged as an
error with its traceback. A request that names an entry the protocol does not
declare, or an option a choice does not offer, is invalid in the same way.

{meth}`~pulserver.design.SequencePlugin.generate` is the hook that builds the
sequence of a requested protocol. The default for a function app calls the app
with the arguments of the protocol, and the default for a sequence application
constructs it. A plugin overrides `generate` to return a sequence, or a list of
them, built another way. `validate` and `design` are not overridden: they are the
boundary between a request and the code of a plugin, and `design` writes nothing
for an invalid request.

(stating-the-rf-layout)=
## Stating the RF layout

An evaluation may state the RF its protocol plays, from which a scanner
estimates the RF of a prescription before the design is generated.
{meth}`RfLayout.of <pulserver.design.RfLayout.of>` takes the RF instances of a
sequence, as {meth}`pypulseqpp.Sequence.rf_instances` returns them, and the
control each instance's amplitude follows: the flip angle, `UIParam.FLIP`, or a
float user entry, `UIParam.user_value(n)`, either declared in `protocol`. One
control applies to every instance; a list gives one per instance in play order,
`None` for an instance no entry scales. The scanner multiplies the amplitude of
an instance by the ratio of the value of its control in the protocol it plays to
the value in the evaluated protocol. The `period` is the time in seconds over
which the instances repeat, and the duration of the sequence where omitted:

```pycon
>>> from pulserver.design import RfLayout
>>> from pulserver.protocol import format_rf_layout
>>> class Costed(PulseAcquire):
...     def evaluate(self, system, protocol):
...         seq = pulse_acquire(system, **protocol.arguments)
...         layout = RfLayout.of(seq, UIParam.FLIP)
...         return Evaluation(protocol, seq.duration()[0], rf_layout=layout)
>>> reply = Costed().validate(system, {"nex": 4})
>>> print(format_rf_layout(reply.rf_layout), end="")
[RfLayout]
period 0.4
run 0 1 flip 4
[RfLayout End]

```

An evaluation may build one representative repetition, a TR, a shot or a
train, and state its `period`: `RfLayout.of(seq, UIParam.FLIP, period=tr)`.

The layout is optional. An evaluation that states none is valid and states no
estimate, and one whose control is not an entry of `protocol`, or has no
positive value in the evaluated protocol, is invalid. What the scanner checks
before the scan is the stored design, not the layout. The blocks that carry the
layout are described in {doc}`../explanations/protocol`.

## See also

* {doc}`../explanations/protocol` — resolution, presets, units and precision.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
