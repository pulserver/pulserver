# Scanner sequences

A scanner sequence is a plugin file, `<name>.py` in a `--plugins` directory,
that binds a sequence function to the scanner protocol with one
{class}`~pulserver.design.SequencePlugin` subclass. The sequence function is a function that
returns the designed sequences ({ref}`sequence-functions`), as every sequence of
pypulseqpp is:

```python
# sequences/gre.py
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d
from pulserver.design import SequencePlugin

class Gre(SequencePlugin):
    app = gre2d
```

The interpreter names it `--plugin gre`. Without `protocol`, the protocol is the
sequence function's defaults. A PyPulseq script becomes a sequence function by taking the scanner
limits and its parameters as arguments and returning its sequence
([from a PyPulseq script](https://pulserver.github.io/pypulseqpp/latest/user-guide/from-pypulseq.html)).

## Binding the protocol

`protocol` maps the parameters of the interpreter's table, the members of
{data}`~pulserver.protocol.ProtocolKey`, to entries;
{class}`~pulserver.protocol.UIParam` collects those of the UI controls. A plain
string naming a parameter is stored as its member, and a name outside the table
is refused when the class is defined. An entry names the keyword argument of the
sequence function it sets. The class does not name a reconstruction: the console's scan
or the reconstruction client does ({doc}`reconstruction-plugins`). A `recon`
attribute has no effect, and defining one raises a `DeprecationWarning`.
`ui` and `ScannerSequence` are deprecated names of `protocol` and
`SequencePlugin`, and a class using either warns with a `DeprecationWarning`.

```pycon
>>> from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d
>>> from pulserver.design import (FloatParam, IntParam, SequencePlugin, TEPreset,
...                               TimeParam, TRPreset, UIParam)
>>> class Gre2D(SequencePlugin):
...     app = gre2d
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
| {class}`~pulserver.design.FloatParam` | the argument divided by `scale`, in `unit` | as the sequence function takes it |
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
`{TEPreset.MINIMUM: None}` passes `None`, for which the sequence function designs its
shortest echo time. An entry's `default` replaces the sequence function's default as the
initial value, in UI units: `default=TEPreset.MINIMUM` offers a protocol a
scanner with weaker gradients than the sequence function's default assumes can play.

(sequence-functions)=
## Sequence functions

A sequence function takes the scanner limits, a {class}`pypulseqpp.Opts`, and one
keyword argument for each entry of `protocol`, and returns the designed
{class}`pypulseqpp.Sequence`. A list of sequences is a chain, the prescans first
and the main sequence last; each file names the next as its `NextSequence`
definition. The initial value of an entry is the default of its argument in the
signature, so a {func}`functools.partial` is a sequence function too. The default evaluation
does not call the function; {meth}`~pulserver.design.SequencePlugin.generate`
calls it when the design is generated ({ref}`evaluating-a-protocol`).

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
| `gre2d`, `se2d`, `bssfp2d`, `gre_multiecho2d`, `gre3d`, `se3d`, `bssfp3d`, `gre_multiecho3d` | `pics` |
| `fse3d`, `mprage3d` | `pmc` |
| `gre_radial2d`, `gre_spiral2d`, `se_radial2d`, `se_spiral2d`, `gre_propeller2d`, `se_propeller2d`, `se_epi_propeller2d`, `zte3d` | `nufft` |
| `gre_stack_of_stars3d`, `gre_stack_of_spirals3d`, `gre_stack_of_blades3d`, `se_stack_of_stars3d`, `se_stack_of_spirals3d`, `se_stack_of_blades3d` | `nufft` |
| `mprage_stack_of_stars3d`, `mprage_stack_of_spirals3d` | `nufft_train` |
| `epi2d`, `epi3d` | `epi` |

The pair is what a console reconstructs a shipped sequence with when the scan
names no reconstruction ({doc}`virtual-scanner`); a scan can name another, and
the reconstruction client of a scanner names one in its config
({doc}`reconstruction-client`).

Optional features are switched by a module constant of the plugin, off in
the shipped files; a copy of the file in a `--plugins` directory with the
constant set replaces the shipped plugin. A switch adds the entries the
feature needs, a scanner entry where there is one and a user entry otherwise:

| Plugin | Constant | Feature | Entries added |
| --- | --- | --- | --- |
| `epi2d` | `MULTIBAND` | Simultaneous multislice | `multiband` |
| `fse3d` | `OPTIMIZED` | Refocusing train designed with blochsim, shared by every shot | Flip angle: the refocusing angle at the TE echo |
| `fse3d` | `DUAL_REGION` | Trains designed with blochsim for the centre and the periphery of k-space, each shot's a cubic step between them; `TR` and `ETL` are the centre's | Flip angle, as `OPTIMIZED`; user entries 0 and 1: TR and ETL at the periphery |
| `fse3d`, `mprage3d` | `NAVIGATOR` | Three-plane navigators; the design sets `EnablePmc` and `pmc` states each pose to the scan | None |
| `gre3d`, `gre_multiecho3d`, `mprage3d`, `fse3d` | `WAVE` | Wave-CAIPI at `WAVE_AMPLITUDE` (T/m) and `WAVE_CYCLES`; the calibration region is acquired first without the wave, and `pics` and `pmc` reconstruct the wave-encoded volume | None |
| `gre_spiral2d`, `se_spiral2d`, `gre_stack_of_spirals3d`, `se_stack_of_spirals3d`, `mprage_stack_of_spirals3d` | `VARIABLE_DENSITY` | Variable-density spirals | User entry 0: periphery undersampling |

A 3D sequence takes its number of partitions from the number of slices. `Ry`
undersamples the phase encode of the Cartesian sequences, the blades of
`gre_propeller2d` and `se_propeller2d` and the shells of `zte3d`, and `Rz` the
partition encode of the 3D Cartesian ones. The `ETL` of `se_epi_propeller2d` is
its blade width, and the MPRAGE sequences take their inversion time as
`prep_time`.

Each shipped plugin binds the function of its pypulseqpp sequence and evaluates
a protocol from one repetition of the scan, one line, partition, spoke,
interleaf, blade line, echo train or inversion shot of one slice without dummy
repetitions ({ref}`evaluating-a-protocol`). Most call the function with the
arguments that reduce it to that repetition; `epi3d` and `zte3d` build the
repetition from the modules their function builds it from. The scan time is the duration of the repetition
times the number of repetitions the prescription plays, with the slices that
share a TR counted in the packets the sequence plays them in. The values are
those the main sequence states for the prescription: where the plugin has the
entry, the echo time and the repetition time in its `TE` and `TR` definitions,
the receiver bandwidth as the inverse of the dwell time of its first ADC event,
and the slice thickness in its `SliceThickness` definition; a multi-echo
sequence lists every echo time, and the `TE` entry holds the first. A
prescription the design refuses is invalid with the message of its error. The
RF layout is the first TR of the repetition, its shot repeated once per slice of
the largest packet, with the TR as its period; for the balanced steady state it
is the TR after the half-angle pulse ({ref}`stating-the-rf-layout`). The TR of
a fast spin echo holds an echo train, that of an MPRAGE an inversion shot, and
that of `epi3d` a volume. The flip angle is the control of every excitation of
the plugins with a `flip` entry, which an MPRAGE's inversion is not. The
spin-echo plugins, `fse3d` among them, have none, and the amplitude of their
excitation and refocusing pulses is that of the design.

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
the host checks the design ({doc}`../explanations/designs`).

{meth}`~pulserver.design.SequencePlugin.validate` evaluates a request under the
scanner limits and returns the protocol the design plays. Entries the request
omits keep their initial values. The default evaluation of a sequence function
accepts the protocol unchanged, so the reply repeats the request, a preset
included, and states no scan time:

```pycon
>>> import pypulseqpp as pp
>>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
>>> reply = Gre2D().validate(system, {"TE": TEPreset.MINIMUM, "nx": 96})
>>> reply.valid, reply.duration, reply.info
(True, None, '')
>>> reply.values["TE"] is TEPreset.MINIMUM, reply.values["nx"]
(True, 96)

```

An evaluation that designs the sequence states the values the design achieved.
The `TE` and `TR` definitions of a pypulseqpp sequence record the echo time and
the repetition time of the design, and
{func}`pypulseqpp.sequences.duration` sums the durations of the sequences of a
chain:

```pycon
>>> from pypulseqpp import sequences
>>> from pulserver.design import Evaluation
>>> class Resolved(Gre2D):
...     def evaluate(self, system, protocol):
...         seq = self.app(system, **protocol.arguments)
...         achieved = {UIParam.TE: seq.definitions["TE"][0],
...                     UIParam.TR: seq.definitions["TR"][0]}
...         return Evaluation(protocol.replace(achieved), sequences.duration(seq))
>>> reply = Resolved().validate(system, {"TE": TEPreset.MINIMUM, "nx": 96})
>>> reply.valid, round(reply.duration, 3), reply.info
(True, 36.0, '')
>>> {name: reply.values[name] for name in ("TE", "TR", "fov", "nx")}
{'TE': 3080, 'TR': 250000, 'fov': 220.0, 'nx': 96}

```

The resolved `TE` is the shortest echo time of the design in place of the
preset. Values are rounded to the precision a scanner parameter stores, so a
resolved protocol sent back resolves to itself.

A protocol the evaluation rejects is invalid, and the message of the error it
raised is the reply's `info`:

```pycon
>>> reply = Resolved().validate(system, {"TE": 1000})
>>> reply.valid, reply.info
(False, 'the requested TE of 1.000 ms is shorter than the 3.400 ms this readout can achieve')

```

`duration` is the scan time in seconds the evaluation states. A duration of
`0.0` states no estimate: the protocol is valid, `duration` is `None`, and the
reply reports the scan time as unknown.

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
sequence function's arguments, and returns an {class}`~pulserver.design.Evaluation`: the
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

The default for a sequence function accepts the protocol unchanged, states no scan
time and does not call the sequence function. A protocol the function cannot realize is then
found when the sequence is generated, as an error of that call, and not reported
as invalid while the operator edits it. An `evaluate` that builds or checks the
design rejects the protocol when it is evaluated.

`ValueError` and `AssertionError`, which pypulseqpp and PyPulseq raise for an
event or a timing they cannot realize, reject the protocol with the message of
the error as `info`, and are logged as a warning with their traceback. Any
other exception also makes the protocol invalid, but the reply names only its
type, as in `TypeError in Timed.evaluate`, and the exception is logged as an
error with its traceback. A request that names an entry the protocol does not
declare, or an option a choice does not offer, is invalid in the same way.

{meth}`~pulserver.design.SequencePlugin.generate` is the hook that builds the
sequence of a requested protocol. The default calls the sequence function with the arguments
of the protocol. A plugin overrides `generate` to return a sequence, or a list
of them, built another way. `validate` and `design` are not overridden: they
are the boundary between a request and the code of a plugin, and `design`
writes nothing for an invalid request.

(stating-the-rf-layout)=
## Stating the RF layout

An evaluation may state the RF its protocol plays, from which a scanner
estimates the RF of a prescription before the design is generated. The layout is
one TR: the RF instances one TR plays, and the TR as its `period`.
{meth}`RfLayout.of <pulserver.design.RfLayout.of>` takes the RF instances of a
sequence that is one TR, as {meth}`pypulseqpp.Sequence.rf_instances` returns
them, and the control each instance's amplitude follows: the flip angle,
`UIParam.FLIP`, or a float user entry, `UIParam.user_value(n)`, either declared
in `protocol`. One control applies to every instance; a list gives one per
instance in play order, `None` for an instance no entry scales. The scanner
multiplies the amplitude of an instance by the ratio of the value of its control
in the protocol it plays to the value in the evaluated protocol. The `period` is
the TR in seconds, and the duration of the sequence where omitted:

```pycon
>>> from pulserver.design import RfLayout
>>> from pulserver.protocol import format_rf_layout
>>> class Costed(PulseAcquire):
...     def evaluate(self, system, protocol):
...         one_tr = pulse_acquire(system, **(protocol.arguments | {"averages": 1}))
...         layout = RfLayout.of(one_tr, UIParam.FLIP)
...         scan_time = protocol[UIParam.NEX] * protocol[UIParam.TR]
...         return Evaluation(protocol, scan_time, rf_layout=layout)
>>> reply = Costed().validate(system, {"nex": 4})
>>> print(format_rf_layout(reply.rf_layout), end="")
[RfLayout]
period 0.1
run 0 1 flip 1
[RfLayout End]

```

A sequence shorter than the TR, such as one TR without its closing delay, is
stated with the TR: `RfLayout.of(seq, UIParam.FLIP, period=tr)`.

The layout is optional. An evaluation that states none is valid and states no
estimate, and one whose control is not an entry of `protocol`, or has no
positive value in the evaluated protocol, is invalid. What the scanner checks
before the scan is the stored design, not the layout. The blocks that carry the
layout are described in {doc}`../explanations/protocol`.

## Logging

A plugin logs with the standard `logging` module, under a logger of its own
(`logging.getLogger(__name__)`). Under a warm design server, what a call logs,
prints or warns, and why it refuses a protocol, are appended to the
file the request names under `log`, or to the server's standard error
({doc}`running`).

## See also

* {doc}`../explanations/protocol` — resolution, presets, units and precision.
* {doc}`../api/design` — the scanner-sequence interface and its UI entries.
