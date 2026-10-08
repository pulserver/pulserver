# Protocol text blocks

The text blocks in which the design calls and the interpreter exchange
protocols and RF descriptions ({doc}`../../explanations/protocol`). The
grammar is implemented in {mod}`pulserver.protocol` for the host and in
`pulseg_protocol.h` for the interpreter.

## Listing and value blocks

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

The RF blocks follow a reply when the call asks for them, and no block follows
where the evaluation is invalid or states no layout. They are lists of numbers:
each number is ASCII decimal with nine significant digits, the numbers of a list
are separated by single spaces, and each list is one line, so that a reader
takes tokens. `[RfDefinitions]` follows the listing of a `list` call asked with
`rf_definitions`, which evaluates the plugin at its default protocol under the
scanner limits. Each definition is a `definition` line, then the sample times,
then the real and the imaginary parts of each channel:

```text
[RfDefinitions]
definition <index> <use> <flip_deg> <peak_hz> <bandwidth_hz> <delay_s> <center_s> <duration_s> <channels> <samples>
<time_s> ...
<real> ...
<imaginary> ...
[RfDefinitions End]
```

`index` is the number the runs of a layout name the definition by, from 0 in the
order of first play, and `use` is the RF use of pypulseqpp. `flip_deg` and
`peak_hz`, in degrees and Hz, are those of the first instance with a nonzero
amplitude, and `bandwidth_hz` is the bandwidth that
{func}`pypulseqpp.calc_rf_bandwidth` measures on the sum of the channels, in Hz.
`delay_s` is the delay of the RF event in its block; `center_s`, `duration_s` and
the sample times are in seconds from the start of the event, its delay excluded.
The channels share the sample times. The waveform is scaled to unit peak
magnitude over all channels and samples, with the RF shim applied; the frequency
and phase offsets of the event are playout parameters and are not applied.

`[RfLayout]` follows the value block of a valid `validate` reply asked with
`rf_layout`. The instances are run-length encoded in play order, one `run`
line for consecutive instances of one definition, control and printed
amplitude, and carry no samples:

```text
[RfLayout]
period <s>
run <index> <amplitude> <control|-> <count>
[RfLayout End]
```

`period` is the TR in seconds, over which the instances repeat. The `amplitude`
of a run is unitless: the peak RF amplitude of its instances over the `peak_hz`
the listing states for the definition, so that `amplitude × peak_hz × waveform`
is what an instance plays. The host reads those peaks by evaluating the plugin
at its default protocol as well. A run whose definition the listing does not
state, or states with a peak of zero, carries its amplitude relative to the first
instance of its definition with a nonzero amplitude in the evaluation of the
request, as does every run where the evaluation at the default protocol is
invalid. `control` is the wire name of the entry the amplitude is proportional
to, or `-`. A reader skips the blocks it does not know.

