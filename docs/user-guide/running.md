# Services

pulserver runs two services. The design calls answer the scanner's interpreter
host processes: they resolve protocols and write the designs the scanner plays
into a design store, one command per call or through a warm design server. The
reconstruction proxy receives each series from the scanner's reconstruction
client, reads the design it was played from in a design store, and returns the
images.

When the two run on different computers, either the design store is a
directory both can reach, or the proxy keeps a store of its own and its design
intake receives each design the design calls push to it. When the
reconstructions run on a computer of their own, a reconstruction server runs
there and the proxy forwards each series to it. The services and the design
store are described in {doc}`../explanations/architecture`.

## Design calls

```bash
pulserver design list     --plugins DIR --plugin NAME [--limits FILE --rf-definitions] [--socket PATH]
pulserver design validate --plugins DIR --plugin NAME --limits FILE [--rf-layout] [--exam DIR] [--socket PATH] < VALUES
pulserver design generate --plugins DIR --plugin NAME --limits FILE --store DIR --scan NAME [--exam DIR] [--push URL] [--socket PATH] < VALUES
pulserver design import   --limits FILE --store DIR --scan NAME [--push URL] [--socket PATH] < IMPORT
pulserver design push     --store DIR --scan NAME --to URL ID...
pulserver design prune    --store DIR [--max-age-days D] [--max-bytes B]
```

Each call is answered from its arguments and standard input alone and writes
its reply to standard output; `python -m pulserver.host` takes the same
arguments. `--plugins` is a directory of scanner-sequence plugin files,
`<plugin>.py`, and may be repeated: a plugin is the file of the first directory
that holds it, and the shipped plugins are searched last
({ref}`shipped-sequences`). `--store` is the design store, created when
missing, and `--scan` the name of the scan the design is stored for: one
directory name, not starting with `.`.

| Call | Reply |
| --- | --- |
| `list` | `PROTOCOL` and the plugin's listing block; with `--rf-definitions`, the `[RfDefinitions]` block |
| `validate` | `VALID <seconds>` or `INVALID`, an `INFO` line and the value block of the resolved protocol; with `--rf-layout`, a valid reply ends with the `[RfLayout]` block |
| `generate` | `GENERATED <id>`, the identifier of the design in the scan's directory |
| `import` | `IMPORTED <id>` |
| `push` | `PUSHED <count>`: the designs sent, not counting those the intake already holds |
| `prune` | `PRUNED <count>`: the scans removed |

A failed call replies `ERROR <message>` and exits with status 1; a plugin that
ends its process ends the call without a reply and with a nonzero exit status.
The listing and value blocks are those of {mod}`pulserver.protocol`, described
in {doc}`../explanations/protocol`.

The RF blocks are sent only when asked for, and only where the plugin's
evaluation states an RF layout ({ref}`stating-the-rf-layout`).
`--rf-definitions` evaluates the plugin at its default protocol under
`--limits`, which it then requires. `--rf-layout` follows a valid reply with the
RF layout of the evaluated request, its amplitudes relative to the peaks the
listing states. Without `--rf-definitions`, `list` depends on the plugin file
and the installed packages only, so its reply can be written once at
installation and read without a call.

`--exam` names the directory of the current exam's cache, which is passed to
the plugin hooks that take `exam` ({ref}`exam-cache`); for such a plugin the
files of that directory are part of the design's identity.

`generate` designs a request once, in the logical frame, checks it in the
physical frame of the prescription rotation, converts it to the IR cache at the
prescribed field-of-view offset and stores it. A request that resolves to a
stored design returns that design's identifier without designing again.

### Import block

`import` stores a Pulseq file written by any sequence development tool with a
Pulseq writer. The block names the first file of a `NextSequence` chain and,
optionally, its prescription and its number of averages:

```text
[Import]
file: /data/sequence.seq
fov_offset_x: 0.0
fov_offset_y: 20.0
fov_offset_z: 0.0
nex: 2
[Import End]
```

The offset is in mm along the logical readout, phase and slice axes, zero along
an axis left out, and the rotation is given by the `fov_rotation_ij` lines of
{doc}`../explanations/protocol`. `nex`, a whole number from 1, plays the main
sequence, the last file of the chain, that many times. The chain is copied,
then checked and converted as `generate` checks and converts a design.

(scanner-limits)=
### Scanner limits

`--limits` names a file holding a `[Limits]` block, one `name: value` per line,
here `system.txt`:

```text
[Limits]
max_grad: 40.0
grad_unit: mT/m
max_slew: 150.0
slew_unit: T/m/s
B0: 3.0
design_max_grad: 23.094
design_max_slew: 86.603
[Limits End]
```

The lines without a prefix are keyword arguments of `pypulseqpp.Opts`, of which
`B0`, the field in T, is required: ppm offsets are resolved at it. The limits
are the `system` the plugin designs under and the checks hold the design to, so
include the scanner's RF and ADC dead times, RF ringdown time and ADC sample
divisor; `pypulseqpp.Opts` sets the dead times to zero and the divisor to four
when they are left out.

`max_grad` and `max_slew` are the limits of the gradient coils, to which the
checks hold every physical axis. Logical axes played together add on one
physical axis, so under an oblique prescription a design held to `max_grad` and
`max_slew` per logical axis can exceed them on a physical one and is refused.
Two design limits cap what each logical axis is designed under, derated by the
scanner for the prescription's rotation:

| Limit | Meaning |
| --- | --- |
| `design_max_grad` | Gradient amplitude each logical axis is designed under, in the `grad_unit` of `max_grad` |
| `design_max_slew` | Slew rate each logical axis is designed under, in the `slew_unit` of `max_slew` |

A design limit above the scanner's leaves the scanner's, and one left out is the
scanner's. `validate` resolves a request under the same limits as `generate`.
An imported chain is not designed, and is checked against `max_grad` and
`max_slew` alone.

Options prefixed `ir_` configure the IR conversion:

| Option | Meaning |
| --- | --- |
| `ir_vendor` | `PULSEG_VENDOR_*` code the cache is tagged with; 0 is vendor-neutral |
| `ir_label_column_map` | Three Pulseq label state indices, separated by spaces, filling the ADC label columns |
| `ir_cache_ext` | Extension of the cache file, `.pseg` by default |
| `ir_vendor_file` | Vendor file the cache is grouped and stored under, read by {func}`~pulserver.ir.read_vendor`; a file it cannot read refuses the call |
| `ir_wave_max_samples` | Samples of waveform memory per gradient axis of the playout |
| `ir_wave_raster_us` | Gradient raster of the playout, in µs per sample |
| `ir_wave_load_us_per_sample` | Time the playout takes to load one sample on one axis, in µs; 0, the default, leaves loading unchecked |
| `ir_wave_headroom` | Fraction of the playout time that loading may take, 0.5 by default |
| `ir_wave_slots` | Slots per segment position that a streaming playout cycles through, 2 by default |

The `ir_wave_` options describe the playout's waveform memory, for which the
cache lays out the waveforms ({class}`~pulserver.ir.WaveBudget`);
`ir_wave_max_samples` and `ir_wave_raster_us` are given together. Without them
every waveform is held at once on the gradient raster of the chain's first
file, and a playout that holds them otherwise refuses the cache.

The remaining options are the limits of the checks besides the gradient limits
and rasters ({doc}`../explanations/safety-checks`). A scanner always sends its
nerve model and forbidden bands, the acoustic transfer function of its gradient
coil where it has one, and the VOPs for pTx. A check whose limits are left out
is not run. For pTx, each subsequence's SAR at the VOPs relative to a reference
pulse is written into the cache; the vendor's routines compute SAR and RF and
gradient heating from it.

| Limit | Meaning |
| --- | --- |
| `pns_chronaxie`, `pns_rheobase`, `pns_alpha` | Chronaxie nerve model: chronaxie in s, rheobase in T/m/s, `alpha` 1 when left out. The rheobase and the alpha are one value or three separated by spaces, one per physical axis |
| `pns_<axis>_<field>` | SAFE nerve model, for the axes `x`, `y` and `z` and the fields `a1` to `a3`, `tau1` to `tau3` in ms, `stim_limit` in T/m/s and `g_scale` |
| `pns_limit` | Largest PNS response allowed, as a fraction of the model's threshold; 1 when left out |
| `forbidden_band_<n>` | One forbidden band: its physical axis (`x`, `y`, `z` or `all`), its lowest and highest frequency in Hz and, optionally, the largest amplitude allowed in it in mT/m, separated by spaces. Without an amplitude, the `min_threshold` of `pypulseqpp.safety.check_mech_resonance` |
| `vop_file` | `.mat` or `.npz` file of VOPs and the head SAR matrix of each body model, at a path the design calls can read; its `safety_factor` multiplies local SAR |
| `vop_head_limit`, `vop_local_limit` | The scanner's head and local SAR limits in W/kg in its current operating mode |
| `vop_drive_per_hz` | Channel drive per Hz of RF amplitude, one value or one per channel separated by spaces; a scale common to every channel cancels. Equal drives when left out |
| `vop_default_shim` | Magnitude and phase in rad of each channel's weight for a pulse played without an RF shim, separated by spaces; equal weights whose squares sum to 1 when left out |
| `vop_coil` | Transmit configuration the scanner reports; the VOP file's `transmit` metadata must name the same one |
| `acoustic_file` | HDF5 file of the gradient coil's acoustic transfer function, at a path the design calls can read; the peak and A-weighted average sound pressure levels are held to the limits of IEC 60601-2-33 and written into the cache |
| `acoustic_interval_us` | Sampling interval in µs that the bins of `acoustic_file` refer to; given with it |

A call refuses limits it cannot read. A design is identified by the contents
of the VOP, vendor and acoustic files as well as by their paths.

### In-process calls

{func}`~pulserver.host.call` answers a design call in the calling process,
which exercises a plugin without a scanner:

```python
from pulserver.host import DesignStore, call

limits = {"max_grad": 40.0, "grad_unit": "mT/m", "max_slew": 150.0,
          "slew_unit": "T/m/s", "B0": 3.0}
block = "[Protocol]\nTE: 5000\nnx: 96\n[Protocol End]\n"
status, reply = call("validate", plugins="sequences", plugin="gre2d",
                     limits=limits, block=block)
status, reply = call("generate", plugins="sequences", plugin="gre2d",
                     limits=limits, block=block,
                     store=DesignStore("designs", "scan"))
```

A plugin file is imported again when its modification time changes.

### Design store

The store holds one directory per scan, `<store>/<scan>/`, and a scan's
directory holds its design, `<store>/<scan>/<id>/`, laid out as
{doc}`../explanations/architecture` shows. A design is reused while its scan
holds it: a call that resolves to the same design replies with it, and one
that resolves to another stores the new design in place of the scan's
previous one. `prune` removes whole scans, least recently used first: those
unused for longer than `--max-age-days`, then others until the store holds at
most `--max-bytes`. A scan used within the last day is kept whatever the
limits, since it may still be playing or reconstructing. A store that receives
pushed designs is pruned the same way, on the reconstruction computer.

### Design push

With `--push`, or the `PULSERVER_DESIGN_PUSH` environment variable, naming the
URL of a proxy's design intake, `http://<host>:<intake-port>`, `generate` and
`import` send their design to that intake before replying; `push` sends the
stored designs it names to the intake `--to` names. A design travels as a
bundle, a gzip-compressed tar of the files of its directory, and the intake
stores it only when every file has the SHA-256 its manifest records. A design
the intake already holds is not sent again.

A `generate` or `import` whose design cannot be pushed replies
`ERROR design <id> is stored but not pushed: <reason>` and exits with status 1,
so the interpreter receives no identifier of a design the intake lacks. The
design stays stored, and the same call made again pushes it without designing
again. `push` replies `ERROR design <id> is not pushed: <reason>` for the first
design it cannot send.

### Warm design server

```bash
pulserver design serve --plugins DIR --socket PATH
```

A call answered in its own process spends most of its time importing the
design engine. A warm server imports it once, designs, checks and converts one
sequence of its own, and imports every plugin of its `--plugins` directories.
It answers each call forwarded to its socket in a child process forked from
it, so nothing a call does outlives the call, and a plugin that ends its
process fails only its own call, with
`ERROR the design call ended with exit status N`. Calls run concurrently, one
child each.

`list`, `validate`, `generate` and `import` are forwarded when `--socket`, or
the `PULSERVER_DESIGN_SOCKET` environment variable, names the socket of a
running server, and are answered in their own process otherwise; the reply is
the same either way. Forwarding a call imports neither NumPy nor pypulseqpp.
The server stops on `SIGINT` or `SIGTERM` and ends the calls it is running. The
socket is neither authenticated nor encrypted: its file permissions decide who
may call.

The server logs one line per call to standard error, naming the call, its
plugin, its exit status and its duration. A request sent to the socket whose
JSON carries `log`, a file path, has its child append everything else it logs
and prints to that file, so an interpreter can keep the design calls of one
scan beside the rest of that scan's log.

## Reconstruction proxy

```bash
pulserver proxy --store DIR --port N --plugins DIR [--host ADDR] [--intake-port N] [--queue DIR] [--exams DIR] [--slots N] [--gpu-slots 1] [--spares 1] [--recon-timeout S] [--dicom] [--save-data DIR] [--idle-timeout S] [--logfile FILE] [--log-level LEVEL]
pulserver proxy --store DIR --port N --forward HOST:PORT [--forward-config NAME] [--host ADDR] [--intake-port N] [--recon-timeout S] [--dicom] [--save-data DIR] [--idle-timeout S] [--logfile FILE] [--log-level LEVEL]
```

| Option | Meaning |
| --- | --- |
| `--store` | The design store the proxy reads: the one the design calls write, or the one the intake writes. It also holds each scan's pose file, `<store>/<scan>/motion.buf`, for a scan corrected for motion |
| `--port` | TCP port the scanner's reconstruction client connects to |
| `--host` | Address to listen on; the loopback interface when unset, `0.0.0.0` for every interface |
| `--intake-port` | TCP port of the design intake, on the `--host` address; no intake when unset |
| `--plugins` | Directory of reconstruction plugin files, `<plugin>.py`, repeatable, searched in order before the shipped ones; required unless `--forward` is given |
| `--queue` | Directory the series waiting for a slot are written to; a temporary directory, removed when the proxy stops, when unset |
| `--exams` | Directory in which the proxies of this host share the caches of an exam, one directory per exam named by its identity: `exam-<ExamID>` for a header carrying `ExamID`. Created readable by the proxy's user alone; one that exists is used with the permissions the site gave it, so design calls running as another user can read the exam's maps. A location under the system temporary directory, private to the proxy's user, when unset |
| `--slots` | Series reconstructed at once; derived from available memory and the GPUs when unset |
| `--gpu-slots` | Series reconstructed at once on each GPU when `--slots` is unset; 1 by default |
| `--spares` | Worker processes started ahead of a series; 1 by default |
| `--recon-timeout` | Seconds a reconstruction may run after its series ends; the worker is then terminated, or the connection to the server closed, and the client told. Unlimited when unset |
| `--forward` | `HOST:PORT` of the MRD server that reconstructs every series, instead of local workers |
| `--forward-config` | Config name sent to the `--forward` server; the series' reconstruction plugin when unset |
| `--dicom` | Convert each image to DICOM before it is relayed, whatever reconstructed it; for a client that reads DICOM only |
| `--save-data` | Directory each series is kept in as the scanner sends it; kept nowhere when unset |
| `--idle-timeout` | Seconds without a client after which the proxy closes, never while a series is reconstructed. Unlimited when unset |
| `--logfile` | File the log is appended to, together with the standard output and error of the proxy and its workers, so whatever a reconstruction plugin logs, prints or warns lands there too; standard error when unset |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` or `ERROR`; `INFO` when unset |

The proxy and its workers are described in
{doc}`../explanations/reconstruction-session`, and the messages and header
fields the client sends in {doc}`reconstruction-client`.

### Series routing

The MRD header of each series names the scan in the `pulserver_scan` user
parameter and the design it was played from in `pulserver_design`, and the
proxy refuses a series whose header names no design stored for that scan. The client's config names the reconstruction plugin, as
a bare plugin name or the path of its file, or as either under
`parameters.config`; the design names none. A series whose config names none
is refused unless the proxy forwards with `--forward-config`.

A series that finds every slot busy is written to the queue directory as it
arrives and reconstructed once a slot frees; the client stays connected
meanwhile. What goes back to a client whose connection has dropped is kept for
a day: a client that opens a series with the same `measurementID` is sent it in
place of a new reconstruction.

### Saved series

With `--save-data`, each series is also written to
`<save-data>/mrd_<timestamp>.h5` as the scanner sends it, header first and each
acquisition as it arrives, before it is enriched. `pulserver replay` runs a
plugin on that file and reconstructs it offline as the proxy did
({ref}`series-replay`). The queue directory holds a series only until it is
reconstructed.

### Proxy lifetime

A proxy started for one scan outlives whatever started it, so `--idle-timeout`
ends it: it closes once no client has been connected for that long, counted
from the last client leaving, and never while a series is being reconstructed.
Without it, the proxy serves until it receives `SIGINT` or `SIGTERM`, and then
waits for the series it is running before it exits.

### Exam directory

A reconstruction computer may run one proxy per acquisition, so the series of
one exam are reconstructed by different processes. What a calibration series
measures is kept under `--exams`, in a directory of the exam's own, and read
there by a later series whichever proxy takes it: a hook assigns
`context.b0_map`, `context.b1_map` or `context.coil_sensitivities` and a later
hook reads them ({class}`~pulserver.recon.ReconContext`). An exam's directory
is removed once the last proxy on that exam has moved to another, so a proxy on
the next exam does not remove the data of one still reconstructing the last.

### Forwarding

With `--forward`, the proxy runs no workers: each series is enriched as it
arrives and sent on to the MRD server at `HOST:PORT`, whose own slots and queue
determine when it is reconstructed. The server receives a config file message
naming the series' reconstruction plugin, which a reconstruction server runs,
or the `--forward-config` name, such as the configuration by which a Gadgetron
or FIRE server selects its pipeline. The client's config text is not
forwarded. What the server returns is relayed to the client; a message type the
proxy cannot read ends the relay, and the client receives a `pulserver:` text
naming the type.

### Sequence description

Whether forwarded or reconstructed by a worker, the series carries, after its
header and before its first acquisition, one `TEXT` message (5) describing the
sequence chain for a signal simulation: a JSON object under the key
`pulserver_sequence_description`, holding per file of the chain one row per
block, RF at the pulse centre and ADC at the sample nearest the k-space centre,
and the pulses the RF rows name. It is meant for a reconstruction that
simulates the signal, which chooses the signal model the rows drive, and is not
relayed back to the client.

### Design intake

With `--intake-port`, the proxy runs a design intake beside it
({class}`~pulserver.proxy.DesignIntake`), an HTTP endpoint that writes the
designs pushed to it into `--store`. `HEAD /designs/<scan>/<id>` answers 200
when the scan holds the design and 404 otherwise; `PUT /designs/<scan>/<id>`
stores a bundle as the scan's design and answers 201, 200 for a design already
stored, or 400 with the reason for a bundle that is not the design its path
names. A `HEAD` for a stored design marks its scan as used for `prune`.

### Network exposure

The MRD stream is neither authenticated nor encrypted, and its header carries
patient data. The design intake is neither authenticated nor encrypted either,
and writes the designs it accepts into the store the proxy reconstructs from.
Run both on the network between the scanner and the reconstruction computer,
with `--host` naming the address of the interface on it; a scanner's
reconstruction client cannot reach the loopback default.

## Reconstruction server

```bash
pulserver recon --plugins DIR --port N [--host ADDR] [--queue DIR] [--slots N] [--gpu-slots 1] [--spares 1] [--recon-timeout S] [--save-data DIR] [--idle-timeout S] [--logfile FILE] [--log-level LEVEL]
```

The reconstruction server receives the series a forwarding proxy sends, on the
computer that reconstructs them. It reconstructs each series with the plugin
its config names, as a config file message or as a config text naming it as a
bare name or under `parameters.config`, and enriches nothing: a forwarded
series arrives enriched. Its options are those of the proxy, with the same
workers, slots, queue and exam directories; `--save-data` keeps each series as
the proxy forwarded it, enriched. Its stream is neither authenticated nor
encrypted, and `--host` names the interface the proxy reaches it on. It stops
on `SIGINT` or `SIGTERM` and waits for the series it is running.

(series-replay)=
## Series replay

```bash
pulserver replay SERIES --plugin FILE (--store DIR | --design SEQ) [--config TEXT] [--output DIR]
```

`pulserver replay` reconstructs one series kept by `--save-data` in the calling
process, with no proxy running. The series is enriched from the design it was
played from, at the field-of-view offset its header states, as the proxy
enriches it, and runs through the plugin's hooks
({ref}`offline-reconstruction`). The design is the one the header names in the
store `--store`, or the one whose sequence file, the first of its chain, is
`--design`. That file must be one the manifest beside it records, and the
manifest must name the design the header names. A series exported beside a
copy of its design's files replays with no store. Images go to `images.h5` in `--output`, and DICOM datasets to the files
they are named by. An exception raised by the plugin is not caught, so its
traceback names the line that raised it.

## Playout validation

```bash
pulserver validate SEQ [--vendor ge] [--played FILE] [--dead-time-us N] [--rf-wait-us N] [--tolerance MT] [--rf-tolerance PCT] [--limits FILE] [--cache-ext EXT]
```

| Option | Meaning |
| --- | --- |
| `SEQ` | The sequence file to check |
| `--vendor` | The machine the recording came from. Without one, the check is against the sequence's own IR cache |
| `--played` | The recording to check against; without one, and with a vendor named, the recording is requested from that vendor's tooling if it is installed |
| `--dead-time-us` | Time by which the recording runs ahead of the sequence on every channel, in µs; 0 by default |
| `--rf-wait-us` | Time by which the transmit channels run further ahead, in µs; 0 by default |
| `--tolerance` | Largest gradient difference that counts as agreement, in mT/m; derived from the slew rate and raster of `--limits` when unset |
| `--rf-tolerance` | Largest transmit difference, as a percentage of the peak; 1 by default |
| `--limits` | The `[Limits]` block, for a sequence that has no cache yet |
| `--cache-ext` | Extension of the cache, dot included; `.pseg` by default |

The command exits 0 where the two agree, 1 where they differ, and 2 where it
could not compare, and prints the largest difference on each gradient axis
against the peak that axis reaches.

Naming a vendor compares against a recording of a machine playing the
sequence, which establishes that the machine plays what was written. Naming
none compares against the waveforms the sequence's own IR cache holds, which
establishes only that the conversion preserved the sequence: a conversion and
a playout that are wrong in the same way agree. The output states which
comparison ran.

The gradients are compared in mT/m. The transmit channels are compared as the
recording stores them: the phase over a converter that spans one cycle, and the
magnitude by its shape, because converting a recorded count into Hz needs the
peak B1 of the scan being played, which the recording does not carry. The
fitted scale of the shape is printed; it is the number a machine's own
calibration would have to supply.

A machine starts a unit before it plays any of it, and drives its transmit and
gradient channels on separate timelines, so a recording runs ahead of the
sequence by one constant on every channel and by a second one on the transmit
channels. State them with `--dead-time-us` and `--rf-wait-us`; they are not
estimated, since an alignment fitted to the two waveforms could remove the
disagreement the comparison is meant to find. The default gradient tolerance is
what the gradients can slew through in the few raster steps by which the two
renderings may differ.

The check leaves nothing beside the sequence: where no cache exists, one is
built beside a copy and discarded.

## See also

* {doc}`../explanations/architecture`: the services, the design store and the identity of a design.
* {doc}`../explanations/reconstruction-session`: the proxy, its workers and forwarding.
* {doc}`reconstruction-client`: the MRD stream of a series.
* {doc}`../api/host` and {doc}`../api/proxy`: the design calls, the proxy and the reconstruction server.
