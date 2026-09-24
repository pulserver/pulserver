# Running the services

Pulserver runs as two services. The host daemon answers the scanner's PSD host
processes: it resolves protocols and writes the designs the scanner plays. The
reconstruction proxy receives the raw data of each series from the scanner's
reconstruction client and returns the images. Both read and write one
directory, the *base*, which holds `bucket/`; when the two run on different
computers, the base is a directory both can reach. The layout of the bucket is
described in {doc}`../explanations/sessions`.

## Host daemon

```bash
python -m pulserver.host --base DIR --socket PATH --plugins DIR [--workers N]
```

| Option | Meaning |
| --- | --- |
| `--base` | Directory holding `bucket/`; created when missing |
| `--socket` | Unix socket to listen on; an existing file at this path is replaced |
| `--plugins` | Directory of scanner-sequence plugin files, `<plugin>.py` |
| `--workers` | Design worker processes, default 2 |

A PSD host process opens a session with the plugin it plays and the scanner
limits, then validates and generates protocols through it. The limits are
keyword arguments of `pypulseqpp.Opts`, one `name: value` per line, plus three
optional options for the IR conversion:

| Option | Meaning |
| --- | --- |
| `ir_vendor` | `PULSEG_VENDOR_*` code the cache is tagged with; 0 is vendor-neutral |
| `ir_label_column_map` | Three Pulseq label state indices, separated by spaces, filling the ADC label columns |
| `ir_cache_ext` | Extension of the cache file, `.pseg` by default |

A generated design and an imported chain are checked against the limits before
their IR cache is written ({doc}`../explanations/ir-cache`), so the limits
include the RF and ADC dead times, the RF ringdown time and the ADC sample
divisor of the scanner; `pypulseqpp.Opts` sets the dead times to zero and the
divisor to four when they are left out.

The limits also carry what the host checks besides the gradient limits and
rasters, the scanner's nerve model and its forbidden gradient bands, and, where
SAR is computed from virtual observation points, the VOPs. A check whose limits
are left out is not run. No SAR limit is checked on the host: it writes into
the cache each subsequence's SAR at the VOPs relative to a reference pulse
({doc}`../explanations/ir-cache`), and the PSD computes the SAR and the
gradient heating.

| Limit | Meaning |
| --- | --- |
| `pns_chronaxie`, `pns_rheobase`, `pns_alpha` | Chronaxie nerve model: chronaxie in s, rheobase in T/m/s, `alpha` 1 when left out |
| `pns_<axis>_<field>` | SAFE nerve model, for the axes `x`, `y` and `z` and the fields `a1` to `a3`, `tau1` to `tau3` in ms, `stim_limit` in T/m/s and `g_scale` |
| `pns_limit` | Largest PNS response allowed, as a fraction of the model's threshold; 1 when left out |
| `forbidden_band_<n>` | One forbidden band: its physical axis (`x`, `y`, `z` or `all`), its lowest and highest frequency in Hz and, optionally, the largest amplitude allowed in it in mT/m, separated by spaces |
| `vop_file` | `.mat` or `.npz` file of VOPs and, optionally, a global SAR matrix, at a path the host daemon can read |
| `vop_drive_per_hz` | Channel drive per Hz of RF amplitude, one per channel separated by spaces; a scale common to every channel cancels, and equal drives when left out |
| `vop_default_shim` | Magnitude and phase in rad of each channel's weight for a pulse played without an RF shim, separated by spaces; equal weights when left out |

A band given no amplitude is held to the `min_threshold` of
`pypulseqpp.safety.check_mech_resonance`. The gradient, PNS and resonance
checks are made in the physical frame of the prescription rotation each
request carries. `OPEN` refuses limits it cannot read, and a revision is
identified with the contents of the VOP file as well as its path.

The commands and their replies are listed in
{class}`~pulserver.host.HostDaemon`. {class}`~pulserver.host.HostClient` sends
them as a PSD host process does, which exercises a plugin through the daemon
without a scanner:

```python
from pulserver.host import HostClient, SessionKey

client = HostClient("/tmp/pulserver.sock", SessionKey(pid=4242, day=20711))
client.open("gre2d", {"max_grad": 40.0, "grad_unit": "mT/m",
                      "max_slew": 150.0, "slew_unit": "T/m/s"})
listing = client.list_protocol()
validation = client.validate({"TE": 5000, "nx": 96})
revision = client.generate(validation.values)
client.close()
```

A session opened without a plugin only imports sequence files
({meth}`~pulserver.host.HostClient.import_sequence`): the file and its
`NextSequence` chain are copied into a revision, checked in the physical frame
of the rotation the import names and converted to the IR cache there.

Plugin code runs in spawned worker processes, so a plugin that crashes fails
the command it was running and the daemon replaces the pool. A plugin file is
imported again when its modification time changes.

## Design command

```bash
pulserver design list     --plugins DIR --plugin NAME
pulserver design validate --plugins DIR --plugin NAME --limits FILE < VALUES
pulserver design generate --plugins DIR --plugin NAME --limits FILE --store DIR < VALUES
pulserver design import   --limits FILE --store DIR < IMPORT
pulserver design prune    --store DIR [--max-age-days D] [--max-bytes B]
```

Each call is answered from its arguments and standard input alone, and writes
the reply of the daemon command of the same purpose to standard output: `list`
replies as `LIST_PROTOCOL`, `validate` as `VALIDATE`, and `generate` and
`import` reply `GENERATED <id>` and `IMPORTED <id>`, where `<id>` names the
design in the store. A call that fails replies `ERROR <message>` and exits with
status 1. The limits file holds the `[Limits]` block `OPEN` takes: the scanner
limits, the conversion options and the check limits above. A plugin that ends
its process ends the call without a reply and with a nonzero exit status.

The store holds one directory per design, `<store>/<id>/`: the Pulseq files of
the chain, the IR cache, the resolved protocol and `manifest.json`, which
records the plugin, the reconstruction plugin, the limits, the package versions
and the SHA-256 of every file. The identifier is the first 18 hexadecimal
digits of the SHA-256 of what the design depends on: the plugin and its source,
the package versions, the limits and the resolved protocol, prescription
included. A request that resolves to a stored design returns its identifier
without designing again, and the identifier is three 24-bit integers, each held
exactly by a float32 CV. A design is written into a stage and renamed into
place, so it is read whole or not at all. `prune` removes designs least
recently used first: those unused for longer than `--max-age-days`, then others
until the store holds at most `--max-bytes`. Nothing is removed otherwise.

`list` depends on the plugin file and the installed packages only, so its reply
can be written when they are installed and read without a call.

## Reconstruction proxy

```bash
python -m pulserver.vre --base DIR --port N --plugins DIR [--host ADDR] [--slots N] [--gpu-slots 1] [--spares 1] [--recon-timeout S]
```

| Option | Meaning |
| --- | --- |
| `--base` | The host daemon's base directory |
| `--port` | TCP port the scanner's reconstruction client connects to |
| `--host` | Address to listen on; the loopback interface when unset, `0.0.0.0` for every interface |
| `--plugins` | Directory of reconstruction plugin files, `<plugin>.py` |
| `--slots` | Series reconstructed at once; derived from available memory and the GPUs when unset |
| `--gpu-slots` | Series reconstructed at once on each GPU when `--slots` is unset, default 1 |
| `--spares` | Worker processes started ahead of a series, default 1 |
| `--recon-timeout` | Seconds a reconstruction may run after its series ends; the worker is then terminated and the client told. Unlimited when unset |

The MRD header of each series names the design it was played from in the
`pulserver_session` and `pulserver_revision` user parameters, and the proxy
refuses a series whose header names no generated revision. The reconstruction
plugin is the one the scanner sequence names in its `recon` attribute; when it
names none, the client's config text names it, either as a bare plugin name or
under `parameters.config`. The messages and fields the client sends are listed
in {doc}`reconstruction-client`.

A series that finds every slot busy is written to
`bucket/<session>/queue/` as it arrives and reconstructed once a slot frees;
the client stays connected meanwhile.

The MRD stream is neither authenticated nor encrypted, and its header carries
patient data. The proxy belongs on the network between the scanner and the
reconstruction computer, with `--host` naming the address of the interface on
it; a scanner's reconstruction client cannot reach the loopback default.

Both services stop on `SIGINT` or `SIGTERM`. The proxy waits for the series it
is running before it exits.

## See also

* {doc}`../explanations/architecture` — the services and what passes between them.
* {doc}`../explanations/sessions` — the bucket layout and revision identity.
* {doc}`reconstruction-client` — the MRD stream of a series.
* {doc}`../api/host` and {doc}`../api/vre` — the daemon and proxy interfaces.
