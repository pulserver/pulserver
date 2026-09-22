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
`NextSequence` chain are copied into a revision and converted to the IR cache
there.

Plugin code runs in spawned worker processes, so a plugin that crashes fails
the command it was running and the daemon replaces the pool. A plugin file is
imported again when its modification time changes.

## Reconstruction proxy

```bash
python -m pulserver.vre --base DIR --port N --plugins DIR [--slots N] [--spares 1]
```

| Option | Meaning |
| --- | --- |
| `--base` | The host daemon's base directory |
| `--port` | TCP port the scanner's reconstruction client connects to |
| `--plugins` | Directory of reconstruction plugin files, `<plugin>.py` |
| `--slots` | Series reconstructed at once; derived from available memory when unset |
| `--spares` | Worker processes started ahead of a series, default 1 |

The MRD header of each series names the design it was played from in the
`pulserver_session` and `pulserver_revision` user parameters, and the proxy
refuses a series whose header names no generated revision. The reconstruction
plugin is the one the scanner sequence names in its `recon` attribute; when it
names none, the client's config text names it, either as a bare plugin name or
under `parameters.config`.

A series that finds every slot busy is written to
`bucket/<session>/queue/` as it arrives and reconstructed once a slot frees;
the client stays connected meanwhile.

Both services stop on `SIGINT` or `SIGTERM`. The proxy waits for the series it
is running before it exits.
