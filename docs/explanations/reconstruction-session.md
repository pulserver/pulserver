# The reconstruction session

```{admonition} TL;DR
:class: tldr

- The proxy speaks Gadgetron's MRD streaming messages: one series per connection, images back on the same one.
- The client's config names the recon plugin, so one design can be reconstructed by any plugin.
- Each series runs in its own worker process, queued on disk when every slot is busy, or is forwarded to a reconstruction server.
```

The scanner's reconstruction client opens a connection per series, sends a
config, the MRD header and the acquisitions, and reads images back. The proxy
on the other end refuses a series it cannot serve, enriches it from its design
({doc}`raw-data`), and runs your recon plugin on it. Lesson 4 of the
{doc}`Course <../examples/course>` runs one series through it.

## What pulserver does

:::{container} capabilities

- **Accepts one series per connection and returns its images, then closes.**

  Code: {class}`~pulserver.proxy.ReconProxy`. Tests: *a series returns images then closes*; *a connection that sends nothing is not a failed series* (`test_proxy.py`).
- **Runs the plugin the client's config names, by name or by the path of its file.**

  Code: {class}`~pulserver.proxy.ReconProxy`. Tests: *the proxy resolves a plugin file path to its stem*; *a series opening with its header names no reconstruction and is refused*; *a manifest naming a reconstruction is read and the name ignored* (`test_proxy.py`).
- **Reads each design once from the store, or from its own intake, and refuses a series naming none.**

  Code: {class}`~pulserver.proxy.DesignCache`, {class}`~pulserver.proxy.DesignIntake`. Tests: *a design pushed to the intake is reconstructed from its store*; *a series naming no stored design is refused* (`test_proxy.py`).
- **Refuses a malformed stream with a text the console shows.**

  Code: {class}`~pulserver.proxy.ReconProxy`. Tests: *a series carrying no header or two is refused*; *a series that ends before its last readout is refused*; *a message the proxy cannot read ends the output with its type* (`test_proxy.py`).
- **Reconstructs each series in a fresh worker process, with spares that have already imported the engine.**

  Code: {class}`~pulserver.proxy.ReconProxy`. Tests: *the spare worker is replaced after each series*; *the proxy process does not import the reconstruction engine*; *a crashing plugin closes its series and frees the slot* (`test_proxy.py`).
- **Bounds concurrent series by slots, one per GPU by default, and holds the rest on disk.**

  Code: `--slots`, `--gpu-slots`, `--queue`. Tests: *a second series waits for a slot and still returns images*; *a series with every slot busy is held on disk until one frees*; *a series reads the gpu its slot holds* (`test_proxy.py`).
- **Stops a reconstruction past its timeout and tells the client.**

  Code: `--recon-timeout`. Tests: *a reconstruction past the recon timeout is stopped and reported*; *a reconstruction outlasting the worker timeout still returns its image* (`test_proxy.py`).
- **Returns images as MRD images, or as DICOM.**

  Code: `--dicom`. Tests: *images come back as dicom when asked*; *forwarded images come back as dicom when asked* (`test_proxy.py`).
- **Forwards each enriched series to a reconstruction server instead of reconstructing it.**

  Code: `--forward`, {class}`~pulserver.proxy.ReconServer`. Tests: *a forwarded series returns the image a local worker returns*; *a forwarded series names its reconstruction in a config file message* (`test_proxy.py`).
- **Shares an exam cache between the series of one exam, across proxies on one host.**

  Code: `--exams`. Tests: *the series of one exam share its exam cache*; *a closed proxy leaves the exam root for the proxies still running* (`test_proxy.py`).
- **Publishes the poses of a motion-corrected series to the running scan.**

  Code: `pulserver.proxy._motion`. Tests: *a motion corrected series publishes its poses to the scan*; *a series not corrected for motion writes no pose file* (`test_proxy.py`).
- **Keeps each series as the scanner sent it, for an offline rerun that gives the same image.**

  Code: `--save-data`. Tests: *a kept series runs offline and gives the image the proxy returned*; *what a proxy keeps is the header the scanner sent* (`test_proxy.py`).
- **Holds images for a client that dropped, and closes on its own when idle.**

  Code: `--idle-timeout`. Tests: *images whose client has gone reach it when it returns*; *the idle timeout counts from the last client leaving* (`test_proxy.py`).

:::

## What the scanner and your plugin do

- **The client** sends the series and shows what comes back ({doc}`../user-guide/reconstruction-client`).
- **Your recon plugin** turns the series into images ({doc}`reconstruction`).

## How it works

```{figure} ../_static/session.svg
:figclass: only-light

A series streamed to the proxy, enriched, and reconstructed in a worker or forwarded.
```

```{figure} ../_static/session-dark.svg
:figclass: only-dark

A series streamed to the proxy, enriched, and reconstructed in a worker or forwarded.
```

### Messages

A series is `CONFIG`, `HEADER`, the `ACQUISITION`s and optional `WAVEFORM`s,
then `CLOSE`; what returns is `IMAGE`, `DICOM_WITHNAME` and `TEXT` messages,
then `CLOSE`. These are the identifiers of Gadgetron's MRD streaming protocol.
A refused series receives one `TEXT` beginning `pulserver:` and a `CLOSE` at
once. An MRD message carries no length, so a message type the proxy cannot
read ends the series.

### Workers and slots

A worker reconstructs one series and exits, which frees whatever memory the
reconstruction allocated; the import cost is paid by a spare started in
advance. Slots are derived from memory and the GPUs found through
`CUDA_VISIBLE_DEVICES` or `nvidia-smi`. A series without a slot is written to
the queue, enriched, as it arrives, and the client stays connected until its
images come back. {doc}`../developer-guide/internals/reconstruction-proxy`
states the details.

### Forwarding

With `--forward`, the proxy enriches each series and sends it to an MRD server
on another computer, with a config naming the plugin or `--forward-config`. A
{class}`~pulserver.proxy.ReconServer` runs the plugin in its own workers and
enriches nothing.

## See it run

- {doc}`../generated/gallery/01-course/04_reconstruction_plugin`: one series through the proxy.
- {doc}`../user-guide/running`: the proxy's options.
- {doc}`../user-guide/reconstruction-client`: the messages and header a client sends.
- {doc}`../api/proxy`: {class}`~pulserver.proxy.ReconProxy` and {class}`~pulserver.proxy.ReconServer`.
