# Explanations

Conceptual background for the interfaces documented in {doc}`../api/index` and
used in the {doc}`user guide <../user-guide/index>`. These pages state the
components, the representations exchanged between them and the conventions the
rest of the documentation assumes.

| Page | Contents |
| --- | --- |
| {doc}`architecture` | Design calls, reconstruction proxy and the design store. |
| {doc}`protocol` | Prescription, resolved protocol, units and wire precision. |
| {doc}`safety-checks` | Checks run before a design is stored, and those computed by the vendor's routines. |
| {doc}`scanner-representation` | Base blocks, virtual segments, segment instances and the execution stream. |
| {doc}`raw-data` | Counters, flags, encoding spaces, trajectory and field-of-view phase. |
| {doc}`reconstruction-session` | MRD streaming, design lookup, workers and slots, forwarding and DICOM. |
| {doc}`reconstruction` | Reconstruction units, results and the exam's calibration context. |
| {doc}`virtual-scanner` | Virtual interpreter, Fourier engine and reconstruction client. |

Implementation detail -- the cache layout, the wire format, the proxy's
buffers and workers, and the virtual scanner's Fourier engine -- is in
{doc}`../developer-guide/internals/index`.

```{toctree}
:hidden:

architecture
protocol
safety-checks
scanner-representation
raw-data
reconstruction-session
reconstruction
virtual-scanner
```
