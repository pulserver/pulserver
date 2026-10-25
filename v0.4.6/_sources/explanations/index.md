# Explanations

The concepts behind pulserver, one page each, in the order of a scan.
Interfaces are in the {doc}`API reference <../api/index>`, procedures in the
{doc}`user guide <../user-guide/index>`.

| Page | Contents |
| --- | --- |
| {doc}`architecture` | Design calls, reconstruction proxy and the design store |
| {doc}`protocol` | Prescription, resolved protocol, units and wire precision |
| {doc}`safety-checks` | Checks run before a design is stored, and those computed by the vendor's routines |
| {doc}`scanner-representation` | Base blocks, virtual segments, segment instances and the execution stream |
| {doc}`raw-data` | Counters, flags, encoding spaces, trajectory and field-of-view phase |
| {doc}`reconstruction-session` | MRD streaming, design lookup, workers and slots, forwarding and DICOM |
| {doc}`reconstruction` | Reconstruction units, results and the exam's calibration context |
| {doc}`virtual-scanner` | Virtual interpreter, Fourier engine and reconstruction client |

Layouts, wire formats and numerical methods are in the
{doc}`internals <../developer-guide/internals/index>`.

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
