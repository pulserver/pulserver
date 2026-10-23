# Internals

Implementation design behind the {doc}`../../explanations/index`: layouts,
wire formats, schedules and numerical methods, for maintainers and interpreter
authors.

| Page | Contents |
| --- | --- |
| {doc}`calibration` | Prewhitening, coil compression, sensitivity maps and the reuse of stored maps |
| {doc}`ir-cache` | Conversion passes, waves, playout stages, the gradient stretches, the cache file and the C89 library |
| {doc}`protocol-wire` | RF layout, protocol keys and values, and the protocol text blocks |
| {doc}`reconstruction-proxy` | Reconstruction units, image headers, enrichment and trajectory rules, readout placement and workers |
| {doc}`sar-ratios` | pTx SAR ratios from VOPs against a reference pulse |
| {doc}`virtual-scanner` | Played trajectory, signal model, coils, Fourier engine, export, test coverage and scan clock |

```{toctree}
:hidden:

calibration
ir-cache
protocol-wire
reconstruction-proxy
sar-ratios
virtual-scanner
```
