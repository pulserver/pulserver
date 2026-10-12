# Internals

Implementation design behind the {doc}`../../explanations/index`: layouts,
wire formats, schedules and numerical methods, for maintainers and interpreter
authors.

| Page | Scope |
| --- | --- |
| {doc}`calibration` | The channel model of prewhitening, coil compression and sensitivity maps, and the conditions under which stored maps are reused. |
| {doc}`ir-cache` | The prescription, the conversion passes, waves, the playout stages, the heaviest repetition, the cache file and its playback, and the C89 constraint. |
| {doc}`protocol-wire` | The RF layout, protocol keys and values, and the text blocks in which the host and the interpreter exchange them. |
| {doc}`reconstruction-proxy` | Reconstruction units and image headers, enrichment and trajectory rules, the placement of readouts in a unit's buffers, and the proxy's workers. |
| {doc}`sar-ratios` | The pTx SAR ratios computed from VOPs against a reference pulse, and their use by the interpreter. |
| {doc}`virtual-scanner` | Played waveforms and trajectory, the signal model, the coils, the Fourier engine, export to external simulators, test coverage, and the scan clock. |

```{toctree}
:hidden:

calibration
ir-cache
protocol-wire
reconstruction-proxy
sar-ratios
virtual-scanner
```
