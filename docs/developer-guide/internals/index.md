# Internals

The implementation behind the concepts of {doc}`../../explanations/index`: the
layouts, wire formats, schedules and numerical methods a contributor changing
that code has to keep.

| Page | Scope |
| --- | --- |
| {doc}`calibration` | The channel model of prewhitening, coil compression and sensitivity maps, and the conditions under which stored maps are reused. |
| {doc}`ir-cache` | Waves, the two stages of a playout, the heaviest repetition, the cache file and its playback, and the C89 constraint. |
| {doc}`protocol-wire` | The RF layout, protocol keys and values, and the text blocks in which the host and the interpreter exchange them. |
| {doc}`reconstruction-proxy` | Reconstruction units and image headers, enrichment and trajectory rules, the placement of readouts in a unit's buffers, and the proxy's workers. |
| {doc}`sar-ratios` | The SAR ratios computed from VOPs against a reference pulse, and how the interpreter uses them. |
| {doc}`virtual-scanner` | The Fourier engine's timeline, event streams, bases and grids, export to external simulators, and what a run on the virtual scanner establishes. |

```{toctree}
:hidden:

calibration
ir-cache
protocol-wire
reconstruction-proxy
sar-ratios
virtual-scanner
```
