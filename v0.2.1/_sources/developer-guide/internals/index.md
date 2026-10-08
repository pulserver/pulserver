# Internals

The implementation behind the concepts of {doc}`../../explanations/index`: the
layouts, wire formats, schedules and numerical methods a contributor changing
that code has to keep.

| Page | Scope |
| --- | --- |
| {doc}`ir-cache` | Waves, the two stages of a playout, the heaviest repetition, the cache file and its playback, and the C89 constraint. |
| {doc}`protocol-wire` | The text blocks in which the host and the interpreter exchange protocols and RF descriptions. |
| {doc}`reconstruction-proxy` | The placement of readouts in a reconstruction unit's buffers, and the proxy's workers. |
| {doc}`virtual-scanner` | The Fourier engine's timeline, event streams, bases and grids, export to external simulators, and what a run on the virtual scanner establishes. |

```{toctree}
:hidden:

ir-cache
protocol-wire
reconstruction-proxy
virtual-scanner
```
