# `pulserver.pypulseq`

A drop-in replacement for `pypulseq`. The complete upstream namespace is
re-exported unchanged, so this import covers the whole event layer and
`import pypulseq` alongside it is never necessary:

```python
import pulserver.pypulseq as pp

seq = pp.Sequence(pp.Opts())
seq.add_block(pp.make_delay(1e-3))
```

Everything upstream documents is available here under the same name and with
the same behaviour — see the [PyPulseq API
reference](https://pypulseq.readthedocs.io/) for it. This page documents only
the **difference**: the objects Pulserver replaces, and the extension events
upstream has no equivalent for. Programmatically that set is
`pulserver.pypulseq.OVERRIDES`; the rest is `pulserver.pypulseq.UPSTREAM`.

This namespace is the event layer *only*. The factories that assemble events
into whole modules and scan loops — RF pulses, readouts, sampling plans, phase
schedules — live in {doc}`pulserver.design <design>`.

A few upstream names are deliberately withheld. `make_adiabatic_pulse` is not
re-exported, because Pulserver's inversion and preparation modules design
their adiabatic pulses internally. `compress_shape`, `decompress_shape` and
`convert` are not re-exported either: upstream exposes them as modules rather
than as part of its authoring vocabulary, and nothing a plugin writes should
need to reach the shape codec directly.

## Replaced objects

`Sequence` subclasses upstream's with a lower per-block overhead. `Opts` uses
zero dead/ringdown times and vendor-neutral 2 us RF/ADC and 10 us
gradient/block rasters — scanner-specific code should still pass explicit
limits. `make_label` accepts user-defined label strings the upstream set does
not cover, and `get_supported_labels` documents every label and what it means.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.Opts
   pulserver.pypulseq.Sequence
   pulserver.pypulseq.get_supported_labels
   pulserver.pypulseq.make_label
```

Timing is validated with `Sequence.check_timing()`.

## Extension events

Pulseq extension objects with no upstream factory. Both are ordinary block
events: pass them to `seq.add_block` like any other.

```{eval-rst}
.. autosummary::
   :toctree: generated/pypulseq
   :nosignatures:

   pulserver.pypulseq.make_rf_shim
   pulserver.pypulseq.make_rotation
```

## Label constants

The Pulseq label set splits in two. **Counters** say where an acquisition
belongs and come from a scan loop's axes — {meth}`~pulserver.ScanLoop.labels`
builds the events, {meth}`~pulserver.SequenceModule.set_labels` places them.
**Flags** say how a block is played or classified and come from
{meth}`~pulserver.SequenceModule.set_flags`, which scopes them to the module
unless they are in `STICKY_FLAGS`.

`COUNTER_LABELS`, `FLAG_LABELS` and `STICKY_FLAGS` are that split as module
constants: the ten ISMRMRD `EncodingCounters` fields, everything else, and the
flags that outlive the module which set them.
