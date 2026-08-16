# `pulserver.sequences`

Ready-to-run sequence callbacks for a Python REPL. They take explicit,
keyword-only sequence controls and return an in-memory
`pulserver.pypulseq.Sequence`; no bridge protocol dictionary or output path is
part of this API. The bridge plugins under `examples/sequences/` are the
single implementation: these callbacks invoke those plugins without their
final write side effect. A returned sequence may therefore be inspected,
combined by an application, or written explicitly with `pulserver.io.write`.

```python
from pulserver.sequences import design_gre_2d

seq = design_gre_2d(te_ms=8.0, tr_ms=250.0, nx=128, ny=128)
```

`pulserver.sequence` is a singular compatibility alias for this namespace.

## Nothing is shipped yet

The namespace is empty. The plugins these callbacks wrap are being rebuilt on
the current {doc}`design <design>` classes, and each returns here as it is
ported.

To build a sequence today, compose the design classes yourself — that is what
a plugin does, and {doc}`../getting_started` is a complete example that ends
in a written file.
