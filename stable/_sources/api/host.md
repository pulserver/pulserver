# Design calls

The design calls of the interpreter host processes of one scanner, and the
store of the designs they generate.

```{eval-rst}
.. currentmodule:: pulserver.host
```

An interpreter host process lists, validates, generates and imports through
the `pulserver design` command, or `python -m pulserver.host`, answered in its
own process or in a warm server's; the calls, their replies and the store are
described in {doc}`../user-guide/running`, and the identity of a design in
{doc}`../explanations/designs`.

## Calls

| Object | Description |
| --- | --- |
| {obj}`~pulserver.host.call` | Answer one design call in the calling process: its exit status and reply. |

## Designs

| Object | Description |
| --- | --- |
| {obj}`~pulserver.host.DesignStore` | The designs under one directory, each in `<id>/` and immutable once written. |
| {obj}`~pulserver.host.design_identity` | SHA-256 of what a design depends on: plugin, limits, resolved protocol and source. |
| {obj}`~pulserver.host.design_id` | Identifier of a design: the first 18 hexadecimal digits of its identity. |
