# Examples

Executable pages, run when the documentation is built, so every figure and
every printed number on them is produced by the code as it stands.

The sections follow an acquisition through pulserver: the protocol a
prescription resolves to, the representation the scanner plays, and the raw
data returned for reconstruction.

| Section | What it covers |
| --- | --- |
| {doc}`/examples/protocol` | How the design limits determine the protocol a scanner sequence reports for a prescription. |
| {doc}`/examples/scanner-ir` | The repeating unit, segments and subsequences a sequence is reduced to for playout. |
| {doc}`/examples/reconstruction` | Enrichment of a simulated series from its sequence, and its reconstruction by a plugin. |

The concepts these pages rely on are in {doc}`/explanations/index`, and the
interfaces they call are documented in {doc}`/api/index`.

```{toctree}
:hidden:

/examples/protocol
/examples/scanner-ir
/examples/reconstruction
```
