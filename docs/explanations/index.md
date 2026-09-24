# Explanations

Conceptual background for the interfaces documented in {doc}`../api/index` and
used in the {doc}`user guide <../user-guide/index>`. These pages state the
components, the representations exchanged between them and the conventions the
rest of the documentation assumes.

| Explanation | What it covers |
| --- | --- |
| {doc}`architecture` | The four layers of an acquisition through pulserver, the service each machine runs, and the representation exchanged at each boundary. |
| {doc}`protocol` | How a prescription edited in the scanner UI is resolved into the protocol a sequence plays, and the units and precision it is exchanged in. |
| {doc}`designs` | The identity of a design, the design store, and how the reconstruction side finds a design. |
| {doc}`ir-cache` | The segmented representation of a sequence that a scanner interpreter plays, and the passes that compute it. |
| {doc}`reconstruction` | Enrichment of the raw data from the sequence that played it, and the routing of each series to a reconstruction. |
| {doc}`virtual-scanner` | The stand-ins that replace the scanner in tests: the playout, the physics and the reconstruction client, and what a run establishes. |

```{toctree}
:hidden:

architecture
protocol
designs
ir-cache
reconstruction
virtual-scanner
```
