# Explanations

Conceptual background for the interfaces documented in {doc}`../api/index` and
used in the {doc}`user guide <../user-guide/index>`. These pages state the
components, the representations exchanged between them and the conventions the
rest of the documentation assumes.

| Explanation | What it covers |
| --- | --- |
| {doc}`architecture` | The two services beside the scanner, what each one does, and the design store they share: how a design is identified, stored, pushed and found again. |
| {doc}`protocol` | How a prescription edited in the scanner UI is resolved into the protocol a sequence plays, and the units and precision it is exchanged in. |
| {doc}`safety-checks` | What pulserver checks before a design is stored, and what it leaves to the scanner's vendor routines. |
| {doc}`scanner-representation` | How pulserver converts a Pulseq chain into the base blocks, virtual segments and segment instances the scanner plays, and what the interpreter does with them. |
| {doc}`virtual-scanner` | The stand-ins that replace the scanner in tests: the playout, the Fourier engine's signal model and the reconstruction client. |
| {doc}`raw-data` | What the proxy adds to each MRD series from the sequence that played it: counters, flags, encoding spaces, trajectory and field-of-view phase. |
| {doc}`reconstruction` | The reconstruction units a plugin receives, the images it returns, and forwarding to a reconstruction server. |
| {doc}`calibration` | Coil sensitivity maps, noise prewhitening and coil compression as transforms of the receive channels, and the conditions under which sensitivity maps are reused. |

Implementation detail -- the cache layout, the wire format, the proxy's
buffers and workers, and the virtual scanner's Fourier engine -- is in
{doc}`../developer-guide/internals/index`.

```{toctree}
:hidden:

architecture
protocol
safety-checks
scanner-representation
virtual-scanner
raw-data
reconstruction
calibration
```
