# Examples

Executable pages, run when the documentation is built, so every figure and
every printed number on them is produced by the code as it stands.

The concepts these pages rely on are in {doc}`/explanations/index`, and the
interfaces they call are documented in {doc}`/api/index`.

## Course

The shortest path from a protocol to a reconstructed image, read in order.

| Lesson | What it teaches |
| --- | --- |
| {doc}`/generated/gallery/01-course/01_protocol_to_image` | A whole scan on the virtual scanner, from the protocol to the image. |
| {doc}`/generated/gallery/01-course/02_sequence_plugin` | A scanner sequence and the protocol it resolves. |
| {doc}`/generated/gallery/01-course/03_scanner_representation` | The repetition, virtual segments and execution stream of a design. |
| {doc}`/generated/gallery/01-course/04_reconstruction_plugin` | Enrichment of a series and a reconstruction plugin. |
| {doc}`/generated/gallery/01-course/05_testing_on_the_virtual_scanner` | Optional: the Bloch simulation and the conversion check. |

## Tours

Applications, advanced branches and specialised workflows, each standalone.

| Tour | What it establishes |
| --- | --- |
| {doc}`/generated/gallery/02-tours/01_protocol_resolution` | Bandwidth quantization and the minimum echo time of a scanner sequence's protocol. |
| {doc}`/generated/gallery/02-tours/02_segmentation` | The repetition, virtual segments and subsequences a sequence is reduced to for playout. |
| {doc}`/generated/gallery/02-tours/03_fov_offset_enrichment` | The field-of-view offset applied to the IR, and the phase the proxy restores from the sequence. |

```{toctree}
:hidden:

course
tours
```
