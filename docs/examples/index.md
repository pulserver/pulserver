# Examples

Executable pages, run when the documentation is built, so every figure and
every printed number on them is produced by the code as it stands.

The concepts these pages rely on are in {doc}`/explanations/index`, and the
interfaces they call are documented in {doc}`/api/index`.

## Course

The shortest path from a protocol to a reconstructed image, read in order.

| Lesson | What it teaches |
| --- | --- |
| {doc}`/generated/gallery/01-course/01_protocol_to_image` | Your first scan: a whole scan on the virtual scanner, from the protocol to the image. |
| {doc}`/generated/gallery/01-course/02_sequence_plugin` | Your sequence in the scanner UI: a sequence plugin and its `evaluate`. |
| {doc}`/generated/gallery/01-course/03_scanner_representation` | What the scanner plays: the checks, the segments and what changes from one TR to the next. |
| {doc}`/generated/gallery/01-course/04_reconstruction_plugin` | What your recon receives: the raw data before and after pulserver labels it. |
| {doc}`/generated/gallery/01-course/05_recon_plugin` | Your recon plugin: a Fourier transform, then `pics` with bartorch. |
| {doc}`/generated/gallery/01-course/06_testing_on_the_virtual_scanner` | Test your pair on the virtual scanner: a flip-angle series against the closed form. |

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
