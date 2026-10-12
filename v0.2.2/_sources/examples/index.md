# Examples

Executable pages, run when the documentation is built, so every figure and
every printed number on them is produced by the code as it stands.

The concepts these pages rely on are in {doc}`/explanations/index`, and the
interfaces they call are documented in {doc}`/api/index`.

## Course

The shortest path from a protocol to a reconstructed image, read in order.

| Lesson | Contents |
| --- | --- |
| {doc}`/generated/gallery/01-course/01_protocol_to_image` | An acquisition on the virtual scanner, from protocol to image. |
| {doc}`/generated/gallery/01-course/02_sequence_plugin` | Sequence plugin, protocol entries and `evaluate`. |
| {doc}`/generated/gallery/01-course/03_scanner_representation` | Safety checks, repetition, virtual segments and segment instances. |
| {doc}`/generated/gallery/01-course/04_reconstruction_plugin` | MRD enrichment: counters, flags, encoding spaces and trajectory. |
| {doc}`/generated/gallery/01-course/05_recon_plugin` | Reconstruction units, inverse FFT and `pics`. |
| {doc}`/generated/gallery/01-course/06_testing_on_the_virtual_scanner` | Flip-angle series against the closed-form signal. |

## Tours

Applications, advanced branches and specialised workflows, each standalone.

| Tour | Contents |
| --- | --- |
| {doc}`/generated/gallery/02-tours/01_protocol_resolution` | Bandwidth quantization and minimum echo time. |
| {doc}`/generated/gallery/02-tours/02_segmentation` | Repetition, virtual segments and subsequences; segment count against scan length. |
| {doc}`/generated/gallery/02-tours/03_fov_offset_enrichment` | Field-of-view offset in the IR cache and its phase in MRD enrichment. |

```{toctree}
:hidden:

course
tours
```
