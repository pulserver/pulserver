# Course

The shortest path from a protocol to a reconstructed image through pulserver,
one stage of a scan per lesson, read in order. The first lesson runs a whole
scan on the virtual scanner; the next three follow the data through the
scanner sequence, the scanner representation and the reconstruction. The
fifth is optional.

| Lesson | What it teaches |
| --- | --- |
| {doc}`/generated/gallery/01-course/01_protocol_to_image` | A scan from protocol to image in one process: the design calls, an exam and a scan on the virtual scanner. |
| {doc}`/generated/gallery/01-course/02_sequence_plugin` | A scanner sequence: entries, presets, an evaluation of one repetition with its RF layout, validation and design. |
| {doc}`/generated/gallery/01-course/03_scanner_representation` | Conversion into the IR cache: the repetition, virtual segments, the execution stream and the grouping rule. |
| {doc}`/generated/gallery/01-course/04_reconstruction_plugin` | Enrichment of a recorded series from its design, and a reconstruction plugin run on it offline. |
| {doc}`/generated/gallery/01-course/05_testing_on_the_virtual_scanner` | Optional: the contrast of a scan on the Fourier engine against the closed-form steady state, and the check of a cache against its sequence. |

```{toctree}
:hidden:

/generated/gallery/01-course/01_protocol_to_image
/generated/gallery/01-course/02_sequence_plugin
/generated/gallery/01-course/03_scanner_representation
/generated/gallery/01-course/04_reconstruction_plugin
/generated/gallery/01-course/05_testing_on_the_virtual_scanner
```
