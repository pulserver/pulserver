# Course

Six lessons that take you from a scan run with the shipped plugins to a scan
run with your own sequence and your own reconstruction. They follow the steps
of a scan in order, so read them in order.

You need nothing installed: every lesson has an *Open in Colab* button that
runs it in your browser. The scanner is the virtual scanner throughout, so
everything you see is computed by the code on the page.

| Lesson | Contents |
| --- | --- |
| {doc}`1. Your first scan </generated/gallery/01-course/01_protocol_to_image>` | An acquisition on the virtual scanner, from protocol to image. |
| {doc}`2. Your sequence in the scanner UI </generated/gallery/01-course/02_sequence_plugin>` | Sequence plugin, protocol entries and `evaluate`. |
| {doc}`3. What the scanner plays </generated/gallery/01-course/03_scanner_representation>` | Safety checks, repetition, virtual segments and segment instances. |
| {doc}`4. What your recon receives </generated/gallery/01-course/04_reconstruction_plugin>` | MRD enrichment: counters, flags, encoding spaces and trajectory. |
| {doc}`5. Your recon plugin </generated/gallery/01-course/05_recon_plugin>` | Reconstruction units, inverse FFT and `pics`. |
| {doc}`6. Test your pair on the virtual scanner </generated/gallery/01-course/06_testing_on_the_virtual_scanner>` | Flip-angle series against the closed-form signal. |

By the end you have two files, a sequence plugin and a recon plugin, that run
unchanged on a real scanner through its interpreter
({doc}`/user-guide/running`).

If you know Gadgetron, lessons 4 and 5 map each of its pieces to pulserver's.

```{toctree}
:hidden:

/generated/gallery/01-course/01_protocol_to_image
/generated/gallery/01-course/02_sequence_plugin
/generated/gallery/01-course/03_scanner_representation
/generated/gallery/01-course/04_reconstruction_plugin
/generated/gallery/01-course/05_recon_plugin
/generated/gallery/01-course/06_testing_on_the_virtual_scanner
```
