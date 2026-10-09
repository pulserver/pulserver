# Course

Six lessons that take you from a scan run with the shipped plugins to a scan
run with your own sequence and your own reconstruction. They follow the steps
of a scan in order, so read them in order.

You need nothing installed: every lesson has an *Open in Colab* button that
runs it in your browser. The scanner is the virtual scanner throughout, so
everything you see is computed by the code on the page.

| Lesson | The question it answers |
| --- | --- |
| {doc}`1. Your first scan </generated/gallery/01-course/01_protocol_to_image>` | What does pulserver do, end to end? |
| {doc}`2. Your sequence in the scanner UI </generated/gallery/01-course/02_sequence_plugin>` | How does my sequence show up for the operator, and what happens when they edit it? |
| {doc}`3. What the scanner plays </generated/gallery/01-course/03_scanner_representation>` | What happens to my sequence before it reaches the scanner? |
| {doc}`4. What your recon receives </generated/gallery/01-course/04_reconstruction_plugin>` | What does the raw data look like when it reaches my plugin? |
| 5. Your recon plugin | How do I turn those readouts into images? |
| {doc}`6. Test your pair on the virtual scanner </generated/gallery/01-course/05_testing_on_the_virtual_scanner>` | Do my sequence and my recon work together, with real contrast? |

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
/generated/gallery/01-course/05_testing_on_the_virtual_scanner
```
