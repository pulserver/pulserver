# The sequence model

What a sequence *is* on each side of the download, and how the three views
stay one representation.

On the scanner side, a sequence is a segmented block table with a detected
repeating unit — a {doc}`PulSeg reading <pulseg_representation>` whose
{doc}`TR and segments are found in the content <tr_and_segmentation>` rather
than annotated by the designer. On the design side, it is a
{doc}`service the console asks <design_service>`: a Python program on a
compiled sequence core, answering protocol edits in interactive time. On the
reconstruction side, it is an {doc}`MRD stream <mrd_architecture>` that
arrives already carrying its encoding counters, trajectory and sequence
description, and a server that reconstructs it with the exam, rather than the
connection, as its unit of context.

```{toctree}
:maxdepth: 1

pulseg_representation
tr_and_segmentation
design_service
mrd_architecture
```
