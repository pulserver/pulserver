# MRD data

Acquisitions, header entries and images as a reconstruction plugin reads them,
and the definitions and readouts a Pulseq sequence states.

```{eval-rst}
.. currentmodule:: pulserver.mrd
```

The acquisition and header objects are those of the `ismrmrd` package; the
functions here read flags, counters and user parameters from them by their MRD
names. {class}`ReadoutTable` and {class}`SequenceDefinitions` are read from the
sequence files of a revision and are what the reconstruction proxy enriches a
stream from (see {doc}`../explanations/reconstruction`).

## Acquisitions

| Object | Description |
| --- | --- |
| {obj}`~pulserver.mrd.AcquisitionFlag` | ISMRMRD acquisition flags as bit masks, combinable with `\|`. |
| {obj}`~pulserver.mrd.has_acquisition_flag` | Whether an acquisition carries a flag. |
| {obj}`~pulserver.mrd.acquisition_label` | One encoding counter of an acquisition, by MRD field name. |
| {obj}`~pulserver.mrd.acquisition_labels` | The encoding space reference and every encoding counter of an acquisition. |
| {obj}`~pulserver.mrd.AcquisitionBucket` | Acquisitions accumulated up to a boundary, split as Gadgetron splits them. |
| {obj}`~pulserver.mrd.AcquisitionBucketStats` | Distinct values of each encoding counter in a bucket. |

## Header

| Object | Description |
| --- | --- |
| {obj}`~pulserver.mrd.MrdMetadata` | Accessors over a parsed MRD XML header. |
| {obj}`~pulserver.mrd.EncodingSpace` | Buffer layout of one encoding space. |
| {obj}`~pulserver.mrd.LOOP_COUNTERS` | Encoding counters that become buffer axes when they vary, outermost first. |
| {obj}`~pulserver.mrd.user_parameter` | A header user parameter by name. |
| {obj}`~pulserver.mrd.max_stored_value` | Largest pixel value of the header's stored bit depth. |

## Images

| Object | Description |
| --- | --- |
| {obj}`~pulserver.mrd.coil_combine` | Combine the coil axis of a stack of coil images. |
| {obj}`~pulserver.mrd.center_crop` | Crop the trailing axes of an image to a centred window. |
| {obj}`~pulserver.mrd.as_numpy` | An array as NumPy on the host; Torch tensors are detached first. |

## Sequence definitions and readouts

| Object | Description |
| --- | --- |
| {obj}`~pulserver.mrd.read_chain` | Read a sequence file and every file its `NextSequence` definitions name, in play order. |
| {obj}`~pulserver.mrd.SequenceDefinitions` | Definitions describing what a sequence acquires, in Pulseq units. |
| {obj}`~pulserver.mrd.ReadoutTable` | Every ADC readout of a sequence, in play order. |
