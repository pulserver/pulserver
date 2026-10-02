# Validation

Checking that a sequence plays as it was written: the gradients it asks for
against a recording of a machine playing it, or against the waveforms its own
cache holds.

```{eval-rst}
.. currentmodule:: pulserver.validate
```

The two comparisons answer different questions. A recording establishes that
the machine plays the sequence, end to end through the conversion, the
interpreter and the hardware. The cache establishes only that the conversion
kept the sequence: a conversion and a playout that are wrong in the same way
agree with each other. {attr}`~pulserver.validate.Comparison.reference` says
which was answered, and the command prints it.

A machine drives its transmit and its gradient channels on separate timelines
and records each as it drove it, so the two are offset in a recording by that
machine's own constant. `shift_us` states it. Nothing here estimates it: a
comparison that quietly aligns two waveforms can align away the disagreement it
exists to find.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.validate.validate` | Check a sequence against a recording of it, or against its own cache. |
| {obj}`~pulserver.validate.Comparison` | What a comparison established, and what it compared against. |
| {obj}`~pulserver.validate.ChannelAgreement` | How far apart one channel's two renderings are. |
| {obj}`~pulserver.validate.read_waveform_xml` | Read the waveform XML a scanner's plotter writes. |
| {obj}`~pulserver.validate.PlayedWaveforms` | What a scanner played, as its plotter recorded it. |
| {obj}`~pulserver.validate.VENDORS` | The machines a recording can be read from. |
