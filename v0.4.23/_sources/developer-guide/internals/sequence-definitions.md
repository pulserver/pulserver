# Sequence definitions

The `[DEFINITIONS]` entries of a Pulseq file that pulserver and the scanner
interpreters read or write. A definition name is CamelCase; the `snake_case`
keys of a limits block and of an interpreter's configuration are not
definitions. Values are numbers unless stated; a vector is several numbers
separated by spaces.

## Design and checks

| Definition | Written by | Read by | Meaning |
| --- | --- | --- | --- |
| `NextSequence` | {meth}`~pulserver.design.SequencePlugin.design` | the checks, the conversion, the proxy | File name of the next sequence of the chain, played after this one |
| `TotalDuration` | pypulseqpp | the conversion, the proxy | Duration of the sequence in s |
| `EnablePmc` | the sequence function | the design service, the checks, the conversion, the proxy | 1 when the scan follows the poses its reconstruction publishes; the design is then made and checked for every rotation where the site follows poses |
| `EnableSarBurstMode` | the sequence function | the conversion, the interpreter | 1 to ask the scanner for SAR burst limits; the interpreter reads it from the file before it is converted |
| `StrictLimits` | the design service, for a strict plugin; or the sequence function | the design service | 1 when a design past a limit is refused as designed, never designed again at a lower slew rate |
| `DesignMaxSlew` | the design service | — | Slew rate per logical axis the sequence was designed under, in T/m/s, after any lowering for nerve stimulation or slew rate |
| `PrescanOnly` | the conversion, on the receive-gain calibration it generates from the chain | the conversion, the interpreter | 1 for the subsequence a prescan plays and the scan does not |
| `NumGainCalibrationReadouts` | the sequence function, or the conversion on its calibration | the conversion | Readouts the receive-gain prescan acquires from the first block, a whole number of repetitions; through the first repetition that acquires when not stated |

## Prescription and reconstruction

| Definition | Written by | Read by | Meaning |
| --- | --- | --- | --- |
| `Name` | the sequence function | the conversion | Name of the sequence, a string |
| `FOV` | the sequence function | the conversion, the proxy | Field of view `x y z` in m |
| `Matrix` | the sequence function | the conversion, the proxy | Encoded matrix `x y z` |
| `NavFOV`, `NavMatrix` | the sequence function | the conversion, the proxy | Field of view and matrix of the navigator readouts |
| `SlicePositions` | the sequence function | the proxy | Position of each slice the `SLC` label indexes, in m from the field-of-view centre |
| `TE` | the sequence function | the shipped evaluation, the proxy | Echo time in s, written to the MRD header; one per echo for a multi-echo sequence |
| `EffectiveEchoTime` | the sequence function | the shipped evaluation, the proxy | Contrast-equivalent echo time of a variable-refocusing train in s, shown in place of `TE` |
| `TR` | the sequence function | the shipped evaluation, the proxy | Repetition time in s, written to the MRD header |
| `TI` | the sequence function | the proxy | Inversion time in s, written to the MRD header |
| `EchoSpacing` | the sequence function | the shipped evaluation | Echo spacing in s |
| `SliceThickness` | the sequence function | the shipped evaluation | Slice thickness in m |
| `ReconParameters` | {func}`~pulserver.design.recon_parameters` | the proxy | Names of the definitions copied into the MRD header as user parameters, such as `QsmAnatomy` |

A sequence function writes further definitions for its own reconstruction or
for record, such as `TRPeriphery` and `EchoTrainLengthPeriphery` of a
dual-region fast spin echo; the services do not read them.
