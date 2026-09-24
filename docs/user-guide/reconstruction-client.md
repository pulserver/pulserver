# Reconstruction clients

The reconstruction client is the scanner-side program that streams the raw
data of each series to the reconstruction proxy and returns the images to the
console. It connects to the proxy's TCP port ({doc}`running`), sends one series
per connection as MRD messages, and reads what the reconstruction returns on
the same connection.

The proxy replaces the encoding counters, flags, encoding spaces and trajectory
of the series with those the sequence states ({doc}`../explanations/reconstruction`).
The client therefore sends what the scanner measured and the identity of the
design the series was played from, and leaves the Pulseq structure of the scan
to the proxy.

## Messages

A series is sent in this order:

| Message | Identifier | Content |
| --- | --- | --- |
| `CONFIG` | 2 | Optional. The reconstruction plugin, as a bare name or under `parameters.config` of a JSON, YAML or XML text, used when the design names none |
| `HEADER` | 3 | The MRD XML header, once |
| `ACQUISITION` | 1008 | One per readout of the sequence chain, in play order |
| `WAVEFORM` | 1026 | Optional, anywhere after the header: physiological waveforms, passed to the reconstruction unchanged |
| `CLOSE` | 4 | The end of the series |

The client then reads until the proxy's `CLOSE`: images (`IMAGE`, 1022),
DICOM datasets (`DICOM_WITHNAME`, 1018) and texts (`TEXT`, 5), in the order the
reconstruction produces them. A text beginning `pulserver:` reports a series
that failed. A refused series receives that text and a `CLOSE` at once; the
proxy then discards what the client still sends until it closes the
connection, so the client can finish sending and read the reason.

## Header

| Element | Requirement | Use |
| --- | --- | --- |
| `pulserver_design`, a `userParameterString` | Required | The identifier of the design the interpreter plays, from the `GENERATED` or `IMPORTED` reply: 18 hexadecimal digits, which are three 24-bit integers of six digits each |
| `experimentalConditions.H1resonanceFrequency_Hz` | Required by the MRD schema | DICOM `ImagingFrequency` |
| `acquisitionSystemInformation.receiverChannels` | Optional | The coil count of a reconstruction buffer laid out before its first acquisition |
| `ExamID`, a `userParameterString`, or else `studyInformation.studyInstanceUID`, or else `studyInformation.studyID` | Optional | The exam; the series of one exam share its exam cache, and a series without one is an exam of its own |
| `subjectInformation`, `studyInformation`, `measurementInformation`, `acquisitionSystemInformation` | Optional | Copied into the DICOM datasets; `relativeTablePosition` is not |
| `measurementInformation.measurementID` | A non-negative integer when `systemVendor` names GE | The DICOM series number on a GE system |
| `encoding` | Optional | Replaced by one encoding space per subsequence, and one more for the navigator readouts of a subsequence that has them; a matrix size or field of view the sequence does not define is kept from the client's encoding at the same index |
| `sequenceParameters` | Optional | TR, TE, TI and flip angle replaced by those the sequence defines |

A header that the ISMRMRD schema does not accept, or that names no design of
the store, is refused before any acquisition is read.

## Acquisitions

The proxy matches acquisitions to readouts by their position in the stream, so
the client sends every readout the sequence chain plays, those of dummy scans,
noise scans and navigators included, and no other.

| Field | Client | Proxy |
| --- | --- | --- |
| `number_of_samples` | The sample count of the readout's ADC event | Checked |
| `active_channels`, data | The samples, demodulated to the prescribed field-of-view centre by each ADC's frequency and phase offsets and its phase modulation | Passed on unchanged |
| `scan_counter` | 0 on every acquisition, or increasing by one from any start | Checked |
| `flags` | None, or `LAST_IN_MEASUREMENT` on the last acquisition | Replaced |
| `idx`, `sample_time_us`, `encoding_space_ref` | Any value | Replaced |
| `center_sample` | Any value | Replaced, except on a readout whose k-space location does not change |
| `trajectory_dimensions`, `traj` | Any value | Replaced on a readout whose k-space location changes, in 1/m along the sequence's gradient axes |
| `measurement_uid`, `position`, `read_dir`, `phase_dir`, `slice_dir`, `patient_table_position`, `acquisition_time_stamp`, `physiology_time_stamp`, `user_int`, `user_float` | The scanner's values, as MRD defines them | Passed on unchanged; an image takes them from its reference acquisition |

A series is refused, and its reconstruction stopped, when an acquisition has a
sample count other than its readout's, when the scan counters skip or repeat
one, when an acquisition before the last is flagged `LAST_IN_MEASUREMENT`, or
when the stream carries more or fewer acquisitions than the chain plays.

## See also

* {doc}`running` — starting the proxy.
* {doc}`../explanations/reconstruction` — enrichment and routing.
* {doc}`../explanations/designs` — the design store and the identifier of a design.
* {class}`~pulserver.vre.ReconProxy` — the proxy.
