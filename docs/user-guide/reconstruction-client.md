# Reconstruction client

The reconstruction client is the scanner-side program that streams the raw
data of each series to the reconstruction proxy and returns the images to the
console. It connects to the proxy's TCP port ({doc}`running`), sends one series
per connection as MRD messages with the message identifiers of Gadgetron's
streaming protocol, and reads what the reconstruction returns on the same
connection.

The proxy replaces the encoding counters, flags, encoding spaces and trajectory
of the series with those the sequence states ({doc}`../explanations/raw-data`).
Your client therefore sends what the scanner measured, the identifier of the
design the series was played from and the reconstruction plugin to run, and
leaves the Pulseq structure of the scan to the proxy.

## Messages

A series is sent in this order:

| Message | Identifier | Content |
| --- | --- | --- |
| `CONFIG_FILE` or `CONFIG_TEXT` | 1 or 2 | The reconstruction plugin: a config file message holding its name, or a config text holding a bare name or the path of its `<plugin>.py` file, or either under `parameters.config` of a JSON, YAML or XML text |
| `HEADER` | 3 | The MRD XML header, once |
| `ACQUISITION` | 1008 | One per readout of the sequence chain, in play order |
| `WAVEFORM` | 1026 | Optional, anywhere after the header: physiological waveforms, passed to the reconstruction unchanged |
| `CLOSE` | 4 | The end of the series |

The design names no reconstruction, so the proxy refuses a series whose config
names none, or that sends no config, unless it forwards with a
`--forward-config`. A path ending in `.py` names the plugin by the file's stem;
the proxy searches its own `--plugins` directories for that name, whatever the
path's directory.

The client then reads until the proxy's `CLOSE`: images (`IMAGE`, 1022),
DICOM datasets (`DICOM_WITHNAME`, 1018) and texts (`TEXT`, 5), in the order the
reconstruction produces them. A text beginning `pulserver:` reports a series
that failed. A refused series receives that text and a `CLOSE` at once; the
proxy then discards what the client still sends until it closes the
connection, so the client can finish sending and read the reason.

## Header

| Element | Requirement | Use |
| --- | --- | --- |
| `pulserver_scan`, a `userParameterString` | Required | The scan the design is stored for: the `--scan` of the design call, one directory name of the design store |
| `pulserver_design`, a `userParameterString` | Required | The identifier of the design the interpreter plays, from the `GENERATED` or `IMPORTED` reply: 18 hexadecimal digits, which are three 24-bit integers of six digits each |
| `experimentalConditions.H1resonanceFrequency_Hz` | Required by the MRD schema | DICOM `ImagingFrequency` |
| `acquisitionSystemInformation.receiverChannels` | Optional | The coil count of a reconstruction buffer laid out before its first acquisition |
| `ExamID`, a `userParameterString`, or else `studyInformation.studyInstanceUID`, or else `studyInformation.studyID` | Optional | The exam; the series of one exam share its exam cache, and a series without one is an exam of its own |
| `fov_offset_mm`, a `userParameterString` | Optional | The position of the field-of-view centre, three millimetres along the logical readout, phase and slice axes separated by spaces; a difference from the offset the design was converted at is applied to the samples as a receive phase. Without it, the converted offset |
| `subjectInformation`, `studyInformation`, `measurementInformation`, `acquisitionSystemInformation` | Optional | Copied into the DICOM datasets; `relativeTablePosition` is not |
| `measurementInformation.measurementID` | A non-negative integer when `systemVendor` names GE | The DICOM series number on a GE system: of the first series the reconstruction returns, and `100 × measurementID + k` of the `k`-th after it |
| `dicom:<attribute>`, a `userParameterString` | Optional | A DICOM attribute of every image ({ref}`dicom-attributes`) |
| `encoding` | Optional | Replaced by one encoding space per subsequence, and one more for the navigator readouts of a subsequence that has them. The reconstruction space is the sequence's matrix size and field of view, one slice deep for a stack of 2D slices; the encoded space of a Cartesian space has the readout widened to the full echo, oversampling included, and its field of view in proportion. The encoding limits are the extremes of the counters the readouts reach, centred on the k-space centre line and partition the sequence defines. A matrix size or field of view the sequence does not define is kept from the client's encoding at the same index |
| `sequenceParameters` | Optional | TR, TE, TI and flip angles replaced by those the sequence defines; TR and TE it does not define are measured by pypulseqpp's `test_report_dict`, flip angles it does not define are the distinct values of `rf_flip_angles` |

A header that the ISMRMRD schema does not accept, or that names no design the
store holds for its scan, is refused before any acquisition is read.

(dicom-attributes)=
### DICOM attributes

A client states the DICOM attributes the MRD header has no element for as
`userParameterString` entries named `dicom:<attribute>`. The DICOM builder
reads them once, with the rest of the header, into the attributes every image
of the series starts from; an entry whose value names a quantity of the image
is filled in for each image instead. An image's own meta attributes take
precedence over both. `<attribute>` is one of:

| Form | Example | Resolved by |
| --- | --- | --- |
| A keyword of the DICOM standard | `dicom:ImagedNucleus` = `23Na` | pydicom's dictionary |
| An explicit tag and value representation | `dicom:(0027,1041)FL` = `{location}` | Itself; a private group's creator is one more entry, `dicom:(0027,0010)LO` |
| A keyword of a private dictionary | `dicom:InternalPulseSeqName` = `pulserver` | The file `dicom:dictionary` names, in MATLAB's format, one `(gggg,eeee) VR Keyword VM` line per attribute |

A value separates multiple values with `\` and may name, in braces, a quantity
of the image it is written into; `{{` and `}}` are literal braces:

| Name | Value |
| --- | --- |
| `slice_location` | The image centre along the plane normal, in mm: the `SliceLocation` written by default |
| `location` | The image centre along the patient axis the normal lies closest to, in mm, positive towards right, anterior or superior; for an oblique plane, along the normal turned towards those |
| `location_letter` | The side `location` lies on: `R` or `L`, `A` or `P`, `S` or `I`, and `+` or `-` for an oblique plane |
| `instance_number`, `series_number` | The image's `InstanceNumber` and `SeriesNumber` |

An attribute that resolves to no tag, for want of the dictionary, is logged and
left out. `pulserver replay --dicom-dictionary` names another dictionary for a
series replayed where the header's is not.

## Acquisitions

The proxy matches acquisitions to readouts by their position in the stream, so
your client sends every readout the sequence chain plays, those of dummy scans,
noise scans and navigators included, and no other.

| Field | Client | Proxy |
| --- | --- | --- |
| `number_of_samples` | The sample count of the readout's ADC event | Checked |
| `active_channels`, data | The samples, demodulated by each ADC's frequency and phase offsets | Multiplied by the phase a field-of-view offset adds to a readout under a moving gradient, and by any phase modulation the sequence file stores |
| `scan_counter` | 0 on every acquisition, or increasing by one from any start | Checked |
| `flags` | None, or `LAST_IN_MEASUREMENT` on the last acquisition | Replaced |
| `idx`, `sample_time_us`, `encoding_space_ref` | Any value | Replaced |
| `center_sample` | Any value | Replaced, except on a readout whose k-space location does not change |
| `trajectory_dimensions`, `traj` | Any value | Replaced on a readout whose k-space location changes, in 1/m along the sequence's gradient axes |
| `position` | The prescribed field-of-view centre, in mm from isocentre | Moved along `slice_dir` to the centre of the readout's slice, for a stack of 2D slices whose sequence defines `SlicePositions`; an image takes it from its reference acquisition |
| `measurement_uid`, `read_dir`, `phase_dir`, `slice_dir`, `patient_table_position`, `acquisition_time_stamp`, `physiology_time_stamp`, `user_int`, `user_float` | The scanner's values, as MRD defines them | Passed on unchanged; an image takes them from its reference acquisition |

A series is refused, and its reconstruction stopped, when an acquisition has a
sample count other than its readout's, when the scan counters skip or repeat
one, when an acquisition before the last is flagged `LAST_IN_MEASUREMENT`, or
when the stream carries more or fewer acquisitions than the chain plays.

## See also

* {doc}`running`: starting the proxy.
* {doc}`../explanations/raw-data`: MRD enrichment.
* {doc}`../explanations/reconstruction-session`: the messages, workers and forwarding.
* {doc}`../explanations/architecture`: the design store and the identifier of a design.
* {class}`~pulserver.proxy.ReconProxy`: the proxy.
