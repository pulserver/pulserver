# recon (C++)

The reconstruction side of the file: a `.seq` chain read into trajectories,
encoding spaces, labels and a sequence description, and MRD acquisitions
enriched with them.

```cpp
#include "sequence_file_reader.h"

mrdserver::SequenceCache cache = mrdserver::read_sequence_files("scan.seq");
mrdserver::enrich_ismrmrd_header(header, cache);

auto trajectories = mrdserver::pre_compute_trajectories(cache);
mrdserver::enrich_ismrmrd_acquisition(
    acquisition, index, measurement_uid, table_position_z,
    cache, trajectories, readout_index_in_es);
```

The data source is the sequence file itself — text or binary, lazily indexed
through {cpp:class}`pulseq::SequenceFile` — rather than any scanner-written
sidecar. One file per subsequence: the lead file names its successor in the
`NextSequence` definition and the chain is walked to the end, mirroring how
the interpreter builds its collection.

````{only} not doxygen
```{note}
The reference below is generated from the headers by Doxygen, which is not
installed in this build. Everything else on this page is unaffected;
`apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## The cache

`SequenceCache` is the in-memory description a reconstruction works from: the
kshot library and the per-acquisition table, the encoding spaces with their
label limits, the per-subsequence `[DEFINITIONS]`, and — when the design side
declared `TRSize` — the sequence description.

Everything downstream consumes the struct and never touches a file.

````{only} doxygen
```{doxygenfunction} mrdserver::read_sequence_files
```

```{doxygenstruct} mrdserver::SequenceCache
:members:
```

```{doxygenstruct} mrdserver::EncodingSpace
:members:
```

```{doxygenstruct} mrdserver::Kshot
:members:
```

```{doxygenstruct} mrdserver::TrajTableEntry
:members:
```

```{doxygenstruct} mrdserver::LabelLimit
:members:
```
````

## Trajectories

The k a readout traverses is held in the logical frame, composed with the
rotation library downstream — which is what lets one stored curve serve every
acquisition that plays the same shape at a different amplitude or orientation.
`pre_compute_trajectories` materialises what fits in a budget;
`materialize_readout` produces one on demand.

````{only} doxygen
```{doxygenfunction} mrdserver::pre_compute_trajectories
```

```{doxygenfunction} mrdserver::materialize_readout
```

```{doxygenstruct} mrdserver::PrecomputedTrajectory
:members:
```
````

## Sequence description

Derived only when the subsequence carries a `TRSize` definition: the design
side states the structural TR, and without it the description is skipped
rather than re-derived. It feeds subspace reconstruction and parameter
fitting, never basic imaging.

````{only} doxygen
```{doxygenstruct} mrdserver::SequenceDescription
:members:
```

```{doxygenstruct} mrdserver::SequenceParameters
:members:
```

```{doxygenstruct} mrdserver::SeqEvent
:members:
```

```{doxygenstruct} mrdserver::RfDef
:members:
```

```{doxygenstruct} mrdserver::RfShapeTuple
:members:
```

```{doxygenstruct} mrdserver::RfShapeSamples
:members:
```

```{doxygenenum} mrdserver::AdcRole
```

```{doxygenenum} mrdserver::SeqEventType
```
````

## Enriching MRD

What the scanner sent says what was measured; the sequence file says what was
played. These put the second on the first, so a reconstruction reads one
stream and finds the trajectory, the counters and the parameters already
there.

````{only} doxygen
```{doxygenfunction} mrdserver::enrich_ismrmrd_header
```

```{doxygenfunction} mrdserver::enrich_ismrmrd_acquisition
```

```{doxygenfunction} mrdserver::add_diffusion_parameters
```

```{doxygenfunction} mrdserver::add_tensor_resource_path
```

```{doxygenfunction} mrdserver::add_gradwarp_coefficients
```

```{doxygenfunction} mrdserver::set_user_parameter_string
```

```{doxygenfunction} mrdserver::demodulate_fov_shift
```
````

### Waveforms

MRD carries a waveform stream beside the acquisitions, and that is where the
sequence description travels: header, events, RF shapes and shims, each as its
own waveform.

````{only} doxygen
```{doxygenfunction} mrdserver::add_waveform_information
```

```{doxygenfunction} mrdserver::make_physio_waveform
```

```{doxygenfunction} mrdserver::make_seqdesc_header_waveform
```

```{doxygenfunction} mrdserver::make_seqdesc_events_waveform
```

```{doxygenfunction} mrdserver::make_seqdesc_rf_shapes_waveform
```

```{doxygenfunction} mrdserver::make_seqdesc_shims_waveform
```
````

## See also

{doc}`../python/recon` is the Python reconstruction stack these acquisitions
arrive in, and `decode_sequence_description` there is the other end of the
waveforms written here.
