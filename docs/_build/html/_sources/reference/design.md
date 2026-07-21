# `pulserver` — sequence-building blocks

`pulserver.pypulseq` collects the reusable building blocks shared by the
example sequence plugins under [`examples/sequences/`](../../examples/sequences/).
It replaces the old per-family `_*_common.py` helpers (loaded via a
`spec_from_file_location` hack) with an installed, importable package: a
plugin imports module factories and sampling helpers directly from
`pulserver.pypulseq` even when the plugin file itself is exec-loaded
standalone by the bridge or tests.

Waveform-authoring helpers are **only** importable from `pulserver.pypulseq`.
The `pulserver` root namespace carries the plugin contract (base classes,
typed protocol parameters, protocol serialisation, `run_cli`) and nothing else.

Requires the optional `pypulseq` dependency (same tier as
`pulserver.pypulseq` / `pulserver.io`).

## Public surface

| API group | Contents |
| --- | --- |
| `pulserver.params` | Protocol-dict getters/setters, phase-FOV and ACS resolution, readout/phase axis resolution. |
| `pulserver.pypulseq.make_*_pulse` | RF excitation and magnetization-preparation module factories. |
| `pulserver.pypulseq.make_*_readout` | Cartesian, EPI, FSE/CPMG, non-Cartesian, and ZTE readout module factories. |
| `pulserver.pypulseq.make_crusher`, `make_spoiler`, `make_phase_encoding`, `make_phase_blip` | Gradient factories. |
| `pulserver.pypulseq.make_*_schedule` | RF phase, phase-cycling, and refocusing-flip schedules. |
| [Flat sampling helpers](sampling.md) | Structured mask/tilt support and acquisition ordering for Cartesian, EPI, radial/spherical non-Cartesian, slice, and SMS sampling. |
| `pulserver.run_cli` | Declarative offline CLI shared by every plugin. |
| `pulserver.SequenceModule`, `pulserver.SamplingPattern`, `pulserver.SliceGroup` | The abstract types those factories return. |

The corresponding implementation modules under `pulserver.pypulseq` use
leading underscores and are not public import locations.

## View orderings

The public package now separates sampled support from acquisition order with
`SamplingPattern`. See the [sampling reference](sampling.md) for absolute FSE
trains, relative EPI shifts, non-Cartesian tilts, and slice/SMS grouping.

The echo-train / segment view-ordering helpers apply to any acquisition
with an outer loop and an inner echo train or MPRAGE segment — 2D/3D FSE and
segmented GRE alike. They take phase-encode locations in the (ky, kz) plane
and return a list of shots (each an ordered list of view indices; the
echo/segment index is the position within the shot):

- `make_linear_order`, `make_outer_inner_order` — the sequential nested-loop orderings
  used by the current FSE/MPRAGE plugins.
- `make_fse_linear_order` — raster (kz-major) linear reordering.
- `make_fse_radial_order` — center-out radial (wedge) reordering.
- `make_fse_radial_adaptive_order` — adaptive radial reordering with per-shot
  parameter support.
- `make_fse_shuffling_order` — randomly shuffled (T2-Shuffling) reordering with
  spatial clustering to limit gradient switching.

References: Buonincontri et al., *Doubling the repetition time without paying
the price: 3D TSE with individually parameterized echo trains*, ISMRM
566-05-007 ([`refcode/Abstract 566-05-007.pdf`](../../../../refcode/)) for the
linear / radial / adaptive schemes; Tamir et al., *T2 Shuffling*, Magn Reson
Med 2017;77:180–195 ([`refcode/nihms804984.pdf`](../../../../refcode/)) for
random shuffling.

## Advanced excitation

Beyond the slice-selective, hard, and adiabatic builders,
`make_frequency_selective_pulse` creates spectrally selective pulses and the
spatial factories accept existing or generated multidimensional trajectories.
All return the common `pulserver.SequenceModule` protocol.

## API documentation rendering

The Sphinx API reference is configured in `docs/conf.py`; build it with
`sphinx-build -E -W -b html docs docs/_build/html`.
