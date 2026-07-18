# `pulserver.design` — sequence-building blocks

`pulserver.design` collects the reusable building blocks shared by the
example sequence plugins under [`examples/sequences/`](../../examples/sequences/).
It replaces the old per-family `_*_common.py` helpers (loaded via a
`spec_from_file_location` hack) with an installed, importable subpackage: a
plugin now writes `from pulserver.design import readout, sampling, ...`
even when the plugin file itself is exec-loaded standalone by the bridge or
tests.

Requires the optional `pypulseq` dependency (same tier as
`pulserver.pulseq` / `pulserver.io`).

## Modules

| Module | Contents |
| --- | --- |
| `pulserver.design.system` | System-limit derates, readout-timing quantization, raster rounding, `copy_event` (`copy.deepcopy`). |
| `pulserver.design.params` | Protocol-dict getters/setters, phase-FOV and ACS resolution, readout/phase axis resolution. |
| `pulserver.design.excitation` | Slice-selective sinc, non-selective hard, and adiabatic (hypsec) excitation builders. |
| `pulserver.design.preparations` | Inversion, MT-saturation, T2-prep (with `final_tip="up"/"down"` T1/T2-hybrid switch), and Stejskal–Tanner diffusion modules. |
| `pulserver.design.encoding` | Phase-encode gradients, single-axis crushers, 3-axis spoilers, 3D partition (Z) combination. |
| `pulserver.design.readout` | Cartesian line / echo-train (with optional wave-CAIPI `wave=True` sinusoids), the bridged `unbalanced_line` split-gradient readout, EPI train, FSE/CPMG refocusing train (incl. gradient surgery), non-Cartesian spiral (`forward`/`reversed`/`in_out`/`out_in` variants)/rosette, and ZTE half-echo readouts. |
| `pulserver.design.sampling` | Undersampling masks (uniform, random, Poisson-disc, CAIPIRINHA), golden/uniform angle generators, and echo-train / segment view orderings. |
| `pulserver.design.pulses` | Fat saturation (with `NOPOS`/`NOROT` FOV-transform exemption labels) and the Bloch–Siegert B1-mapping Fermi pulse. |
| `pulserver.design.cli` | `run_cli(plugin, argv, arg_map=...)` — the declarative offline CLI shared by every plugin. |

## View orderings (`pulserver.design.sampling`)

The echo-train / segment view-ordering helpers apply to any acquisition
with an outer loop and an inner echo train or MPRAGE segment — 2D/3D FSE and
segmented GRE alike. They take phase-encode locations in the (ky, kz) plane
and return a list of shots (each an ordered list of view indices; the
echo/segment index is the position within the shot):

- `linear_order`, `outer_inner_order` — the sequential nested-loop orderings
  used by the current FSE/MPRAGE plugins.
- `fse_linear_order` — raster (kz-major) linear reordering.
- `fse_radial_order` — center-out radial (wedge) reordering.
- `fse_radial_adaptive_order` — adaptive radial reordering with per-shot
  parameter support.
- `fse_shuffling_order` — randomly shuffled (T2-Shuffling) reordering with
  spatial clustering to limit gradient switching.

References: Buonincontri et al., *Doubling the repetition time without paying
the price: 3D TSE with individually parameterized echo trains*, ISMRM
566-05-007 ([`refcode/Abstract 566-05-007.pdf`](../../../../refcode/)) for the
linear / radial / adaptive schemes; Tamir et al., *T2 Shuffling*, Magn Reson
Med 2017;77:180–195 ([`refcode/nihms804984.pdf`](../../../../refcode/)) for
random shuffling.

## Advanced excitation (`pulserver.design.excitation`)

Beyond the slice-selective/hard/adiabatic builders, `frequency_selective`
builds spectrally selective (optionally multiband) Gaussian pulses, and
`multiband` applies SMS phase modulation to any slice-selective base pulse
with `quadratic` (Grissom), `wong`, `malik`, or `none` phase schemes (ported
from `refcode/sigpy`, `sigpy.mri.rf.multiband`). Spectral-spatial (SPSP) and
2D/3D volume-selective pulse design are **not** included: the reference
implementations (`refcode/Spectral-Spatial-RF-Pulse-Design`) are LP-based
FIR/SLR design toolboxes whose port requires genuine pulse-design judgment
(plan decision point D1) and is deferred.

## API documentation rendering

Every public function carries a NumPy-style docstring with `Parameters`,
`Returns`, an `Examples` doctest, and a Sphinx `.. plot::` directive
(`:include-source: false`) that renders an illustrative figure. This repo's
`docs/` tree is currently Markdown-only (no Sphinx `conf.py`), so those
`.. plot::` blocks are not rendered by any wired-up build yet. When a Sphinx
site is added, enable `matplotlib.sphinxext.plot_directive` in its
`extensions` and add an `automodule` page per submodule to render them; the
docstrings are already written for that.
