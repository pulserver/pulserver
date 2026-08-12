# Staged for reinstatement

Nothing in this directory is imported. There are no `__init__.py` files, so it
is not a package and `pulserver.design` cannot reach it.

These files are written against a module protocol that no longer exists —
`set_state`, `payloads`, `ScanLoop`, and per-shot `copy_event` of Pulseq
events. They are kept because their **physics** is the source to port from as
each family is reinstated on the current `SequenceModule`. Port a file, then
delete it from here.

| Family | Source |
|---|---|
| EPI 2D/3D, blipped and flyback | `_readout/epi.py`, `_sampling/epi.py` |
| FSE / CPMG trains | `_readout/fse.py` |
| ZTE | `_readout/zte.py` |
| bSSFP | `_readout/bssfp.py` |
| Fat saturation, MT, ihMT | `_rf/_preparation.py` |
| T2 prep, hybrid T1/T2 prep | `_rf/_preparation.py`, `_rf/_preparation_helpers.py` |
| Diffusion preparation | `_rf/_preparation.py` |
| Bloch–Siegert | `_rf/_preparation.py` |
| Refocusing, hard, adiabatic, frequency-selective pulses | `_rf/_excitation.py`, `_rf/_excitation_helpers.py` |
| SPSP, SMS, PINS, multiband | `_rf/_multiband.py` |
| 2D/3D spatially selective pulses | `_rf/_spatial.py` |
| Sampling masks, orderings, `ScanLoop` | `_sampling/` |
| The `make_*` factory layer | `_readout/_factories.py`, `_readout/_procedural.py` |
| Z-channel gradient combination, crushers | `_gradients.py` |
| `pulserver.sequences` REPL factories | `_shim.py` |

Three things every port has to change, because they are why this code cannot
run as it stands:

- `import pypulseq as pp` → `from ... import pypulseq as pp`. Upstream's
  factories build `SimpleNamespace` events; Pulserver's build `pulseqpp`
  events, and only the latter reach the C++ `Sequence`.
- `_system.copy_event` / `_system.scale_grad` → `pp.scale_grad`. Pulseq events
  are C extension types and cannot be copied at all.
- `_traj2grad.traj2grad` → `pp.traj_to_grad`; `_system.round_to_raster`,
  `ceil_to_raster`, `apply_system_derates`, `quantize_readout_timing` and the
  `_schedules` functions are all exported from `pulserver.pypulseq` now.
