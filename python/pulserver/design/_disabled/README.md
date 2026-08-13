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
| PINS | `_rf/_multiband.py` |
| 3D and plane-selective pulses | `_rf/_spatial.py` |
| Sampling masks, orderings, `ScanLoop` | `_sampling/` |
| The `make_*` factory layer | `_readout/_factories.py`, `_readout/_procedural.py` |
| `pulserver.sequences` REPL factories | `_shim.py` |

`_readout/fse.py` stays only because `_readout/_factories.py` imports from it;
the FSE family itself lives in `design/readout/fse.py`. `_factories.py` also
imports `.bssfp` and `.zte`, which are gone — `design/readout/bssfp.py` and
`design/readout/zte.py` replace them, and
`pulserver.pypulseq.calc_projection_shell` replaces the shell generator ZTE
needed from `_sampling/`.

`_factories.py` also imports `.epi`, which is gone —
`design/readout/epi.py` and `design/readout/propeller.py` replace it, with the
flyback variants folded in as `flyback=True` rather than kept as classes of
their own, and `pulserver.pypulseq.calc_epi_order` replacing its within-shot
patterns. `_sampling/epi.py` stays: `make_skipped_caipi_order` is
superseded by `calc_epi_order`, but `_from_shots`,
`build_from_relative_shifts` and `interleaved` are `ScanLoop` machinery that
`_sampling/patterns.py` and `_sampling/_scanloop.py` still import.

`_rf/_multiband.py` and `_rf/_spatial.py` are **partial** ports.
`design/excitation/multiband.py` and `spectral.py` cover SPSP, SMS and the
ihMT multiband pulse over `pp.make_sms_pulse` / `pp.make_spsp_pulse`, and
`design/excitation/spatial2d.py` covers the 2D-selective one over the new
`pp.make_2d_selective_pulse`. What is left in those two files is PINS, and the
plane- and 3D-selective designs.

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
