# ruff: noqa: F822
"""Ready-to-run sequence factories for interactive Pulserver use.

The bridge plugins in :mod:`examples.sequences` are the source of truth for
these factories.  The public functions below expose the same controls as
explicit keyword arguments and return the in-memory
:class:`pulserver.pypulseq.Sequence` instead of receiving a serialized
interpreter protocol and an output path.  Consequently, a sequence designed
at a REPL and one designed through the bridge have exactly one implementation.

Examples
--------
Build and inspect a Cartesian GRE sequence directly::

    >>> from pulserver.sequences import design_gre_2d
    >>> seq = design_gre_2d(nx=64, ny=64, nslices=1)
    >>> seq.get_definition("Name")
    'gre_2d'

``pulserver.sequence`` is an alias for this namespace for callers that prefer
the singular spelling.  New code should use :mod:`pulserver.sequences`.
"""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pulserver.pypulseq as pp
from pulserver._core._params import set_protocol_value

_PACKAGE = __name__.rpartition(".")[0]

__all__ = [
    "design_bssfp_2d",
    "design_bssfp_3d",
    "design_epi_2d",
    "design_epi_3d",
    "design_fse_2d",
    "design_fse_3d",
    "design_gre_2d",
    "design_gre_3d",
    "design_gre_mprage_radial_2d",
    "design_gre_mprage_radial_3d",
    "design_gre_multiecho_2d",
    "design_gre_multiecho_3d",
    "design_gre_noncart_2d",
    "design_gre_noncart_3d",
    "design_gre_radial_2d",
    "design_gre_radial_3d",
    "design_mprage_2d",
    "design_mprage_3d",
    "design_zte_2d",
    "design_zte_3d",
    "design_gre",
]


_PLUGIN_NAMES = tuple(name.removeprefix("design_") for name in __all__ if name != "design_gre")
_PLUGIN_LOCK = threading.RLock()
_WRITE_LOCK = threading.RLock()
_LOADED: dict[str, ModuleType] = {}


def _load_plugin(name: str) -> ModuleType:
    """Import one self-contained bridge plugin as a package submodule."""
    if name not in _PLUGIN_NAMES:
        raise ValueError(f"Unknown Pulserver sequence plugin: {name}")

    with _PLUGIN_LOCK:
        if name in _LOADED:
            return _LOADED[name]

        # ``tool.scikit-build.wheel.packages`` maps examples/sequences directly
        # to this package path in wheels and editable installs.
        try:
            module = importlib.import_module(f"{_PACKAGE}.{name}")
        except ModuleNotFoundError as error:
            expected = f"{_PACKAGE}.{name}"
            if error.name != expected:
                raise
            # A source checkout has the plugins in examples/ until it is
            # installed.  The fallback uses normal package importing too.
            source_root = Path(__file__).resolve().parents[3]
            if not (source_root / "examples" / "sequences" / f"{name}.py").is_file():
                raise ImportError(
                    f"Pulserver sequence plugin {name!r} is unavailable. "
                    "Install pulserver with its sequence examples or use an editable checkout."
                ) from error
            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
            module = importlib.import_module(f"examples.sequences.{name}")

        _LOADED[name] = module
        return module


def _default_value(protocol: dict[str, dict], key: object, kind: object) -> object:
    """Translate one bridge argument's stored default into its public value."""
    stored = protocol[str(key)]["value"]
    if isinstance(kind, tuple) and kind[0] == "const":
        return stored == kind[1]
    if isinstance(kind, dict):
        for public, internal in kind.items():
            if stored == internal:
                return public
        raise RuntimeError(f"No public choice maps to default {stored!r} for {key!s}")
    return stored


def _parameter_description(name: str, help_text: str) -> str:
    """Give formerly CLI-only controls meaningful REPL documentation."""
    if help_text:
        return help_text.rstrip(".") + "."
    labels = {
        "te_ms": "Echo time in ms",
        "tr_ms": "Repetition time in ms",
        "ti_ms": "Inversion time in ms",
        "te_prep_ms": "T2-preparation echo time in ms",
        "trecovery_ms": "Recovery delay in ms",
        "flip_deg": "RF flip angle in degrees",
        "fov_mm": "Readout field of view in mm",
        "phase_fov_mm": "Phase-encode field of view in mm",
        "slice_thickness_mm": "Slice thickness in mm",
        "slab_thickness_mm": "Slab thickness in mm",
        "slice_spacing_mm": "Center-to-center slice spacing in mm",
        "partition_spacing_mm": "Center-to-center partition spacing in mm",
        "nx": "Readout matrix size",
        "ny": "Phase-encode matrix size",
        "nslices": "Number of slices",
        "npartitions": "Number of partitions",
        "num_shots": "Number of interleaves or projections",
        "num_echoes": "Number of readout echoes",
        "echo_spacing_ms": "Spacing between successive readout echoes in ms",
        "etl": "Echo-train length",
        "ry": "Phase-encode acceleration factor",
        "rz": "Partition acceleration factor",
        "bandwidth_hz_px": "Readout bandwidth in Hz/pixel",
        "acs_lines": "Number of fully sampled ACS lines",
        "ramp_sample": "Enable EPI ramp sampling",
        "flyback": "Use flyback rather than bipolar multi-echo readout",
        "bipolar": "Use bipolar rather than flyback multi-echo readout",
        "swap_phase_freq": "Swap physical readout and phase-encode axes",
        "bvalue": "Diffusion weighting in s/mm²",
        "directions": "Number of diffusion directions",
        "trajectory": "Non-Cartesian trajectory family",
        "order_mode": "View-angle ordering mode",
        "segments": "Number of 3D bSSFP segments",
        "slab_selective": "Use a slab-selective instead of nonselective excitation",
        "spsp": "Use water-selective spectral-spatial slab excitation",
        "spsp_bandwidth_hz": "SPSP water passband bandwidth in Hz",
        "fat_sat": "Play a spectrally selective fat-saturation module before each shot",
        "b0_t": "Main field strength used to place the fat-saturation band in tesla",
        "multiband": "Simultaneous-multislice acceleration factor",
        "num_frames": "Number of dynamic volumes or phases",
        "ttl_output": "Emit an external TTL pulse at the start of every volume",
        "trigger": "Optional physiological trigger source at the start of every phase",
        "prep_type": "Magnetization preparation family",
        "inversion_mode": "Inversion RF pulse family",
        "refocus_mode": "T2-preparation refocusing RF pulse family",
        "refocus_variable": "Use a variable refocusing-flip schedule",
        "mt": "Enable magnetization-transfer preparation",
    }
    return labels.get(name, name.replace("_", " ").capitalize()) + "."


def _factory_doc(name: str, fields: list[tuple[str, object, object, str]], plugin: ModuleType) -> str:
    """Build a complete NumPy-style docstring from the bridge's declared UI."""
    summary = (plugin.__doc__ or "").strip().splitlines()[0].rstrip(".")
    params_doc = "\n".join(
        f"{field} : {'bool' if isinstance(kind, tuple) else 'str' if isinstance(kind, dict) else kind.__name__}\n"
        f"    {_parameter_description(field, help_text)}"
        for field, _key, kind, help_text in fields
    )
    examples = [
        f">>> from pulserver.sequences import design_{name}",
        f">>> seq = design_{name}()",
        '>>> seq.get_definition("Name")',
        f"'{name}'",
    ]
    if name.startswith("epi_"):
        examples.extend(
            [
                "",
                "A structural volume uses the default high-resolution geometry; an fMRI",
                "volume uses a shorter TR and is designed once for each requested volume::",
                "",
                f"    structural = design_{name}(nx=128, ny=128)",
                f"    fmri_run = design_{name}(tr_ms=2000.0, nx=64, ny=64, num_frames=2, ttl_output=True)",
            ]
        )
        rendered_examples = f""".. plot::
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   structural = design_{name}(nx=128, ny=128, tr_ms=3000.0)
   sequence_figure(structural, time_range=(0.0, min(0.08, structural.duration()[0])), title="{name} — structural volume")

.. plot::
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   fmri_run = design_{name}(nx=64, ny=64, tr_ms=2000.0, num_frames=2, ttl_output=True)
   sequence_figure(fmri_run, time_range=(0.0, min(0.08, fmri_run.duration()[0])), title="{name} — fMRI run with volume TTL")"""
    elif name.startswith("bssfp_"):
        examples.extend(
            [
                "",
                "A dynamic cardiac acquisition uses the multiphase count and optional",
                "cardiac trigger directly::",
                "",
                f"    cine = design_{name}(num_frames=20, trigger=\"cardiac\")",
            ]
        )
        rendered_examples = f""".. plot::
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   structural = design_{name}()
   sequence_figure(structural, time_range=(0.0, min(0.05, structural.duration()[0])), title="{name} — structural acquisition")

.. plot::
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   cine = design_{name}(num_frames=2, trigger="cardiac")
   sequence_figure(cine, time_range=(0.0, min(0.05, cine.duration()[0])), title="{name} — cardiac-triggered multiphase acquisition")"""
    elif name.startswith(("gre_noncart_", "gre_radial_")):
        examples.extend(
            [
                "",
                "Dynamic view ordering continues across frames and can optionally wait",
                "for a physiological trigger at each frame boundary::",
                "",
                f"    dynamic = design_{name}(num_frames=20, trigger=\"cardiac\")",
            ]
        )
        rendered_examples = f""".. plot::
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   structural = design_{name}(nx=32, num_shots=8)
   sequence_figure(structural, time_range=(0.0, min(0.05, structural.duration()[0])), title="{name} — structural acquisition")

.. plot::
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   dynamic = design_{name}(nx=32, num_shots=8, num_frames=2, trigger="cardiac")
   sequence_figure(dynamic, time_range=(0.0, min(0.05, dynamic.duration()[0])), title="{name} — triggered dynamic acquisition")"""
    elif name.startswith("mprage_"):
        rendered_examples = f""".. plot::
   :context:
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   seq = design_{name}()
   sequence_figure(seq, time_range=(0.0, min(0.03, seq.duration()[0])), title="{name} — inversion preparation")

.. plot::
   :context: close-figs
   :include-source: false

   from _figures import sequence_figure

   ti = seq.get_definition("TI")
   sequence_figure(seq, time_range=(max(0.0, ti - 0.025), min(ti + 0.08, seq.duration()[0])), title="{name} — gradient-echo train, same acquisition")"""
    else:
        rendered_examples = f""".. plot::
   :include-source: false

   from _figures import sequence_figure
   from pulserver.sequences import design_{name}

   seq = design_{name}()
   sequence_figure(seq, time_range=(0.0, min(0.05, seq.duration()[0])), title="{name} — default acquisition")"""
    return f"""{summary}.

This is the REPL callback for ``examples/sequences/{name}.py``.  It accepts
only the plugin's sequence controls as keyword arguments; it does not accept
a bridge protocol dictionary.  The returned sequence has not been written.
Use :func:`pulserver.io.write` when a ``.seq`` file is wanted.

The representative acquisition timeline below uses this callback directly,
not a hand-built surrogate.  EPI shows both structural and fMRI volumes.

{rendered_examples}

Parameters
----------
{params_doc}
system : pulserver.pypulseq.Opts, optional
    Hardware limits and rasters.  ``None`` uses :class:`pulserver.pypulseq.Opts`.

Returns
-------
pulserver.pypulseq.Sequence
    Fully designed sequence, ready to inspect, append to a larger workflow, or write.

Examples
--------
{chr(10).join(examples)}
"""


def _invoke(name: str, values: dict[str, Any]) -> pp.Sequence:
    """Apply explicit arguments to the bridge defaults and capture its sequence."""
    module = _load_plugin(name)
    system = values.pop("system")
    opts = pp.Opts() if system is None else system
    if not isinstance(opts, pp.Opts):
        raise TypeError("system must be a pulserver.pypulseq.Opts instance or None")

    protocol = module.get_default_protocol(opts)
    for flag, key, kind, _help in module._ARG_MAP:
        field = flag.lstrip("-").replace("-", "_")
        value = values[field]
        if isinstance(kind, tuple) and kind[0] == "const":
            if value:
                set_protocol_value(protocol, key, kind[1])
        elif isinstance(kind, dict):
            try:
                set_protocol_value(protocol, key, kind[value])
            except KeyError as error:
                choices = ", ".join(repr(choice) for choice in kind)
                raise ValueError(f"{field} must be one of {choices}; got {value!r}") from error
        else:
            set_protocol_value(protocol, key, value)

    validation = module.validate_protocol(opts, protocol)
    if not validation.get("valid", False):
        raise ValueError(validation.get("info", f"Invalid {name} protocol"))

    captured: list[pp.Sequence] = []

    def _capture(sequence: pp.Sequence, **_kwargs: Any) -> None:
        captured.append(sequence)

    # Bridge plugins own the waveform construction and normally finish by
    # calling pio.write.  Substitute that final side effect only; no plugin
    # implementation is copied into this namespace.
    with _WRITE_LOCK:
        original_write = module.pio.write
        module.pio.write = _capture
        try:
            module.make_sequence(opts, protocol, "unused-by-repl.seq")
        finally:
            module.pio.write = original_write

    if len(captured) != 1:
        raise RuntimeError(f"{name} plugin did not produce exactly one sequence")
    return captured[0]


def _make_factory(name: str):
    """Create a real explicit-keyword function from one plugin's ``_ARG_MAP``."""
    plugin = _load_plugin(name)
    protocol = plugin.get_default_protocol(pp.Opts())
    fields = [
        (flag.lstrip("-").replace("-", "_"), key, kind, help_text)
        for flag, key, kind, help_text in plugin._ARG_MAP
    ]
    seen: set[str] = set()
    parameters: list[str] = []
    for field, key, kind, _help in fields:
        if field in seen:
            continue
        seen.add(field)
        parameters.append(f"{field}={_default_value(protocol, key, kind)!r}")
    parameters.append("system=None")
    function_name = f"design_{name}"
    namespace = {"_invoke": _invoke}
    exec(
        f"def {function_name}(*, {', '.join(parameters)}):\n"
        f"    return _invoke({name!r}, locals())\n",
        namespace,
    )
    factory = namespace[function_name]
    factory.__module__ = _PACKAGE
    factory.__doc__ = _factory_doc(name, fields, plugin)
    return factory


for _plugin_name in _PLUGIN_NAMES:
    globals()[f"design_{_plugin_name}"] = _make_factory(_plugin_name)

# The unqualified GRE factory is intentionally the canonical 2D Cartesian one.
# Build a separate callable so ``design_gre_2d`` retains its own identity in
# tracebacks, autodoc output, and interactive help.
design_gre = _make_factory("gre_2d")
design_gre.__name__ = "design_gre"
design_gre.__qualname__ = "design_gre"
design_gre.__doc__ = design_gre.__doc__.replace("design_gre_2d", "design_gre")
