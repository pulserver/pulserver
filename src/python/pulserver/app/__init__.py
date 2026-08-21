"""Pulserver's shipped plugins: one complete, self-contained module each.

A plugin is a worked example of a whole stack, written the way you would write
it yourself. A sequence module composes :mod:`pulserver.design` modules into an
encoding plan and loops over it writing blocks; a reconstruction module takes
what the scanner sends back and turns it into images.

Every module here is callable, and calling it does the module's job: a sequence
module designs a sequence, a reconstruction module reconstructs a file. So the
two families read the same way::

    from pulserver.app import gre3D_sequence, cartesian3D_recon

    seq = gre3D_sequence(n_x=128, n_y=128, n_z=64, slab_thickness=0.128)
    seq.write("gre3D.seq")

    images = cartesian3D_recon("scan.h5")

That call is the module's ``main``, and it is the whole of its public surface:
a sequence module writes one, a reconstruction module has one built from the
settings its plugin takes, so ``help`` and the API reference answer for the call
you are about to make either way. Beside it, ``PLUGIN`` is the same thing behind
the scanner contract -- a :class:`pulserver.design.SequencePlugin`, so the
bridge can offer the sequence in the UI, or a
:class:`pulserver.recon.ReconPlugin`, so the reconstruction can be driven over a
live stream. The reconstruction is not a different code path from the inline
one: the file holds what the scanner sent, and it is streamed to the plugin in
this process, through the same lifecycle hooks.

One flat namespace: every plugin is ``pulserver.app.<name>``, and there is no
subpackage to remember. The families are a way of reading the zoo rather than a
way of importing from it, so the grouping -- gradient echo, spin echo, echo
planar, Cartesian, non-Cartesian -- lives in the API reference.

:mod:`pulserver.recon` is a different thing: the reconstruction toolbox these
plugins are built out of, as :mod:`pulserver.design` is for the sequences.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Any

#: A plugin module's name says which family it belongs to, and the family says
#: which package holds it. This is the only place that mapping is written down.
_FAMILIES = {"_sequence": "sequence", "_recon": "recon"}

#: What :meth:`pulserver.recon.ReconPlugin.run` takes beyond the file itself.
#: A generated ``main`` passes these straight through and keeps everything else
#: for the plugin's constructor.
_STREAM_ARGUMENTS = ("group", "exam_id", "config")


def _plugin_names() -> list[str]:
    """Every plugin this package ships, both families in one list."""
    names: list[str] = []
    for family in _FAMILIES.values():
        names += importlib.import_module(f"{__name__}.{family}").__all__
    return sorted(names)


__all__ = _plugin_names()


class PluginModule(ModuleType):
    """A plugin module, callable as the job it does.

    A plugin *is* its ``main``, so the module carries ``main``'s docstring and
    signature rather than the file's: :func:`help` and
    :func:`inspect.signature` on the module answer for the call you are about
    to make, and so does the API reference.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.main(*args, **kwargs)


def _as_plugin(module: ModuleType) -> ModuleType:
    """Present ``module`` as the callable it wraps.

    A sequence module's ``main`` is the whole of it, docstring included. A
    reconstruction module carries a configured ``PLUGIN`` instead, so its
    ``main`` is built here from the settings that plugin takes and the file its
    reconstruction reads.
    """
    module.__class__ = PluginModule
    main = getattr(module, "main", None)
    if main is None:
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            return module
        main = _recon_entry_point(plugin)
        module.main = main
    module.__doc__ = main.__doc__
    module.__signature__ = inspect.signature(main)
    return module


def _recon_entry_point(plugin: Any) -> Any:
    """Build the ``main`` one reconstruction module is called through.

    A reconstruction is a configured object rather than a function, so the
    entry point is derived from it: the settings its constructor takes become
    keyword arguments, in front of the file to reconstruct and the stream
    arguments every plugin's :meth:`~pulserver.recon.ReconPlugin.run` accepts.
    Calling it configures one plugin and streams the file to it.
    """
    from pulserver.recon import ReconPlugin

    plugin_class = type(plugin)
    settings = [
        parameter
        for parameter in inspect.signature(plugin_class.__init__).parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    stream = inspect.signature(ReconPlugin.run).parameters
    signature = inspect.Signature(
        [
            inspect.Parameter("path", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *settings,
            *(stream[name] for name in _STREAM_ARGUMENTS),
        ]
    )

    def main(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        path = values.pop("path")
        arguments = {name: values.pop(name) for name in _STREAM_ARGUMENTS}
        return plugin_class(**values).run(path, **arguments)

    main.__name__ = "main"
    main.__qualname__ = "main"
    main.__signature__ = signature
    main.__doc__ = _recon_docstring(plugin_class, ReconPlugin)
    return main


def _recon_docstring(plugin_class: type, base: type) -> str:
    """The generated ``main``'s documentation, from the plugin's own.

    What a plugin's class documents is what its entry point does, so the prose
    and the parameters are the class's, and what the call adds -- the file, the
    stream arguments, the images that come back -- is taken from
    :meth:`~pulserver.recon.ReconPlugin.run`, which every plugin inherits
    unchanged.
    """
    lead, sections = _numpydoc_sections(plugin_class.__doc__)
    _, run_sections = _numpydoc_sections(base.run.__doc__)
    run_parameters = _parameter_entries(run_sections.get("Parameters", ""))

    parameters = [run_parameters["path"]]
    if sections.get("Parameters"):
        parameters.append(sections["Parameters"])
    parameters += [
        run_parameters[name] for name in _STREAM_ARGUMENTS if name in run_parameters
    ]

    parts = [lead] if lead else []
    parts.append("Parameters\n----------\n" + "\n".join(parameters))
    parts.append(
        "Returns\n-------\nlist\n    Every image the reconstruction emitted, in order."
    )
    parts += [
        f"{name}\n{'-' * len(name)}\n{body}"
        for name, body in sections.items()
        if name not in {"Parameters", "Returns", "Examples"}
    ]
    if sections.get("Examples"):
        parts.append("Examples\n--------\n" + sections["Examples"])
    return "\n\n".join(parts)


def _numpydoc_sections(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a NumPy-style docstring into its lead text and named sections."""
    lines = inspect.cleandoc(doc or "").splitlines()
    lead: list[str] = []
    sections: dict[str, list[str]] = {}
    body = lead
    index = 0
    while index < len(lines):
        title = lines[index].strip()
        underline = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if title and set(underline) == {"-"} and len(underline) >= len(title):
            body = sections.setdefault(title, [])
            index += 2
            continue
        body.append(lines[index])
        index += 1
    return (
        "\n".join(lead).strip(),
        {title: "\n".join(text).strip("\n") for title, text in sections.items()},
    )


def _parameter_entries(block: str) -> dict[str, str]:
    """One NumPy ``Parameters`` block as its entries, by parameter name."""
    entries: dict[str, str] = {}
    name: str | None = None
    body: list[str] = []
    for line in block.splitlines():
        if line and not line[0].isspace():
            if name is not None:
                entries[name] = "\n".join(body).rstrip()
            name = line.split(":")[0].strip()
            body = [line]
        elif name is not None:
            body.append(line)
    if name is not None:
        entries[name] = "\n".join(body).rstrip()
    return entries


def _family_of(name: str) -> str | None:
    """The subpackage a plugin name belongs to, by its suffix."""
    for suffix, family in _FAMILIES.items():
        if name.endswith(suffix):
            return family
    return None


def __getattr__(name: str):
    """Import one plugin on first use."""
    family = _family_of(name)
    if family is not None and name in __all__:
        module = importlib.import_module(f"{__name__}.{family}.{name}")
        if type(module) is not PluginModule:
            _as_plugin(module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return every plugin, both families in one namespace."""
    return list(__all__)
