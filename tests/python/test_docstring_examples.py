"""Every ``Examples`` block in the package runs, and says what it claims.

A docstring example is the one piece of documentation a reader will paste
straight into a script, so it is worth exactly as much as it is true. These
run them as tests: one case per module, so a failure names the module whose
example went stale.

The ``.. plot::`` directives embedded alongside them are executed by the
documentation build instead -- they draw rather than assert, and they need
Sphinx's figure machinery -- so ``bash scripts/build_docs.sh`` is what holds
those.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

#: Where to look. Every module under these, minus the tree kept only to port
#: physics from.
ROOTS = (
    "pulserver",
    "pulserver.design",
    "pulserver.pypulseq",
    "pulserver.recon",
    "pulserver.mrd",
    "pulserver.app",
)


def _raise(name: str) -> None:
    """Walking a package swallows import errors by default, which would drop
    a module out of the parametrisation rather than fail it."""
    raise


def _modules() -> list[str]:
    found: list[str] = []
    for root in ROOTS:
        package = importlib.import_module(root)
        found.append(root)
        found += [
            module.name
            for module in pkgutil.walk_packages(
                getattr(package, "__path__", []), root + ".", onerror=_raise
            )
            if "_disabled" not in module.name
        ]
    return sorted(set(found))


def _has_examples(module) -> bool:
    finder = doctest.DocTestFinder()
    return any(test.examples for test in finder.find(module))


@pytest.mark.parametrize("name", _modules())
def test_the_examples_in_this_module_are_true(name):
    try:
        module = importlib.import_module(name)
    except ImportError as unavailable:  # pragma: no cover - optional backends
        pytest.skip(f"{name} is not importable here: {unavailable}")
    if not _has_examples(module):
        pytest.skip(f"{name} has no examples")

    result = doctest.testmod(
        module, optionflags=doctest.NORMALIZE_WHITESPACE, verbose=False
    )
    assert result.failed == 0, f"{result.failed} of {result.attempted} failed"
