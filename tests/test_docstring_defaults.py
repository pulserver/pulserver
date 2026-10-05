"""Every parameter a docstring documents is one the signature has, at its default.

A Parameters entry names a parameter of the callable it documents, and a
default it states as ``default=value`` is the signature's. The sources are
parsed rather than imported, so the check needs no compiled extension.
"""

import ast
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SOURCES = sorted(ROOT.joinpath("src/pulserver").rglob("*.py"))
HEADER = re.compile(
    r"^(\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)(?:\s*:\s*(.*))?$"
)
DEFAULT = re.compile(r"(?:^|,\s*)default\s*=\s*(.+)$")
SECTIONS = ("Parameters", "Other Parameters")


def _callables(tree):
    """``(documented node, function whose signature applies, docstring)``."""
    for node in ast.walk(tree):
        target = node
        if isinstance(node, ast.ClassDef):
            target = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__"
                ),
                None,
            )
        if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=True)
        if doc and any(
            f"{section}\n{'-' * len(section)}" in doc for section in SECTIONS
        ):
            yield node, target, doc


def _documented(doc):
    """Map each documented parameter name to the text after its colon."""
    lines = doc.splitlines()
    out = {}
    for section in SECTIONS:
        if section not in lines:
            continue
        start = lines.index(section) + 2
        for line in lines[start:]:
            if (
                line
                and not line.startswith(" ")
                and line.rstrip()
                in (
                    *SECTIONS,
                    "Returns",
                    "Yields",
                    "Raises",
                    "Warns",
                    "Notes",
                    "Examples",
                    "Attributes",
                    "See Also",
                    "References",
                )
            ):
                break
            if not line or line.startswith(" "):
                continue
            match = HEADER.match(line.strip())
            if match:
                for name in map(str.strip, match.group(1).split(",")):
                    out[name.lstrip("*")] = match.group(2) or ""
    return out


def _value(text):
    """A default as a value, so that ``0.0`` and ``0``, or ``np.inf`` and ``inf``, compare equal."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return eval(text, {"np": np, "math": math, "inf": math.inf})  # noqa: S307
        except Exception:
            return text


def _signature(node):
    positional = node.args.posonlyargs + node.args.args
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    out = {
        arg.arg: None if default is None else ast.unparse(default)
        for arg, default in zip(
            positional + node.args.kwonlyargs,
            defaults + list(node.args.kw_defaults),
            strict=True,
        )
    }
    variadic = {a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None}
    return out, variadic


CASES = [
    pytest.param(
        path, owner, function, doc, id=f"{path.relative_to(ROOT)}:{owner.lineno}"
    )
    for path in SOURCES
    for owner, function, doc in _callables(ast.parse(path.read_text()))
]


@pytest.mark.parametrize("path,owner,function,doc", CASES)
def test_documented_parameters_match_the_signature(path, owner, function, doc):
    defaults, variadic = _signature(function)
    errors = []
    for name, spec in _documented(doc).items():
        if name in variadic:
            continue
        if name not in defaults:
            errors.append(f"{name} is documented but is not a parameter")
            continue
        match = DEFAULT.search(spec)
        if match is None:
            continue
        stated, expected = match.group(1).strip(), defaults[name]
        if expected is None:
            errors.append(f"{name} is required but documents default={stated}")
        elif _value(stated) != _value(expected):
            errors.append(
                f"{name} documents default={stated}; the signature has {expected}"
            )
    assert not errors, "; ".join(errors)


def test_a_documented_parameter_the_signature_lacks_is_found():
    source = '''
def f(x, *, step=0.5):
    """Do something.

    Parameters
    ----------
    x
        The input.
    steps : float, default=0.5
        Stale name.
    """
'''
    ((_, function, doc),) = _callables(ast.parse(source))
    defaults, _ = _signature(function)
    documented = _documented(doc)
    assert set(documented) - set(defaults) == {"steps"}
