"""The vendor file: the grouping and number formats one vendor's sequencer reads a cache in."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from ._convert import Format, Grouping, Quantity, VendorProfile

_FLAGS = {"true": True, "false": False}


def read_vendor(path: Path | str) -> tuple[VendorProfile, Grouping]:
    """Read a vendor file into the profile and grouping a conversion takes.

    The file holds a ``[Grouping]`` and a ``[VendorProfile]`` block, each
    closed by its ``[... End]`` line, with one ``name: value`` line per field
    and every field stated; blank lines and lines starting with ``#`` are
    skipped. A grouping field is a number or ``true``/``false``; a profile
    field is a :class:`Format` name and the SI value of one integer step,
    ``0`` for a float::

        [Grouping]
        boundary_gradient_hz_per_m: 100
        split_by_pulses: true
        split_by_readouts: true
        split_navigators: true
        split_edge_delays: true
        [Grouping End]
        [VendorProfile]
        grad_sample: int16 1.5e-4
        grad_amplitude: float32 0
        rf_sample: int16 3.0e-5
        rf_amplitude: float32 0
        rf_phase: float32 0
        rf_frequency: float32 0
        [VendorProfile End]

    Raises
    ------
    ValueError
        If the file states anything else, leaves a field out, or states one
        twice: nothing falls back to a default.
    """
    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if current is None and line in ("[Grouping]", "[VendorProfile]"):
            name = line[1:-1]
            if name in blocks:
                raise ValueError(f"{path}:{number}: a second {line} block")
            current = blocks[name] = {}
        elif current is not None and line in ("[Grouping End]", "[VendorProfile End]"):
            current = None
        elif current is not None and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            if key in current:
                raise ValueError(f"{path}:{number}: {key} stated twice")
            current[key] = value.strip()
        else:
            raise ValueError(f"{path}:{number}: not part of a vendor file: {line!r}")
    if current is not None or set(blocks) != {"Grouping", "VendorProfile"}:
        raise ValueError(
            f"{path}: a vendor file is one closed [Grouping] and one closed "
            "[VendorProfile] block"
        )
    return (
        VendorProfile(**_fields(VendorProfile, blocks["VendorProfile"], _quantity)),
        Grouping(**_fields(Grouping, blocks["Grouping"], _grouping_value)),
    )


def _fields(kind: type, given: dict[str, str], parse) -> dict:
    names = [f.name for f in fields(kind)]
    if set(given) != set(names):
        missing = sorted(set(names) - set(given))
        unknown = sorted(set(given) - set(names))
        raise ValueError(
            f"[{kind.__name__}] is missing {missing} and states unknown {unknown}"
        )
    values = {}
    for name in names:
        try:
            values[name] = parse(name, given[name])
        except (KeyError, ValueError):
            raise ValueError(
                f"[{kind.__name__}] {name}: cannot read {given[name]!r}"
            ) from None
    return values


def _grouping_value(name: str, text: str) -> float | bool:
    if name == "boundary_gradient_hz_per_m":
        return float(text)
    return _FLAGS[text.lower()]


def _quantity(_: str, text: str) -> Quantity:
    format_name, step = text.split()
    return Quantity(Format[format_name.upper()], float(step))
