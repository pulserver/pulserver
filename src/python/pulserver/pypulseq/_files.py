"""I/O helpers for reading and writing pulserver sequences."""

from __future__ import annotations

__all__ = ["bands_to_hz_per_m", "read", "read_asc_bands", "read_esp_bands", "write"]

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from ._safety import bands_to_hz_per_m, read_asc_bands, read_esp_bands


def write(
    seq,
    output: str | Path | BinaryIO | None = None,
    *,
    create_signature: bool = False,
    remove_duplicates: bool = True,
    check_timing: bool = False,
):
    """Write a Pulseq sequence to path or return binary payload.

    Parameters
    ----------
    seq : pulserver.pypulseq.Sequence
        The sequence to serialise.
    output : str, pathlib.Path or file object, optional
        Where to write. A path without a ``.seq`` suffix gains one. A binary
        stream is written into. By default the bytes are returned instead.
    create_signature : bool, default False
        Append the ``[SIGNATURE]`` section.
    remove_duplicates : bool, default True
        Deduplicate before writing, leaving ``seq`` itself alone.
    check_timing : bool, default False
        Warn if the sequence has timing errors.

    Returns
    -------
    bytes or str or None
        The payload when ``output`` is omitted, otherwise the signature when
        one was created.

    See Also
    --------
    read : the inverse.
    pulserver.pypulseq.Sequence.write : the same file, written to a path.
    """
    from pulserver.pypulseq._sequence import _signature_of

    target = seq._prepared_for_write(
        check_timing=check_timing,
        check_gradients=False,
        remove_duplicates=remove_duplicates,
    )
    payload = target._to_text(create_signature=create_signature)

    if output is None:
        return payload

    writer = getattr(output, "write", None)
    if callable(writer):
        writer(payload)
    else:
        path = Path(output)
        if path.suffix != ".seq":
            path = path.with_suffix(path.suffix + ".seq")
        path.write_bytes(payload)
    return _signature_of(payload) if create_signature else None


#: ``[DEFINITIONS]`` key -> :class:`pypulseq.Opts` attribute for the raster
#: times a Pulseq file records about the system that wrote it.
_RASTER_DEFINITIONS = {
    "GradientRasterTime": "grad_raster_time",
    "RadiofrequencyRasterTime": "rf_raster_time",
    "AdcRasterTime": "adc_raster_time",
    "BlockDurationRaster": "block_duration_raster",
}


def _parse_definitions(payload: bytes) -> dict:
    """Read the ``[DEFINITIONS]`` block, numbers parsed as floats or lists."""
    definitions: dict = {}
    in_section = False
    for raw in payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_section = line == "[DEFINITIONS]"
            continue
        if not in_section:
            continue
        key, _, rest = line.partition(" ")
        fields = rest.split()
        try:
            values = [float(v) for v in fields]
        except ValueError:
            definitions[key] = rest.strip()
            continue
        definitions[key] = values[0] if len(values) == 1 else values
    return definitions


def _system_from_definitions(definitions: dict, system):
    """Build system limits carrying the raster times the file records.

    A ``.seq`` file stores its raster times but not gradient or slew limits,
    so those keep whatever the caller supplied (or Pulserver's vendor-neutral
    defaults). An explicit ``system`` is honoured unchanged.
    """
    if system is not None:
        return system

    from pulserver.pypulseq._opts import Opts

    overrides = {
        attr: float(definitions[key])
        for key, attr in _RASTER_DEFINITIONS.items()
        if key in definitions
    }
    return Opts(**overrides)


def read(source: str | Path | bytes, *, system=None):
    """Read a Pulseq file back into a :class:`pulserver.pypulseq.Sequence`.

    Parameters
    ----------
    source : str, pathlib.Path or bytes
        Path to a ``.seq`` text file or a binary Pulseq file, or the payload
        itself. Which format it is is decided by the leading bytes.
    system : pypulseq.Opts, optional
        System limits to attach. By default the raster times recorded in
        ``[DEFINITIONS]`` are used, on top of Pulserver's vendor-neutral
        defaults.

    Returns
    -------
    pulserver.pypulseq.Sequence
        The sequence, blocks and libraries and all.

    See Also
    --------
    write : the inverse.

    Notes
    -----
    Block rotations, pTx shim vectors and label strings outside Pulseq's
    built-in set are decoded natively, which upstream PyPulseq 1.5.0 cannot
    do. Nothing is replayed through it.
    """
    from pulserver.pypulseq._sequence import Sequence

    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    system = _system_from_definitions(_parse_definitions(payload), system)

    seq = Sequence(system=system)
    if isinstance(source, bytes):
        with NamedTemporaryFile(suffix=".seq") as handle:
            handle.write(payload)
            handle.flush()
            seq.read(handle.name)
    else:
        seq.read(source)
    return seq
