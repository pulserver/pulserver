"""I/O helpers for reading and writing pulserver sequences."""

from __future__ import annotations

__all__ = ["read", "read_asc_bands", "read_esp_bands", "write"]

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from warnings import warn

import numpy as np
from pypulseq.supported_labels_rf_use import get_supported_rf_uses

from pulserver.pypulseq._safety import read_asc_bands, read_esp_bands

#: Pulseq's numeric RF ``use`` codes and the initial each one is written as:
#: undefined, excitation, refocusing, inversion, saturation, preparation,
#: other.
_RF_USE_CODE_TO_CHAR = {0: "u", 1: "e", 2: "r", 3: "i", 4: "s", 5: "p", 6: "o"}


def _render_int_rows(matrix: np.ndarray, widths: tuple[int, ...]) -> str | None:
    """Render a non-negative integer matrix as space-separated right-aligned text.

    ``[BLOCKS]`` is the one section with a row per block rather than per
    distinct event, so on a large 3D protocol it is most of the file. Formatting
    it row by row means turning some five million array elements into Python
    ints first, purely to hand them to ``%``; here the digits are computed with
    arithmetic on the whole column and the section comes out as one string.

    Returns ``None`` when a value will not fit its field -- ``%3d`` widens
    rather than truncating, so those rows are not fixed-width and the caller
    has to fall back to formatting them individually.
    """
    rows, columns = matrix.shape
    if columns != len(widths) or rows == 0 or matrix.min() < 0:
        return None

    # A field is as wide as its format asks unless the value needs more --
    # ``%3d`` widens rather than truncating -- so rows are not all the same
    # length. Each field is laid out at the widest it could be here, and the
    # padding no row actually printed is dropped in one masked compaction at
    # the end.
    maxima = matrix.max(axis=0)
    spans = [max(width, len(str(int(value)))) for width, value in zip(widths, maxima, strict=True)]

    line_width = sum(spans) + len(spans)  # one separator per field, newline last
    text = np.full((rows, line_width), 32, dtype=np.uint8)  # spaces
    keep = np.ones((rows, line_width), dtype=bool)

    cursor = 0
    for column, (span, width) in enumerate(zip(spans, widths, strict=True)):
        values = matrix[:, column]
        field = text[:, cursor : cursor + span]
        remainder = values
        for digit in range(span - 1, -1, -1):
            field[:, digit] = 48 + (remainder % 10)
            remainder = remainder // 10

        # How many characters %{width}d would have emitted for each value.
        digits = np.ones(rows, dtype=np.int64)
        for power in range(1, span):
            digits += values >= 10**power
        printed = np.maximum(digits, width)
        positions = np.arange(span)
        # Everything left of the first significant digit is padding, not a
        # zero; of that padding, only what %{width}d would have emitted is kept.
        field[positions < (span - digits)[:, None]] = 32
        keep[:, cursor : cursor + span] = positions >= (span - printed)[:, None]
        cursor += span + 1

    text[:, -1] = 10  # newline
    return text[keep].tobytes().decode("ascii")


def write(
    seq,
    output: str | Path | BinaryIO | None = None,
    *,
    create_signature: bool = False,
    remove_duplicates: bool = True,
    check_timing: bool = False,
):
    """Write a Pulseq sequence to path or return binary payload.

    When ``output`` is ``None``, bytes are returned instead of writing to disk.
    When ``output`` is a binary file-like object, bytes are written to it.
    """
    target = seq.remove_duplicates() if remove_duplicates else seq

    if check_timing:
        is_ok, error_report = target.check_timing()
        if not is_ok:
            warn(f"write(): {len(error_report)} timing errors found in the sequence", stacklevel=2)

    target.set_definition("TotalDuration", sum(target.block_durations.values()))

    # Deduplication renumbers the block table in one integer matrix rather than
    # row by row, so that matrix -- not the row dicts, which still hold the
    # pre-deduplication IDs -- is what gets written. Reading ``block_events``
    # here would rebuild those dicts for nothing, so nothing below does.
    block_rows = target._block_row_matrix()

    if len(block_rows):
        last_row = block_rows[-1]
        for channel, slot in (("x", 2), ("y", 3), ("z", 4)):
            grad_id = int(last_row[slot])
            if grad_id == 0:
                continue
            grad_type = target.grad_library.type.get(grad_id, "")
            if grad_type != "g":
                continue
            arb_id = int(target.grad_library.data[grad_id][0])
            arb_data = target.arb_library.data[arb_id]
            last_amp = float(arb_data[2])
            if abs(last_amp) > target.system.max_slew * target.system.grad_raster_time:
                warn(
                    f"write(): Gradient on channel {channel} in last sequence block does not ramp down to 0",
                    stacklevel=2,
                )

    has_custom_labels = len(target.custom_labels) != 0
    has_custom_rotation = len(target.rotation_library.data) != 0
    has_custom_rf_shim = len(target.rf_shim_library.data) != 0

    version_major = int(target.version_major)
    version_minor = int(target.version_minor)
    version_revision = int(target.version_revision)
    if has_custom_labels:
        version_revision = max(version_revision, 2)
    elif has_custom_rotation or has_custom_rf_shim:
        version_revision = max(version_revision, 1)

    chunks: list[str] = []

    def w(text: str) -> None:
        chunks.append(text)

    w("# Pulseq sequence file\n")
    w("# Created by PyPulseq\n\n")

    w("[VERSION]\n")
    w(f"major {version_major}\n")
    w(f"minor {version_minor}\n")
    w(f"revision {version_revision}\n")
    w("\n")

    if len(target.definitions) != 0:
        w("[DEFINITIONS]\n")
        keys = sorted(target.definitions.keys())
        values = [target.definitions[k] for k in keys]
        for block_counter in range(len(keys)):
            w(f"{keys[block_counter]} ")
            if isinstance(values[block_counter], str):
                w(values[block_counter] + " ")
            elif isinstance(values[block_counter], int | float):
                w(f"{values[block_counter]:0.9g} ")
            elif isinstance(values[block_counter], list | tuple | np.ndarray):
                for i in range(len(values[block_counter])):
                    if isinstance(values[block_counter][i], int | float):
                        w(f"{values[block_counter][i]:0.9g} ")
                    else:
                        w(f"{values[block_counter][i]} ")
            else:
                raise RuntimeError("Unsupported definition")
            w("\n")
        w("\n")

    w("# Format of blocks:\n")
    w("# NUM DUR RF  GX  GY  GZ  ADC  EXT\n")
    w("[BLOCKS]\n")
    # This is the one section with a row per block, so it is the only one whose
    # cost scales with the scan rather than with the number of distinct events:
    # the raster division and its round-trip check are done for the whole
    # column at once, and each row is emitted with a single %-format.
    n_blocks = len(block_rows)
    if n_blocks:
        block_row_fmt = f"%{len(str(n_blocks))}d %3d %3d %3d %3d %3d %2d %2d\n"
        ticks = np.fromiter(target.block_durations.values(), dtype=float, count=n_blocks)
        ticks /= target.block_duration_raster
        rounded = np.rint(ticks)
        if np.abs(rounded - ticks).max() >= 1e-6:
            worst = int(np.argmax(np.abs(rounded - ticks)))
            raise AssertionError(
                f"block {list(target.block_durations)[worst]} duration is not a multiple of the block "
                f"duration raster ({ticks[worst] * target.block_duration_raster} s)"
            )
        printed = np.empty((n_blocks, 8), dtype=np.int64)
        printed[:, 0] = np.fromiter(target.block_durations, dtype=np.int64, count=n_blocks)
        printed[:, 1] = rounded
        printed[:, 2:] = block_rows[:, 1:]

        widths = (len(str(n_blocks)), 3, 3, 3, 3, 3, 2, 2)
        rendered = _render_int_rows(printed, widths)
        if rendered is not None:
            w(rendered)
        else:
            append = chunks.append
            for row in printed.tolist():
                append(block_row_fmt % tuple(row))
    w("\n")

    if len(target.rf_library.data) != 0:
        w("# Format of RF events:\n")
        w("# id ampl. mag_id phase_id time_shape_id center delay freqPPm phasePPM freq phase use\n")
        w("# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   Hz   rad  ..\n")
        w(f"# Field \"use\" is the initial of: {' '.join(get_supported_rf_uses()).strip()}\n")
        w("[RF]\n")
        rf_fmt = "%.0f %12g %.0f %.0f %.0f %g %g %g %g %g %g %s\n"
        rf_types = target.rf_library.type
        rf_raster = target.rf_raster_time
        append = chunks.append
        for k, data in target.rf_library.data.items():
            center = data[4] * 1e6
            delay = round(data[5] / rf_raster) * rf_raster * 1e6
            use_type = rf_types.get(k, "u")
            use_char = (
                _RF_USE_CODE_TO_CHAR.get(int(use_type), "u") if isinstance(use_type, int | np.integer) else use_type
            )
            append(rf_fmt % (k, *data[0:4], center, delay, *data[6:10], use_char))
        w("\n")

    grad_keys = np.array(list(target.grad_library.data.keys()), dtype=int) if target.grad_library.data else np.array([])
    grad_types = (
        np.array([target.grad_library.type[k] for k in grad_keys], dtype=object) if len(grad_keys) else np.array([])
    )
    arb_grad_mask = grad_types == "g" if len(grad_types) else np.array([], dtype=bool)
    trap_grad_mask = grad_types == "t" if len(grad_types) else np.array([], dtype=bool)

    if np.any(arb_grad_mask):
        w("# Format of arbitrary gradients:\n")
        w("#   time_shape_id of 0 means default timing (stepping with grad_raster starting at 1/2 of grad_raster)\n")
        w("# id amplitude first last amp_shape_id time_shape_id delay\n")
        w("# ..      Hz/m  Hz/m Hz/m        ..         ..          us\n")
        w("[GRADIENTS]\n")
        arb_fmt = "%.0f %12g %12g %12g %.0f %.0f %.0f\n"
        grad_data = target.grad_library.data
        arb_data = target.arb_library.data
        append = chunks.append
        for k in grad_keys[arb_grad_mask].tolist():
            data = arb_data[int(grad_data[k][0])]
            append(arb_fmt % (k, *data[:5], round(data[5] * 1e6)))
        w("\n")

    if np.any(trap_grad_mask):
        w("# Format of trapezoid gradients:\n")
        w("# id amplitude rise flat fall delay\n")
        w("# ..      Hz/m   us   us   us    us\n")
        w("[TRAP]\n")
        # Trapezoids are the bulk of a Cartesian scan's event libraries, so the
        # seconds -> microseconds conversion is done on the four scalars rather
        # than by building a numpy array per entry. %.Nf rounds the same way
        # np.round did, so the text is unchanged.
        trap_fmt = "%2.0f %12g %3.0f %4.0f %3.0f %3.0f\n"
        grad_data = target.grad_library.data
        trap_data = target.trap_library.data
        append = chunks.append
        for k in grad_keys[trap_grad_mask].tolist():
            amplitude, rise, flat, fall, delay = trap_data[int(grad_data[k][0])]
            append(trap_fmt % (k, amplitude, 1e6 * rise, 1e6 * flat, 1e6 * fall, 1e6 * delay))
        w("\n")

    if len(target.adc_library.data) != 0:
        w("# Format of ADC events:\n")
        w("# id num dwell delay freqPPM phasePPM freq phase phase_id\n")
        w("# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n")
        w("[ADC]\n")
        adc_fmt = "%.0f %.0f %.0f %.0f %g %g %g %g %.0f\n"
        append = chunks.append
        for k, data in target.adc_library.data.items():
            num, dwell, delay, freq_ppm, phase_ppm, freq, phase, phase_id = data[0:8]
            append(adc_fmt % (k, num, 1e9 * dwell, 1e6 * delay, freq_ppm, phase_ppm, freq, phase, phase_id))
        w("\n")

    if len(target.extensions_library.data) != 0:
        w("# Format of extension lists:\n")
        w("# id type ref next_id\n")
        w("# next_id of 0 terminates the list\n")
        w("# Extension list is followed by extension specifications\n")
        w("[EXTENSIONS]\n")
        ext_fmt = "%.0f %.0f %.0f %.0f\n"
        append = chunks.append
        for k, data in target.extensions_library.data.items():
            append(ext_fmt % (k, *data))
        w("\n")

    if len(target.trigger_library.data) != 0:
        w("# Extension specification for digital output and input triggers:\n")
        w("# id type channel delay (us) duration (us)\n")
        w(f"extension TRIGGERS {target.get_extension_type_ID('TRIGGERS')}\n")
        trigger_fmt = "%.0f %.0f %.0f %.0f %.0f\n"
        append = chunks.append
        for k, data in target.trigger_library.data.items():
            event_type, channel, delay, duration = data
            append(trigger_fmt % (k, event_type, channel, 1e6 * delay, 1e6 * duration))
        w("\n")

    if len(target.label_set_library.data) != 0:
        w("# Extension specification for setting labels:\n")
        w("# id set labelstring\n")
        tid = target.get_extension_type_ID("LABELSET")
        w(f"extension LABELSET {tid}\n")
        label_fmt = "%.0f %.0f %s\n"
        registry = target._label_registry_inv
        append = chunks.append
        for k, data in target.label_set_library.data.items():
            label_num = int(data[1])
            append(label_fmt % (k, data[0], registry.get(label_num, f"CUSTOM_{label_num}")))
        w("\n")

    if len(target.label_inc_library.data) != 0:
        w("# Extension specification for setting labels:\n")
        w("# id set labelstring\n")
        tid = target.get_extension_type_ID("LABELINC")
        w(f"extension LABELINC {tid}\n")
        label_fmt = "%.0f %.0f %s\n"
        registry = target._label_registry_inv
        append = chunks.append
        for k, data in target.label_inc_library.data.items():
            label_num = int(data[1])
            append(label_fmt % (k, data[0], registry.get(label_num, f"CUSTOM_{label_num}")))
        w("\n")

    if len(target.rf_shim_library.data) != 0:
        w("# Extension specification for RF shimming:\n")
        w("# id num_chan factor magn_c1 phase_c1 magn_c2 phase_c2 ...\n")
        w(f"extension RF_SHIMS {target.get_extension_type_ID('RF_SHIMS')}\n")

        for k in target.rf_shim_library.data:
            shim_vector_length = len(target.rf_shim_library.data[k])
            id_format_str = "{:d} {:d}" + "".join(" {:g}" for _ in range(shim_vector_length)) + "\n"
            s = id_format_str.format(k, int(0.5 * shim_vector_length), *target.rf_shim_library.data[k])
            w(s)
        w("\n")

    if len(target.rotation_library.data) != 0:
        w("# Extension specification for rotation events:\n")
        w("# id RotQuat0 RotQuatX RotQuatY RotQuatZ\n")
        w(f"extension ROTATIONS {target.get_extension_type_ID('ROTATIONS')}\n")
        id_format_str = "{:.0f} {:12g} {:12g} {:12g} {:12g}\n"
        for k in target.rotation_library.data:
            s = id_format_str.format(k, *target.rotation_library.data[k])
            w(s)
        w("\n")

    if len(target.shape_library.data) != 0:
        w("# Sequence Shapes\n")
        w("[SHAPES]\n\n")
        for k in target.shape_library.data:
            shape_data = target.shape_library.data[k]
            w(f"shape_id {k:.0f}\n")
            w(f"num_samples {shape_data[0]:.0f}\n")
            w(("{:.9g}\n" * len(shape_data[1:])).format(*shape_data[1:]))
            w("\n")

    body = "".join(chunks)
    signature = None
    if create_signature:
        signature = hashlib.md5(body.encode("utf-8")).hexdigest()
        sig_chunk = (
            "\n[SIGNATURE]\n"
            "# This is the hash of the Pulseq file, calculated right before the [SIGNATURE] section was added\n"
            "# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n"
            "# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be stripped away for recalculating/verification)\n"
            "Type md5\n"
            f"Hash {signature}\n"
        )
        body += sig_chunk

    payload = body.encode("utf-8")
    if output is None:
        return payload

    if hasattr(output, "write"):
        try:
            output.write(payload)
        except TypeError:
            output.write(body)
        return signature

    file_name = Path(output)
    if file_name.suffix != ".seq":
        file_name = file_name.with_suffix(file_name.suffix + ".seq")
    file_name.write_bytes(payload)
    return signature


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

    overrides = {attr: float(definitions[key]) for key, attr in _RASTER_DEFINITIONS.items() if key in definitions}
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
