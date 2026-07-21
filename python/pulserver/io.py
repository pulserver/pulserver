"""I/O helpers for pulserver sequence writing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO
from warnings import warn

import numpy as np
from pypulseq.supported_labels_rf_use import get_supported_rf_uses

from pulserver.pypulseq._sequence import _RF_USE_CODE_TO_CHAR


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

    if target.block_events:
        last_block_id = next(reversed(target.block_events))
        last_row = target.block_events[last_block_id]
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
    id_format_width = "{:" + str(len(str(len(target.block_events)))) + "d}"
    id_format_str = id_format_width + " {:3d} {:3d} {:3d} {:3d} {:3d} {:2d} {:2d}\n"
    for block_counter in target.block_events:
        block_duration = target.block_durations[block_counter] / target.block_duration_raster
        block_duration_rounded = round(block_duration)
        assert abs(block_duration_rounded - block_duration) < 1e-6
        s = id_format_str.format(
            *(
                block_counter,
                block_duration_rounded,
                *target.block_events[block_counter][1:],
            )
        )
        w(s)
    w("\n")

    if len(target.rf_library.data) != 0:
        w("# Format of RF events:\n")
        w("# id ampl. mag_id phase_id time_shape_id center delay freqPPm phasePPM freq phase use\n")
        w("# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   Hz   rad  ..\n")
        w(f"# Field \"use\" is the initial of: {' '.join(get_supported_rf_uses()).strip()}\n")
        w("[RF]\n")
        id_format_str = "{:.0f} {:12g} {:.0f} {:.0f} {:.0f} {:g} {:g} {:g} {:g} {:g} {:g} {:s}\n"
        for k in target.rf_library.data:
            lib_data1 = target.rf_library.data[k][0:4]
            lib_data2 = target.rf_library.data[k][6:10]
            center = target.rf_library.data[k][4] * 1e6
            delay = round(target.rf_library.data[k][5] / target.rf_raster_time) * target.rf_raster_time * 1e6
            use_type = target.rf_library.type.get(k, "u")
            use_char = (
                _RF_USE_CODE_TO_CHAR.get(int(use_type), "u") if isinstance(use_type, int | np.integer) else use_type
            )
            s = id_format_str.format(k, *lib_data1, center, delay, *lib_data2, use_char)
            w(s)
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
        id_format_str = "{:.0f} {:12g} {:12g} {:12g} {:.0f} {:.0f} {:.0f}\n"
        for k in grad_keys[arb_grad_mask]:
            arb_id = int(target.grad_library.data[int(k)][0])
            data = target.arb_library.data[arb_id]
            s = id_format_str.format(k, *data[:5], round(data[5] * 1e6))
            w(s)
        w("\n")

    if np.any(trap_grad_mask):
        w("# Format of trapezoid gradients:\n")
        w("# id amplitude rise flat fall delay\n")
        w("# ..      Hz/m   us   us   us    us\n")
        w("[TRAP]\n")
        id_format_str = "{:2.0f} {:12g} {:3.0f} {:4.0f} {:3.0f} {:3.0f}\n"
        for k in grad_keys[trap_grad_mask]:
            trap_id = int(target.grad_library.data[int(k)][0])
            data = np.array(target.trap_library.data[trap_id], dtype=float)
            data[1:] = np.round(1e6 * data[1:])
            s = id_format_str.format(k, *data)
            w(s)
        w("\n")

    if len(target.adc_library.data) != 0:
        w("# Format of ADC events:\n")
        w("# id num dwell delay freqPPM phasePPM freq phase phase_id\n")
        w("# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n")
        w("[ADC]\n")
        id_format_str = "{:.0f} {:.0f} {:.0f} {:.0f} {:g} {:g} {:g} {:g} {:.0f}\n"
        for k in target.adc_library.data:
            data = np.multiply(target.adc_library.data[k][0:8], [1, 1e9, 1e6, 1, 1, 1, 1, 1])
            s = id_format_str.format(k, *data)
            w(s)
        w("\n")

    if len(target.extensions_library.data) != 0:
        w("# Format of extension lists:\n")
        w("# id type ref next_id\n")
        w("# next_id of 0 terminates the list\n")
        w("# Extension list is followed by extension specifications\n")
        w("[EXTENSIONS]\n")
        id_format_str = "{:.0f} {:.0f} {:.0f} {:.0f}\n"
        for k in target.extensions_library.data:
            s = id_format_str.format(k, *np.round(target.extensions_library.data[k]))
            w(s)
        w("\n")

    if len(target.trigger_library.data) != 0:
        w("# Extension specification for digital output and input triggers:\n")
        w("# id type channel delay (us) duration (us)\n")
        w(f"extension TRIGGERS {target.get_extension_type_ID('TRIGGERS')}\n")
        id_format_str = "{:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n"
        for k in target.trigger_library.data:
            s = id_format_str.format(k, *np.round(target.trigger_library.data[k] * np.array([1, 1, 1e6, 1e6])))
            w(s)
        w("\n")

    if len(target.label_set_library.data) != 0:
        w("# Extension specification for setting labels:\n")
        w("# id set labelstring\n")
        tid = target.get_extension_type_ID("LABELSET")
        w(f"extension LABELSET {tid}\n")
        id_format_str = "{:.0f} {:.0f} {}\n"
        for k in target.label_set_library.data:
            value = target.label_set_library.data[k][0]
            label_num = int(target.label_set_library.data[k][1])
            label_id = target._label_registry_inv.get(label_num, f"CUSTOM_{label_num}")
            s = id_format_str.format(k, value, label_id)
            w(s)
        w("\n")

    if len(target.label_inc_library.data) != 0:
        w("# Extension specification for setting labels:\n")
        w("# id set labelstring\n")
        tid = target.get_extension_type_ID("LABELINC")
        w(f"extension LABELINC {tid}\n")
        id_format_str = "{:.0f} {:.0f} {}\n"
        for k in target.label_inc_library.data:
            value = target.label_inc_library.data[k][0]
            label_num = int(target.label_inc_library.data[k][1])
            label_id = target._label_registry_inv.get(label_num, f"CUSTOM_{label_num}")
            s = id_format_str.format(k, value, label_id)
            w(s)
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
