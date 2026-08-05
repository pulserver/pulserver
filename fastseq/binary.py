"""Writer for the native Pulseq *binary* sequence file format.

The format is upstream Pulseq's own, not one invented here. Reference:
``refcode/pulseq/matlab/+mr/@Sequence/writeBinary.m`` (and ``readBinary.m``),
with the section codes from ``Sequence.m::getBinaryCodes``. Upstream's own
example names such a file ``.bin``.

Layout: an 8-byte header ``0x01 "pulseq" 0x02``, three ``int64`` version
fields, then a flat run of sections. Each section opens with an ``int64``
code ``0xFFFFFFFF_0000000N`` and an explicit ``int64`` entry count. That
count is the whole point: a reader never has to scan for section boundaries
or make a pass to find the largest id before it can size an array.

What it is *not* is smaller. A block is 32 bytes here against roughly 35
characters of text, so this trades a little disk for a lot of decode time --
the C reader spends about 25 ms where the text reader spends 350.

Precision differs from the text format, deliberately and in the safe
direction: amplitudes are written as ``float64`` and times as picosecond
``int64`` rather than being rounded to a printed decimal. A value read back
from a binary file can therefore differ in the last bits from the same value
read out of a ``.seq``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

__all__ = ['write_binary', 'FILE_HEADER', 'SECTION']

#: Magic bytes. Byte-order free, which is why the version fields below are
#: what a reader actually uses to detect endianness.
FILE_HEADER = bytes([1]) + b'pulseq' + bytes([2])

#: Section codes: 0xFFFFFFFF in the high word, an ordinal in the low word.
_PREFIX = 0xFFFFFFFF << 32
SECTION = {
    'definitions': _PREFIX | 1,
    'blocks': _PREFIX | 2,
    'rf': _PREFIX | 3,
    'gradients': _PREFIX | 4,
    'trapezoids': _PREFIX | 5,
    'adc': _PREFIX | 6,
    'delays': _PREFIX | 7,
    'shapes': _PREFIX | 8,
    'extensions': _PREFIX | 9,
    'triggers': _PREFIX | 10,
    'labelset': _PREFIX | 11,
    'labelinc': _PREFIX | 12,
    'softdelays': _PREFIX | 13,
    'rfshims': _PREFIX | 14,
    'rotations': _PREFIX | 15,
}

# Little-endian throughout, matching what MATLAB and NumPy produce on every
# machine this runs on. The C reader accepts either order regardless.
_I64 = struct.Struct('<q')
_I32 = struct.Struct('<i')
_F64 = struct.Struct('<d')
#: Section codes only. Their top bit is set (0xFFFFFFFF in the high word), so
#: they are negative as int64 and have to be packed unsigned to get the bit
#: pattern MATLAB's `bitshift` produces and `readBinary` switches on.
_U64 = struct.Struct('<Q')

#: Times are stored as integer picoseconds.
_PS = 1e12


def _ps(seconds) -> int:
    """Seconds to integer picoseconds, matching writeBinary's ``round``."""
    return int(round(float(seconds) * _PS))


def _definitions(defs: dict) -> list:
    """The definitions section body: count, then key/type/values per entry.

    Keys are null-terminated; the value count is a single byte, followed by a
    type tag (``c``/``i``/``f``) and then that many values.
    """
    out = [_I64.pack(len(defs))]
    for key in sorted(defs):
        value = defs[key]
        out.append(key.encode('ascii') + b'\0')

        if isinstance(value, str):
            payload = value.encode('ascii')
            if len(payload) > 255:
                raise ValueError(
                    f'definition {key!r} is {len(payload)} bytes; the format stores the '
                    'value count in one byte, so 255 is the maximum'
                )
            out.append(bytes([len(payload)]) + b'c' + payload)
            continue

        values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
        values = list(np.atleast_1d(values))
        if len(values) > 255:
            raise ValueError(f'definition {key!r} has {len(values)} values; the format allows 255')

        # An integer stays an integer: writing 64 as 64.0 would come back as a
        # float and print differently downstream.
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
            out.append(bytes([len(values)]) + b'i')
            out.extend(_I32.pack(int(v)) for v in values)
        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            out.append(bytes([len(values)]) + b'f')
            out.extend(_F64.pack(float(v)) for v in values)
        else:
            payload = ' '.join(str(v) for v in values).encode('ascii')
            out.append(bytes([len(payload)]) + b'c' + payload)
    return out


def _blocks(numbers: np.ndarray, table: np.ndarray, seconds: np.ndarray, raster: float) -> bytes:
    """The blocks section body, built as one array rather than per block.

    This is the section whose size is the length of the scan, so it is the one
    place where a Python-level loop would dominate the write.
    """
    n = numbers.size
    ticks = seconds / raster
    rounded = np.rint(ticks)
    if n and np.abs(rounded - ticks).max() >= 1e-6:
        worst = int(np.argmax(np.abs(rounded - ticks)))
        raise AssertionError(
            f'block {int(numbers[worst])} duration is not a multiple of '
            f'the block duration raster ({ticks[worst] * raster} s)'
        )

    # int64 duration then 6 x int32 = 32 bytes, laid out as a byte view so the
    # two field widths can be written into one buffer.
    buf = np.empty((n, 32), dtype=np.uint8)
    buf[:, :8] = rounded.astype('<i8').view(np.uint8).reshape(n, 8)
    buf[:, 8:] = np.ascontiguousarray(table[:, 1:7]).astype('<i4').view(np.uint8).reshape(n, 24)
    return _I64.pack(n) + buf.tobytes()


def _rf(library) -> list:
    """RF: id, amp, three shape ids, center and delay in ps, four f64, use."""
    keys = list(library.data)
    out = [_I64.pack(len(keys))]
    for k in keys:
        d = library.data[k]
        out.append(_I32.pack(int(k)))
        out.append(_F64.pack(float(d[0])))
        out.append(_I32.pack(int(d[1])) + _I32.pack(int(d[2])) + _I32.pack(int(d[3])))
        out.append(_I64.pack(_ps(d[4])) + _I64.pack(_ps(d[5])))
        out.extend(_F64.pack(float(v)) for v in d[6:10])
        out.append(library.type.get(k, 'u').encode('ascii')[:1] or b'u')
    return out


def _arbitrary_gradients(library, keys) -> list:
    out = [_I64.pack(len(keys))]
    for k in keys:
        d = library.data[k]
        out.append(_I32.pack(int(k)))
        out.extend(_F64.pack(float(v)) for v in d[0:3])
        out.append(_I32.pack(int(d[3])) + _I32.pack(int(d[4])))
        out.append(_I64.pack(_ps(d[5])))
    return out


def _trapezoids(library, keys) -> list:
    out = [_I64.pack(len(keys))]
    for k in keys:
        amplitude, rise, flat, fall, delay = library.data[k]
        out.append(_I32.pack(int(k)))
        out.append(_F64.pack(float(amplitude)))
        out.extend(_I64.pack(_ps(v)) for v in (rise, flat, fall, delay))
    return out


def _adc(library) -> list:
    keys = list(library.data)
    out = [_I64.pack(len(keys))]
    for k in keys:
        d = library.data[k]
        out.append(_I32.pack(int(k)))
        out.append(_I64.pack(int(round(float(d[0])))))
        out.append(_I64.pack(_ps(d[1])) + _I64.pack(_ps(d[2])))
        out.extend(_F64.pack(float(v)) for v in d[3:7])
        out.append(_I32.pack(int(d[7])))
    return out


def _shapes(library) -> list:
    """Shapes: id, uncompressed length, compressed length, float32 samples.

    The samples are the compressed (run-length) form exactly as the text
    format carries them -- the codec is unchanged, only its container is.
    """
    keys = list(library.data)
    out = [_I64.pack(len(keys))]
    for k in keys:
        shape = library.data[k]
        data = np.asarray(shape[1:], dtype='<f4')
        out.append(_I32.pack(int(k)))
        out.append(_I64.pack(int(round(float(shape[0])))))
        out.append(_I64.pack(data.size))
        out.append(data.tobytes())
    return out


def _extension_chains(rows: np.ndarray) -> bytes:
    """The extension list: id, type, ref, next_id -- the other per-scan section."""
    n = rows.shape[0]
    return _I64.pack(n) + np.ascontiguousarray(rows).astype('<i4').tobytes()


def write_binary(seq, output=None, *, check_timing: bool = False):
    """Write `seq` in the Pulseq binary format.

    `output` may be a path, a binary file object, or `None` to get the bytes
    back. Upstream names these files ``.bin``; a path with no suffix or a
    ``.seq`` suffix is given ``.bin`` here so a binary file is never mistaken
    for a text one by eye. (Readers should not care: pulserver's C reader
    dispatches on the magic bytes, not the name.)

    There is no signature section: the binary format has no equivalent of
    ``[SIGNATURE]``, so `create_signature` has no counterpart here.
    """
    plain = getattr(seq, '_seq', seq)
    rotations = getattr(seq, 'rotation_library', None)
    shims = getattr(seq, 'rf_shim_library', None)

    deferred = getattr(seq, '_deferred', None)
    if deferred is not None:
        table, seconds = deferred
        numbers = np.arange(1, table.shape[0] + 1, dtype=np.int64)
    else:
        n = len(plain.block_events)
        numbers = np.fromiter(plain.block_events, dtype=np.int64, count=n)
        table = np.stack(list(plain.block_events.values())) if n else np.zeros((0, 7), int)
        seconds = np.fromiter(plain.block_durations.values(), dtype=float, count=n)

    if check_timing:
        ok, errors = plain.check_timing()
        if not ok:
            from warnings import warn

            warn(f'write_binary(): {len(errors)} timing errors found in the sequence', stacklevel=2)

    plain.set_definition('TotalDuration', float(seconds.sum()))

    has_rotations = rotations is not None and len(rotations.data) != 0
    has_shims = shims is not None and len(shims.data) != 0

    from .sequence import required_revision

    chunks: list[bytes] = [FILE_HEADER]
    chunks.append(_I64.pack(int(plain.version_major)))
    chunks.append(_I64.pack(int(plain.version_minor)))
    chunks.append(_I64.pack(required_revision(plain, has_rotations, has_shims)))

    def section(name):
        chunks.append(_U64.pack(SECTION[name]))

    if len(plain.definitions) != 0:
        section('definitions')
        chunks.extend(_definitions(plain.definitions))

    section('blocks')
    chunks.append(_blocks(numbers, table, seconds, plain.block_duration_raster))

    if len(plain.rf_library.data) != 0:
        section('rf')
        chunks.extend(_rf(plain.rf_library))

    arbitrary = [k for k, kind in plain.grad_library.type.items() if kind == 'g']
    trapezoids = [k for k, kind in plain.grad_library.type.items() if kind == 't']
    if arbitrary:
        section('gradients')
        chunks.extend(_arbitrary_gradients(plain.grad_library, arbitrary))
    if trapezoids:
        section('trapezoids')
        chunks.extend(_trapezoids(plain.grad_library, trapezoids))

    if len(plain.adc_library.data) != 0:
        section('adc')
        chunks.extend(_adc(plain.adc_library))

    if len(plain.shape_library.data) != 0:
        section('shapes')
        chunks.extend(_shapes(plain.shape_library))

    deferred_chains = getattr(seq, '_deferred_chains', None)
    if deferred_chains is not None:
        rows = np.empty((deferred_chains.shape[0], 4), dtype=np.int64)
        rows[:, 0] = np.arange(1, deferred_chains.shape[0] + 1, dtype=np.int64)
        rows[:, 1:] = deferred_chains
    else:
        chains = plain.extensions_library.data
        rows = np.empty((len(chains), 4), dtype=np.int64)
        rows[:, 0] = np.fromiter(chains, dtype=np.int64, count=len(chains))
        for i, value in enumerate(chains.values()):
            rows[i, 1:] = np.rint(value)
    if rows.shape[0]:
        section('extensions')
        chunks.append(_extension_chains(rows))

    if len(plain.trigger_library.data) != 0:
        section('triggers')
        chunks.append(_I32.pack(int(plain.get_extension_type_ID('TRIGGERS'))))
        chunks.append(_I64.pack(len(plain.trigger_library.data)))
        for k, d in plain.trigger_library.data.items():
            kind, channel, delay, duration = d
            chunks.append(_I32.pack(int(k)) + _I32.pack(int(kind)) + _I32.pack(int(channel)))
            chunks.append(_I64.pack(_ps(delay)) + _I64.pack(_ps(duration)))

    for name, library in (
        ('labelset', plain.label_set_library),
        ('labelinc', plain.label_inc_library),
    ):
        if len(library.data) == 0:
            continue
        section(name)
        chunks.append(_I32.pack(int(plain.get_extension_type_ID(name.upper()))))
        chunks.append(_I64.pack(len(library.data)))
        for k, d in library.data.items():
            chunks.append(_I32.pack(int(k)) + _I32.pack(int(d[0])) + _I32.pack(int(d[1])))

    if len(plain.soft_delay_library.data) != 0:
        section('softdelays')
        chunks.append(_I32.pack(int(plain.get_extension_type_ID('DELAYS'))))
        chunks.append(_I64.pack(len(plain.soft_delay_library.data)))
        for k, d in plain.soft_delay_library.data.items():
            hint = str(d[3]).encode('ascii')
            chunks.append(_I32.pack(int(k)) + _I32.pack(int(d[0])))
            chunks.append(_I64.pack(_ps(d[1])) + _F64.pack(float(d[2])))
            chunks.append(_I32.pack(len(hint)) + hint)

    if has_shims:
        section('rfshims')
        chunks.append(_I32.pack(int(plain.get_extension_type_ID('RF_SHIMS'))))
        chunks.append(_I64.pack(len(shims.data)))
        for k, d in shims.data.items():
            chunks.append(_I32.pack(int(k)) + _I32.pack(len(d) // 2))
            chunks.extend(_F64.pack(float(v)) for v in d)

    if has_rotations:
        section('rotations')
        chunks.append(_I32.pack(int(plain.get_extension_type_ID('ROTATIONS'))))
        chunks.append(_I64.pack(len(rotations.data)))
        for k, d in rotations.data.items():
            chunks.append(_I32.pack(int(k)))
            chunks.extend(_F64.pack(float(v)) for v in d)

    payload = b''.join(chunks)

    if output is None:
        return payload
    if hasattr(output, 'write'):
        output.write(payload)
        return None

    path = Path(output)
    if path.suffix in ('', '.seq'):
        path = path.with_suffix('.bin')
    path.write_bytes(payload)
    return None
