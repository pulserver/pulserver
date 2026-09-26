"""The ``pulserver scan`` command: a design scanned on the virtual scanner, without a console."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import wave
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_DESCRIPTION = """\
Scan a phantom on the virtual scanner, as a console would. The design is
generated from a scanner-sequence plugin, or imported from a sequence file,
and checked and converted to its IR cache under --limits, as the design calls
do. Its cache is played on the phantom's isochromats in pypulseqpp's Bloch
simulation, and the series is written to an ISMRMRD file, streamed to a
reconstruction proxy, or both. The design call's reply is written to standard
output, and the scan clock to standard error.
"""

#: Rotations from the logical readout, phase and slice axes to the physical
#: x, y and z axes that ``--orientation`` names, by the physical direction of
#: each logical axis: axial reads along x, encodes phase along y and selects
#: along z; coronal reads along x, encodes phase along z and selects along -y;
#: sagittal reads along y, encodes phase along z and selects along x.
ORIENTATIONS = {
    "axial": np.eye(3),
    "coronal": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "sagittal": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
}

#: Scan time, in s, between two lines of the scan clock.
_SPAN = 0.5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pulserver scan", description=_DESCRIPTION)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seq", type=Path, help="sequence file to import")
    source.add_argument("--plugin", help="scanner-sequence plugin to generate from")
    parser.add_argument("--plugins", type=Path, help="directory of <plugin>.py")
    parser.add_argument(
        "--protocol",
        type=Path,
        help="file holding the protocol block to generate from; the plugin's "
        "defaults without one",
    )
    parser.add_argument(
        "--limits", type=Path, required=True, help="file holding the [Limits] block"
    )
    parser.add_argument(
        "--store", type=Path, help="directory of designs; a temporary one without one"
    )
    parser.add_argument("--push", help="URL of the design intake to push the design to")
    turned = parser.add_mutually_exclusive_group()
    turned.add_argument(
        "--orientation", choices=sorted(ORIENTATIONS), help="prescribed orientation"
    )
    turned.add_argument(
        "--rotation",
        type=float,
        nargs=9,
        metavar="R",
        help="rotation from logical to physical axes, row by row",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="field-of-view centre, in mm along the physical axes",
    )
    parser.add_argument(
        "--phantom",
        type=Path,
        help="JSON file of the phantom's ellipses, or brainweb for BrainWeb's "
        "normal brain (the brainweb extra); vials of several T1 and T2 without one",
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="isochromat spacing, in mm"
    )
    parser.add_argument("--coils", type=int, default=4, help="receive coils")
    parser.add_argument(
        "--speed",
        type=float,
        help="play the scan this many times as fast as a scanner; as fast as "
        "it is simulated without one",
    )
    parser.add_argument("--mrd", type=Path, help="ISMRMRD file to write the series to")
    parser.add_argument("--sound", type=Path, help="WAV file to write the sound to")
    parser.add_argument(
        "--recon", help="HOST:PORT of a reconstruction proxy to stream the series to"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(),
        help="directory the reconstruction's images are written to",
    )
    return parser


def default_phantom(coils: int = 1) -> Any:
    """Return seven vials of water in a circle, one of fat in their middle.

    The vials are disks 15 mm in radius. The water vials have T1 from 0.3 s to
    2.0 s and T2 from 0.04 s to 0.3 s, both increasing around the circle; the
    fat vial has a T1 of 0.35 s and a T2 of 0.1 s. The values span the range of
    tissues and are not those of any one of them.
    """
    from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM

    from . import Ellipse, Phantom

    t1 = np.geomspace(0.3, 2.0, 7)
    t2 = np.geomspace(0.04, 0.3, 7)
    angles = 2.0 * np.pi * np.arange(7) / 7
    vials = [
        Ellipse(
            (0.045 * np.cos(a), 0.045 * np.sin(a), 0.0),
            (0.015, 0.015),
            t1=float(r1),
            t2=float(r2),
        )
        for a, r1, r2 in zip(angles, t1, t2, strict=True)
    ]
    fat = Ellipse(
        (0.0, 0.0, 0.0), (0.015, 0.015), shift_ppm=FAT_SHIFT_PPM, t1=0.35, t2=0.1
    )
    return Phantom([*vials, fat], coils=coils)


def read_phantom(path: Path | str, coils: int = 1) -> Any:
    """Return the phantom a JSON file describes.

    The file holds an object whose ``ellipses`` entry lists the ellipses,
    each an object of the fields of :class:`~pulserver.virtual.Ellipse`, in
    its units: metres, radians, ppm and seconds.

    Raises
    ------
    ValueError
        If the file is not such an object.
    """
    from . import Ellipse, Phantom

    data = json.loads(Path(path).read_text())
    try:
        ellipses = [
            Ellipse(
                **{
                    **fields,
                    "centre": tuple(fields["centre"]),
                    "semi_axes": tuple(fields["semi_axes"]),
                }
            )
            for fields in data["ellipses"]
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} does not describe ellipses: {error}") from None
    return Phantom(ellipses, coils=coils)


def prescription(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation from logical to physical axes and the field-of-view offset along the logical axes, in mm."""
    if args.rotation is not None:
        rotation = np.asarray(args.rotation, dtype=float).reshape(3, 3)
    else:
        rotation = ORIENTATIONS[args.orientation or "axial"]
    return rotation, rotation.T @ np.asarray(args.center, dtype=float)


def _design_call(args: argparse.Namespace, store: Path) -> tuple[int, str]:
    """Import or generate the design, as ``pulserver design`` does; return its status and reply."""
    from ..host._blocks import format_import
    from ..host._server import answer
    from ..protocol import FOV_OFFSET, FOV_ROTATION, PROTOCOL_BEGIN, PROTOCOL_END

    rotation, offset = prescription(args)
    call: dict[str, Any] = {
        "limits": args.limits.read_text(),
        "store": str(store.absolute()),
        "push": args.push,
    }
    if args.seq is not None:
        offset_mm = tuple(float(value) for value in offset)
        call.update(
            call="import", input=format_import(args.seq.absolute(), offset_mm, rotation)
        )
        return answer(call)
    prescribed = {
        **dict(zip(FOV_OFFSET, offset, strict=True)),
        **dict(zip(FOV_ROTATION, rotation.ravel(), strict=True)),
    }
    given = [] if args.protocol is None else args.protocol.read_text().splitlines()
    kept = [
        line
        for line in given
        if line.strip() not in (PROTOCOL_BEGIN, PROTOCOL_END)
        and line.partition(": ")[0] not in prescribed
    ]
    lines = [f"{name}: {float(value)!r}" for name, value in prescribed.items()]
    call.update(
        call="generate",
        plugins=str(args.plugins.absolute()),
        plugin=args.plugin,
        input="\n".join([PROTOCOL_BEGIN, *kept, *lines, PROTOCOL_END]) + "\n",
    )
    return answer(call)


def _played(
    scan: Any, args: argparse.Namespace, audio: wave.Wave_write | None
) -> Iterator[np.ndarray]:
    """Yield the readouts of the scan as they are played, writing its sound and its clock."""
    for chunk in scan.chunks(_SPAN, speed=args.speed, sound=audio is not None):
        if audio is not None:
            audio.writeframes(np.round(32767.0 * chunk.sound.T).astype("<i2").tobytes())
        sys.stderr.write(f"{chunk.stop:9.2f} s of {scan.duration:.2f} s\n")
        yield from chunk.readouts


def _received(items: Sequence[Any], output: Path) -> int:
    """Write the reconstruction's images to ``output`` and its texts to standard output; return the exit status.

    Images go to ``images.h5`` and DICOM datasets to the files they are named
    by; the status is 1 when a text reports a refused or failed series.
    """
    import ismrmrd
    import ismrmrd.hdf5

    from ..recon._runtime.mrd2dicom import DicomWithName

    output.mkdir(parents=True, exist_ok=True)
    status = 0
    with contextlib.ExitStack() as stack:
        images = None
        for item in items:
            if isinstance(item, ismrmrd.Image):
                if images is None:
                    images = ismrmrd.hdf5.Dataset(
                        str(output / "images.h5"), "dataset", create_if_needed=True
                    )
                    stack.callback(images.close)
                images.append_image("images", item)
            elif isinstance(item, DicomWithName) and item.dset is not None:
                item.dset.save_as(output / item.filename)
            elif isinstance(item, str):
                sys.stdout.write(item.rstrip("\n") + "\n")
                status = 1 if item.startswith("pulserver:") else status
    return status


def main(argv: list[str] | None = None) -> int:
    """Run ``pulserver scan`` with ``argv``; return its exit status."""
    args = _parser().parse_args(argv)
    if args.plugin is not None and args.plugins is None:
        _parser().error("--plugin needs --plugins")
    with tempfile.TemporaryDirectory() as temporary:
        store = args.store or Path(temporary)
        status, reply = _design_call(args, store)
        sys.stdout.write(reply)
        if status != 0:
            return status
        return _scan(args, store, reply.split()[1])


def _scan(args: argparse.Namespace, store: Path, design: str) -> int:
    """Scan the stored design; return the exit status."""
    import pypulseqpp as pp

    from ..host import DesignStore
    from ..host._blocks import parse_limits
    from . import Scan, record, send
    from ._stream import SAMPLE_RATE

    rotation, _ = prescription(args)
    field = float(parse_limits(args.limits.read_text())["B0"])
    tissue = _phantom(args)
    scan = Scan(
        DesignStore(store).directory(design) / "sequence.seq",
        tissue.isochromats(1e-3 * args.spacing, field_t=field),
        rotation=rotation,
    )
    series = {
        "frequency_hz": pp.Opts().gamma * field,
        "position_mm": tuple(float(c) for c in args.center),
        "rotation": rotation,
    }
    with contextlib.ExitStack() as stack:
        audio = None
        if args.sound is not None:
            audio = stack.enter_context(wave.open(str(args.sound), "wb"))
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(int(SAMPLE_RATE))
        readouts = _played(scan, args, audio)
        status = 0
        if args.recon is None:
            acquired = list(readouts)
        else:
            acquired = []
            host, _, port = args.recon.rpartition(":")
            sent = _kept(readouts, acquired) if args.mrd is not None else readouts
            status = _received(
                send((host, int(port)), design, sent, **series), args.output
            )
    if args.mrd is not None:
        record(args.mrd, design, acquired, **series)
    return status


def _phantom(args: argparse.Namespace) -> Any:
    from . import BrainWeb

    if args.phantom is None:
        return default_phantom(args.coils)
    if str(args.phantom) == "brainweb":
        return BrainWeb(coils=args.coils)
    return read_phantom(args.phantom, args.coils)


def _kept(
    readouts: Iterator[np.ndarray], kept: list[np.ndarray]
) -> Iterator[np.ndarray]:
    for readout in readouts:
        kept.append(readout)
        yield readout
