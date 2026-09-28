"""A scanner console's gateway to pulserver: the design calls, the exam's localizer and the virtual scanner, over a WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: The design calls a console forwards, answered as ``pulserver design`` answers them.
DESIGN_CALLS = ("list", "validate", "generate", "import")

#: What each design call takes, as ``pulserver design`` passes it.
_CALL_INPUTS = {
    "list": ("plugins", "plugin"),
    "validate": ("plugins", "plugin", "limits", "input"),
    "generate": ("plugins", "plugin", "limits", "input", "store", "push"),
    "import": ("limits", "input", "store", "push"),
}


class Console:
    """What a scanner console asks of pulserver, answered in this process.

    A console lists a plugin's protocol, validates and generates designs with
    the text blocks the interpreter sends, starts an exam on a subject with
    one of the virtual scanner's :data:`~pulserver.virtual.COILS`, and scans
    a stored design on it. Images return as DICOM from the reconstruction
    proxy at ``recon``.

    Parameters
    ----------
    plugins
        Directory of the scanner-sequence plugins.
    limits
        Text of the ``[Limits]`` block every design is made under; its ``B0``
        is the virtual magnet's field.
    store
        Design store.
    recon
        ``(host, port)`` of the reconstruction proxy; without it a scan returns
        no images.
    push
        Recon-side intake each design is pushed to.
    spacing
        Isochromat spacing of the phantom, in metres.
    coil
        Name of the coil an exam is started with unless it names another.
    speed
        Scan time elapsed per wall-clock second; as fast as possible without it.
    """

    def __init__(
        self,
        *,
        plugins: Path | str,
        limits: str,
        store: Path | str,
        recon: tuple[str, int] | None = None,
        push: str | None = None,
        spacing: float = 1e-3,
        coil: str = "body",
        speed: float | None = None,
    ) -> None:
        from ..host._blocks import parse_limits

        self.plugins = Path(plugins)
        self.limits = limits
        self.store = Path(store)
        self.recon = recon
        self.push = push
        self.spacing = spacing
        self.coil = _coil(coil)
        self.speed = speed
        self.field_t = float(parse_limits(limits)["B0"])
        self.subject = ""
        self.phantom = _subject_phantom("")

    def design(self, call: str, plugin: str | None = None, block: str = "") -> dict:
        """Answer a design call; a generated or imported design's id is ``design``."""
        from ..host._server import answer

        if call not in DESIGN_CALLS:
            raise ValueError(
                f"a console makes the design calls {DESIGN_CALLS}, not {call!r}"
            )
        request = {
            "call": call,
            "plugins": str(self.plugins),
            "plugin": plugin,
            "limits": self.limits,
            "store": str(self.store),
            "push": self.push,
            "input": block,
        }
        status, reply = answer(
            {key: request[key] for key in ("call", *_CALL_INPUTS[call])}
        )
        answered: dict[str, Any] = {"status": status, "reply": reply}
        if status == 0 and call in ("generate", "import"):
            answered["design"] = reply.split()[1]
        return answered

    def plugin_names(self) -> list[str]:
        """Return the names of the plugins a console can list."""
        return sorted(path.stem for path in self.plugins.glob("*.py"))

    def coils(self) -> list[dict[str, Any]]:
        """Return each coil an exam can be started with: its ``name`` and its ``transmit`` and ``receive`` channels."""
        from . import COILS

        return [
            {
                "name": coil.name,
                "transmit": coil.transmit_channels,
                "receive": coil.receive_channels,
            }
            for coil in COILS.values()
        ]

    def exam(self, subject: str, coil: str | None = None) -> list[bytes]:
        """Start an exam on the phantom ``subject`` names, in the coil ``coil`` names; return its three-plane localizer as DICOM files.

        ``brainweb`` names BrainWeb's normal brain; any other subject, the
        vials. Without ``coil``, the exam keeps the coil it had.

        Raises
        ------
        ValueError
            If ``coil`` names none of the virtual scanner's coils.
        """
        from ._localizer import localizer

        if coil is not None:
            self.coil = _coil(coil)
        self.subject = subject
        self.phantom = _subject_phantom(subject)
        return [
            _dicom_bytes(dataset)
            for dataset in localizer(
                self.phantom, field_t=self.field_t, subject=subject
            )
        ]

    def scan(
        self,
        design: str,
        *,
        rotation: np.ndarray,
        centre_mm: Sequence[float],
        emit: Callable[[dict], None],
        cancelled: Callable[[], bool] = lambda: False,
        sound: bool = False,
    ) -> int:
        """Scan a stored design on the exam's phantom; return the reconstruction's status.

        ``emit`` receives the scan clock after each span played, as
        ``{"clock": s, "duration": s}``, then the reconstruction's images as
        DICOM, ``{"dicom": base64, "name": file name}``, converting those the
        proxy returns as MRD, and its texts as ``{"text": ...}``. With
        ``sound``, each clock also carries the span's sound as ``sound``,
        base64 of 16-bit little-endian stereo samples at ``rate`` Hz. The
        status is 1 when a text reports a refused or failed series, or when
        the scan is cancelled.
        """
        import ismrmrd
        import pypulseqpp as pp

        from ..host import DesignStore
        from ..recon._runtime.mrd2dicom import DicomWithName, MrdDicomBuilder
        from . import SAMPLE_RATE, Scan, send
        from ._client import _header

        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        scan = Scan(
            DesignStore(self.store).directory(design) / "sequence.seq",
            self.phantom.isochromats(
                self.spacing, field_t=self.field_t, coil=self.coil
            ),
            rotation=rotation,
            default_shim=self.coil.default_shim,
        )
        stopped = threading.Event()

        def played() -> Iterator[np.ndarray]:
            for chunk in scan.chunks(0.1, speed=self.speed, sound=sound):
                if cancelled():
                    stopped.set()
                    return
                clock = {"clock": chunk.stop, "duration": scan.duration}
                if sound:
                    samples = np.round(32767 * np.clip(chunk.sound.T, -1.0, 1.0))
                    clock["sound"] = base64.b64encode(
                        samples.astype("<i2").tobytes()
                    ).decode()
                    clock["rate"] = SAMPLE_RATE
                emit(clock)
                yield from chunk.readouts

        if self.recon is None:
            for _ in played():
                pass
            return 1 if stopped.is_set() else 0
        status = 0
        series = {
            "frequency_hz": pp.Opts().gamma * self.field_t,
            "position_mm": tuple(float(c) for c in centre_mm),
            "rotation": rotation,
        }
        convert = None
        for item in send(self.recon, design, played(), **series):
            if isinstance(item, ismrmrd.Image):
                if convert is None:
                    header = _header(
                        design, self.coil.receive_channels, series["frequency_hz"], None
                    )
                    convert = MrdDicomBuilder(ismrmrd.xsd.CreateFromDocument(header))
                item = convert(item)
            if isinstance(item, DicomWithName) and item.dset is not None:
                emit({"dicom": _dicom_base64(item.dset), "name": item.filename})
            elif isinstance(item, str):
                emit({"text": item})
                status = 1 if item.startswith("pulserver:") else status
        return 1 if stopped.is_set() else status


def _subject_phantom(subject: str) -> Any:
    from . import BrainWeb
    from ._command import default_phantom

    if subject.strip().lower() == "brainweb":
        return BrainWeb()
    return default_phantom()


def _coil(name: str) -> Any:
    from . import COILS

    if name not in COILS:
        raise ValueError(
            f"the virtual scanner's coils are {sorted(COILS)}, not {name!r}"
        )
    return COILS[name]


def _dicom_bytes(dataset: Any) -> bytes:
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _dicom_base64(dataset: Any) -> str:
    return base64.b64encode(_dicom_bytes(dataset)).decode("ascii")


async def _connection(console: Console, websocket: Any) -> None:
    """Answer one console's requests, each a JSON object carrying ``call`` and an ``id`` its replies repeat."""
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    running: set[asyncio.Future] = set()

    def reply(ident: Any, message: Mapping[str, Any]) -> None:
        text = json.dumps({"id": ident, **message})
        asyncio.run_coroutine_threadsafe(websocket.send(text), loop).result()

    def work(request: Mapping[str, Any]) -> None:
        ident, call = request.get("id"), request.get("call")
        try:
            if call == "plugins":
                reply(ident, {"plugins": console.plugin_names()})
            elif call == "coils":
                reply(ident, {"coils": console.coils()})
            elif call in DESIGN_CALLS:
                reply(
                    ident,
                    console.design(
                        call, request.get("plugin"), request.get("block", "")
                    ),
                )
            elif call == "exam":
                coil = request.get("coil")
                files = console.exam(
                    str(request.get("subject", "")),
                    None if coil is None else str(coil),
                )
                reply(
                    ident,
                    {"localizer": [base64.b64encode(f).decode("ascii") for f in files]},
                )
            elif call == "scan":
                cancel.clear()
                status = console.scan(
                    str(request["design"]),
                    rotation=np.asarray(request.get("rotation", np.eye(3).ravel())),
                    centre_mm=request.get("centre_mm", (0.0, 0.0, 0.0)),
                    emit=lambda message: reply(ident, message),
                    cancelled=cancel.is_set,
                    sound=bool(request.get("sound", False)),
                )
                reply(ident, {"done": status})
            else:
                reply(ident, {"error": f"unknown call {call!r}"})
        except Exception as error:
            reply(ident, {"error": f"{type(error).__name__}: {error}"})

    async for message in websocket:
        request = json.loads(message)
        if request.get("call") == "cancel":
            cancel.set()
            continue
        future = loop.run_in_executor(None, work, request)
        running.add(future)
        future.add_done_callback(running.discard)
    for future in list(running):
        with contextlib.suppress(Exception):
            await future


async def serve(console: Console, host: str, port: int) -> None:
    """Serve ``console`` on a WebSocket at ``host``:``port`` until cancelled."""
    import websockets

    async with websockets.serve(
        lambda websocket: _connection(console, websocket), host, port, max_size=None
    ):
        await asyncio.Future()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pulserver console",
        description="Serve a scanner console's calls to pulserver over a WebSocket.",
    )
    parser.add_argument("--plugins", type=Path, required=True, help="plugin directory")
    parser.add_argument(
        "--limits", type=Path, required=True, help="file holding a [Limits] block"
    )
    parser.add_argument("--store", type=Path, required=True, help="design store")
    parser.add_argument("--recon", help="reconstruction proxy, HOST:PORT")
    parser.add_argument("--push", help="recon-side intake each design is pushed to")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on")
    parser.add_argument("--port", type=int, default=8765, help="port to listen on")
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="isochromat spacing, in mm"
    )
    parser.add_argument(
        "--coil", default="body", help="coil an exam starts with unless it names one"
    )
    parser.add_argument(
        "--speed",
        type=float,
        help="scan time per second; as fast as possible without it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``pulserver console`` with ``argv`` until interrupted."""
    args = _parser().parse_args(argv)
    recon = None
    if args.recon is not None:
        host, _, port = args.recon.rpartition(":")
        recon = (host, int(port))
    console = Console(
        plugins=args.plugins,
        limits=args.limits.read_text(),
        store=args.store,
        recon=recon,
        push=args.push,
        spacing=1e-3 * args.spacing,
        coil=args.coil,
        speed=args.speed,
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(console, args.host, args.port))
    return 0
