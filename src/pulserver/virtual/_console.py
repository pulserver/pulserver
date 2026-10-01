"""A scanner console's gateway to pulserver: the design calls, the exam's localizer and the virtual scanner, in process or over a WebSocket."""

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

from .._plugins import PluginPath, directories, names
from ._command import motion_arguments, subject_motion

#: The design calls a console forwards, answered as ``pulserver design`` answers them.
DESIGN_CALLS = ("list", "validate", "generate", "import")

# Spacings, 1 mm apart, a console tries in keeping a scan within its isochromats.
_COARSER = 100

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
    one of the virtual scanner's coils, and scans a stored design on it.
    Images return as DICOM from the reconstruction proxy at ``recon``, or from
    the plugins of ``recon_plugins`` run in this process as the proxy runs
    them; with neither, a scan returns no images.

    Parameters
    ----------
    plugins
        Directories of the scanner-sequence plugins, in search order: a name
        is the plugin of the first directory holding ``<name>.py``.
    limits
        Text of the ``[Limits]`` block every design is made under; its ``B0``
        is the virtual magnet's field.
    store
        Design store.
    recon
        ``(host, port)`` of the reconstruction proxy.
    recon_plugins
        Directories of reconstruction plugin files, ``<plugin>.py``, in search
        order, which reconstruct each scan in this process
        (:class:`~pulserver.proxy.LocalReconstruction`).
    push
        Recon-side intake each design is pushed to.
    spacing
        Finest isochromat spacing of the phantom, in metres.
    max_isochromats
        Most isochromats a scan is simulated on. A scan is simulated on the
        isochromats in the slabs its excitation pulses excite
        (:func:`~pulserver.virtual.excited`), at the finest spacing, from
        ``spacing`` up in steps of 1 mm, that keeps no more of them than this;
        at ``spacing`` without it.
    coil
        Name of the coil an exam is started with unless it names another.
    fields
        Directory of the coils' field maps, ``<coil>.npz``, and of the VOPs of
        those that transmit, ``<coil>_vops.npz``, as mariepy writes them for
        BrainWeb's head. Every exam is then on BrainWeb, and every design is
        made under the VOPs of the exam's transmit coil. Without it, the coils
        are :data:`~pulserver.virtual.COILS`.
    speed
        Scan time elapsed per wall-clock second once the scan's simulation is
        far enough ahead of its clock; as fast as it is simulated without it.
    device
        Device a scan's ADC windows are read and its runs of repetitions
        carried on, as :class:`~pulserver.virtual.Isochromats` takes it; the
        engine does both itself without one.
    spins
        Isochromats per voxel, spread over the T2' line of its tissue.
    voxel
        Where a voxel's isochromats lie: ``"point"``, at its centre, or
        ``"box"``, over it, as the phantom's ``isochromats`` places them.
    diffusion
        Whether BrainWeb's tissue classes diffuse, as
        :attr:`~pulserver.virtual.BrainWeb.DIFFUSION` gives them.
    motion
        The subject's motion, as :class:`~pulserver.virtual.Isochromats`
        takes it, on the clock of each scan; at rest without one. Moving
        isochromats play every block alone.
    """

    def __init__(
        self,
        *,
        plugins: PluginPath,
        limits: str,
        store: Path | str,
        recon: tuple[str, int] | None = None,
        recon_plugins: PluginPath | None = None,
        push: str | None = None,
        spacing: float = 1e-3,
        max_isochromats: int | None = None,
        coil: str = "body",
        fields: Path | str | None = None,
        speed: float | None = None,
        device: str | None = None,
        spins: int = 1,
        voxel: str = "point",
        diffusion: bool = False,
        motion: Any = None,
    ) -> None:
        from ..host._blocks import parse_limits
        from ..proxy import LocalReconstruction
        from ._coils import coils

        if recon is not None and recon_plugins is not None:
            raise ValueError(
                "a console reconstructs through a proxy or in this process, not both"
            )
        self.plugins = directories(plugins)
        self.limits = limits
        self.store = Path(store)
        self.recon = recon
        self.local = (
            None
            if recon_plugins is None
            else LocalReconstruction(self.store, recon_plugins)
        )
        self.push = push
        self.spacing = spacing
        self.max_isochromats = max_isochromats
        self.speed = speed
        self.device = device
        self.spins = spins
        self.voxel = voxel
        self.diffusion = diffusion
        self.motion = motion
        self.field_t = float(parse_limits(limits)["B0"])
        self.fields = None if fields is None else Path(fields)
        self._coils = coils(self.fields, field_t=self.field_t)
        self.coil = self._coil(coil)
        self.subject = ""
        # BrainWeb, kept from one exam to the next with what it has loaded, and
        # the thread its field map is computed in after an exam's localizer.
        self._brainweb: Any = None
        self._warming: threading.Thread | None = None
        self.phantom = self._phantom("")
        # The exam's isochromats between its scans, beside the slabs they were
        # kept in, and how many exams have started, so that a scan hands back
        # only those of the exam in progress.
        self._held = threading.Lock()
        self._isochromats: tuple[Any, Any] | None = None
        self._exams = 0

    def design(self, call: str, plugin: str | None = None, block: str = "") -> dict:
        """Answer a design call; a generated or imported design's id is ``design``."""
        from ..host._server import answer

        if call not in DESIGN_CALLS:
            raise ValueError(
                f"a console makes the design calls {DESIGN_CALLS}, not {call!r}"
            )
        request = {
            "call": call,
            "plugins": [str(directory) for directory in self.plugins],
            "plugin": plugin,
            "limits": self._limits(),
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
        return names(self.plugins)

    def coils(self) -> list[dict[str, Any]]:
        """Return each coil an exam can be started with: its ``name`` and its ``transmit`` and ``receive`` channels."""
        return [
            {
                "name": coil.name,
                "transmit": coil.transmit_channels,
                "receive": coil.receive_channels,
            }
            for coil in self._coils.values()
        ]

    def exam(self, subject: str, coil: str | None = None) -> list[bytes]:
        """Start an exam on the phantom ``subject`` names, in the coil ``coil`` names; return its three-plane localizer as DICOM files.

        ``brainweb`` names BrainWeb's normal brain, and so does every subject
        of a console with field maps; any other subject, the vials. Without
        ``coil``, the exam keeps the coil it had.

        Raises
        ------
        ValueError
            If ``coil`` names none of the virtual scanner's coils.
        """
        from ._localizer import localizer

        if coil is not None:
            self.coil = self._coil(coil)
        self.subject = subject
        self.phantom = self._phantom(subject)
        with self._held:
            self._exams += 1
            self._isochromats = None
        files = [
            _dicom_bytes(dataset)
            for dataset in localizer(
                self.phantom, field_t=self.field_t, subject=subject
            )
        ]
        self._warm()
        return files

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
        DICOM, ``{"dicom": base64, "name": file name}``, converting those it
        returns as MRD, and its texts as ``{"text": ...}``. With
        ``sound``, each clock also carries the span's sound as ``sound``,
        base64 of 16-bit little-endian stereo samples at ``rate`` Hz. At a
        speed, the scan is simulated ahead of its clock, and until the clock
        starts ``emit`` receives ``{"preparing": s}`` about twice a second,
        with the wall-clock time left before it does, or ``null`` before there
        is an estimate. The status is 1 when a text reports a refused or
        failed series, or when the scan is cancelled.

        A scan plays on the phantom's isochromats in the slabs its excitation
        pulses excite, at the spacing ``max_isochromats`` allows. Scans of an exam
        that excite the same slabs play on the same isochromats, each from
        equilibrium, which keeps the pulses the engine has computed; a scan
        started while another plays has isochromats of its own.
        """
        from ..host import DesignStore
        from ._region import excited

        if self.speed is not None:
            emit({"preparing": None})
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        region = excited(
            DesignStore(self.store).directory(design) / "sequence.seq", rotation
        )
        with self._held:
            held, self._isochromats = self._isochromats, None
            exam = self._exams
        if held is not None and held[0] == region:
            isochromats = held[1]
            isochromats.reset()
        else:
            if self._warming is not None:
                self._warming.join()
            isochromats = self.phantom.isochromats(
                self._spacing(region),
                field_t=self.field_t,
                region=region,
                coil=self.coil,
                spins=self.spins,
                voxel=self.voxel,
                motion=self.motion,
                seed=0,
                device=self.device,
            )
        try:
            return self._scan(
                design, isochromats, rotation, centre_mm, emit, cancelled, sound
            )
        finally:
            with self._held:
                if exam == self._exams:
                    self._isochromats = (region, isochromats)

    def _spacing(self, region: Any) -> float:
        """Return the finest spacing, from ``spacing`` up in steps of 1 mm, that keeps at most ``max_isochromats`` of the phantom's isochromats in ``region``.

        Raises
        ------
        ValueError
            If none within 10 cm of it does.
        """
        if self.max_isochromats is None:
            return self.spacing
        for step in range(_COARSER):
            spacing = self.spacing + 1e-3 * step
            kept = self.phantom.count(
                spacing, field_t=self.field_t, region=region, spins=self.spins
            )
            if kept <= self.max_isochromats:
                return spacing
        raise ValueError(
            f"the phantom holds more than {self.max_isochromats} isochromats at "
            f"every spacing from {1e3 * self.spacing:g} mm to {1e3 * spacing:g} mm"
        )

    def _limits(self) -> str:
        """Return the limits block of a design: the console's, with the VOP entries of the exam's coil."""
        from ..host._blocks import format_limits, parse_limits

        entries = self.coil.limits()
        if not entries:
            return self.limits
        return format_limits({**parse_limits(self.limits), **entries})

    def _coil(self, name: str) -> Any:
        if name not in self._coils:
            raise ValueError(
                f"the virtual scanner's coils are {sorted(self._coils)}, not {name!r}"
            )
        return self._coils[name]

    def _phantom(self, subject: str) -> Any:
        from . import BrainWeb
        from ._command import default_phantom

        if self.fields is not None or subject.strip().lower() == "brainweb":
            if self._brainweb is None:
                self._brainweb = BrainWeb(
                    diffusion=BrainWeb.DIFFUSION if self.diffusion else None
                )
            return self._brainweb
        return default_phantom()

    def _warm(self) -> None:
        """Compute the field map of the exam's BrainWeb in a thread of its own, unless it has one or is computing it."""
        brain = self.phantom
        if brain is not self._brainweb or not brain.susceptibility:
            return
        if "field_ppm" in vars(brain) or (
            self._warming is not None and self._warming.is_alive()
        ):
            return

        def compute() -> None:
            # A failure here is raised again by the scan that needs the map.
            with contextlib.suppress(Exception):
                brain.field_ppm  # noqa: B018

        self._warming = threading.Thread(
            target=compute, name="pulserver-field", daemon=True
        )
        self._warming.start()

    def _scan(
        self,
        design: str,
        isochromats: Any,
        rotation: np.ndarray,
        centre_mm: Sequence[float],
        emit: Callable[[dict], None],
        cancelled: Callable[[], bool],
        sound: bool,
    ) -> int:
        from ..host import DesignStore
        from . import SAMPLE_RATE, Scan
        from ._bloch import TOLERANCE

        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        scan = Scan(
            DesignStore(self.store).directory(design) / "sequence.seq",
            isochromats,
            rotation=rotation,
            default_shim=self.coil.default_shim,
            tolerance=TOLERANCE,
        )
        stopped = threading.Event()

        def preparing(left: float | None) -> None:
            if cancelled():
                raise _Cancelled
            emit({"preparing": left})

        def played() -> Iterator[np.ndarray]:
            chunks = scan.chunks(
                0.1, speed=self.speed, sound=sound, preparing=preparing
            )
            try:
                with contextlib.closing(chunks):
                    yield from spans(chunks)
            except _Cancelled:
                stopped.set()

        def spans(chunks: Iterator[Any]) -> Iterator[np.ndarray]:
            for chunk in chunks:
                if cancelled():
                    raise _Cancelled
                clock = {"clock": chunk.stop, "duration": scan.duration}
                if sound:
                    samples = np.round(32767 * np.clip(chunk.sound.T, -1.0, 1.0))
                    clock["sound"] = base64.b64encode(
                        samples.astype("<i2").tobytes()
                    ).decode()
                    clock["rate"] = SAMPLE_RATE
                emit(clock)
                yield from chunk.readouts

        # Closed however the scan ends, which stops its simulation before the
        # isochromats are handed on.
        with contextlib.closing(played()) as readouts:
            return self._acquire(design, readouts, rotation, centre_mm, emit, stopped)

    def _acquire(
        self,
        design: str,
        readouts: Iterator[np.ndarray],
        rotation: np.ndarray,
        centre_mm: Sequence[float],
        emit: Callable[[dict], None],
        stopped: threading.Event,
    ) -> int:
        """Send the readouts to the reconstruction, emitting what it returns; return its status."""
        import ismrmrd
        import pypulseqpp as pp

        from ..recon._runtime.mrd2dicom import DicomWithName, MrdDicomBuilder
        from . import send
        from ._client import _header, _series

        if self.recon is None and self.local is None:
            for _ in readouts:
                pass
            return 1 if stopped.is_set() else 0
        status = 0
        series = {
            "frequency_hz": pp.Opts().gamma * self.field_t,
            "position_mm": tuple(float(c) for c in centre_mm),
            "rotation": rotation,
        }
        convert = None

        def returned(item: Any) -> None:
            nonlocal convert, status
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

        if self.local is None:
            for item in send(self.recon, design, readouts, **series):
                returned(item)
        else:
            header, acquisitions = _series(
                design,
                readouts,
                series["frequency_hz"],
                series["position_mm"],
                rotation,
                None,
            )
            self.local.run(
                ismrmrd.xsd.CreateFromDocument(header), acquisitions, returned
            )
        return 1 if stopped.is_set() else status

    def answer(
        self,
        request: Mapping[str, Any],
        reply: Callable[[dict], None],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        """Answer one request of a console, handing ``reply`` each message of the answer.

        ``request`` carries its ``call`` and that call's fields, and the
        messages are those ``pulserver console`` sends over its WebSocket,
        without the request's ``id``: one for each call, except that a scan
        sends its clock, images and texts before ``{"done": status}``. A call
        that fails, or is not one of the console's, is answered with
        ``{"error": text}``. A scan stops once ``cancelled`` returns true.
        """
        call = request.get("call")
        try:
            if call == "plugins":
                reply({"plugins": self.plugin_names()})
            elif call == "coils":
                reply({"coils": self.coils()})
            elif call in DESIGN_CALLS:
                reply(
                    self.design(call, request.get("plugin"), request.get("block", ""))
                )
            elif call == "exam":
                coil = request.get("coil")
                files = self.exam(
                    str(request.get("subject", "")), None if coil is None else str(coil)
                )
                reply(
                    {"localizer": [base64.b64encode(f).decode("ascii") for f in files]}
                )
            elif call == "scan":
                status = self.scan(
                    str(request["design"]),
                    rotation=np.asarray(request.get("rotation", np.eye(3).ravel())),
                    centre_mm=request.get("centre_mm", (0.0, 0.0, 0.0)),
                    emit=reply,
                    cancelled=cancelled,
                    sound=bool(request.get("sound", False)),
                )
                reply({"done": status})
            else:
                reply({"error": f"unknown call {call!r}"})
        except Exception as error:
            reply({"error": f"{type(error).__name__}: {error}"})


class _Cancelled(Exception):
    """A scan cancelled while its simulation is under way."""


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
        ident = request.get("id")
        if request.get("call") == "scan":
            cancel.clear()
        console.answer(request, lambda message: reply(ident, message), cancel.is_set)

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


async def serve(
    console: Console, host: str, port: int, origins: Sequence[str] | None = None
) -> None:
    """Serve ``console`` on a WebSocket at ``host``:``port`` until cancelled.

    With ``origins``, a browser page is served only from one of those origins,
    such as ``https://pulserver.github.io``; a client that sends no
    ``Origin``, which a browser always sends, is served whatever they are.
    """
    import websockets

    async with websockets.serve(
        lambda websocket: _connection(console, websocket),
        host,
        port,
        max_size=None,
        origins=None if origins is None else [*origins, None],
    ):
        await asyncio.Future()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pulserver console",
        description="Serve a scanner console's calls to pulserver over a WebSocket.",
    )
    parser.add_argument(
        "--plugins",
        type=Path,
        action="append",
        required=True,
        help="plugin directory, repeatable; the first holding a plugin is used",
    )
    parser.add_argument(
        "--limits", type=Path, required=True, help="file holding a [Limits] block"
    )
    parser.add_argument("--store", type=Path, required=True, help="design store")
    recon = parser.add_mutually_exclusive_group()
    recon.add_argument("--recon", help="reconstruction proxy, HOST:PORT")
    recon.add_argument(
        "--recon-plugins",
        type=Path,
        action="append",
        help="reconstruction plugin directory, repeatable, run in this process "
        "instead of a proxy",
    )
    parser.add_argument("--push", help="recon-side intake each design is pushed to")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on")
    parser.add_argument("--port", type=int, default=8765, help="port to listen on")
    parser.add_argument(
        "--origin",
        action="append",
        dest="origins",
        help="origin of the browser pages served, repeatable; every origin without it",
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="finest isochromat spacing, in mm"
    )
    parser.add_argument(
        "--max-isochromats",
        type=int,
        default=2_000_000,
        help="most isochromats a scan is simulated on; the spacing is coarsened "
        "by 1 mm until the slabs a scan excites hold no more",
    )
    parser.add_argument(
        "--coil", default="body", help="coil an exam starts with unless it names one"
    )
    parser.add_argument(
        "--fields",
        type=Path,
        help="directory of the coils' field maps and VOPs, solved in BrainWeb's "
        "head, on which every exam then is; BART's coil models without it",
    )
    parser.add_argument(
        "--speed",
        type=float,
        help="scan time per second, once the simulation is far enough ahead; "
        "as fast as it is simulated without it",
    )
    parser.add_argument(
        "--device",
        help="torch device the ADC windows are read and the runs of repetitions "
        "carried on, such as cuda (the gpu extra); the engine does both without it",
    )
    parser.add_argument(
        "--spins",
        type=int,
        default=1,
        help="isochromats per voxel, spread over the T2' line of its tissue",
    )
    parser.add_argument(
        "--voxel",
        choices=("point", "box"),
        default="point",
        help="where a voxel's isochromats lie: at its centre, or over it, "
        "--spins a square number for the vials and a cube for BrainWeb",
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        help="BrainWeb's tissue classes diffuse, as BrainWeb.DIFFUSION gives them",
    )
    motion_arguments(parser)
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
        recon_plugins=args.recon_plugins,
        push=args.push,
        spacing=1e-3 * args.spacing,
        max_isochromats=args.max_isochromats,
        coil=args.coil,
        fields=args.fields,
        speed=args.speed,
        device=args.device,
        spins=args.spins,
        voxel=args.voxel,
        diffusion=args.diffusion,
        motion=subject_motion(args),
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(console, args.host, args.port, args.origins))
    return 0
