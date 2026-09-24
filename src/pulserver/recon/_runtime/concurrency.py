"""Number of concurrent reconstructions the host memory and its GPUs allow."""

__all__ = ["Slot", "Slots", "compute_max_concurrent", "gpu_devices", "slot_devices"]

import logging
import math
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass

_DEFAULT_PER_RECON_GB: float = 48.0
_DEFAULT_HEADROOM_FRACTION: float = 0.8


def _available_ram_gb() -> float:
    """Available memory in GiB from ``psutil``, else ``sysconf``; 0.0 when neither answers."""
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass
    try:
        page_size: int = os.sysconf("SC_PAGE_SIZE")
        avail_pages: int = os.sysconf("SC_AVPHYS_PAGES")
        return page_size * avail_pages / (1024**3)
    except (AttributeError, ValueError, OSError):
        return 0.0


def compute_max_concurrent(
    per_recon_gb: float = _DEFAULT_PER_RECON_GB,
    headroom_fraction: float = _DEFAULT_HEADROOM_FRACTION,
    override: int | None = None,
) -> int:
    """Return how many reconstructions may run at once.

    ``floor(available * headroom_fraction / per_recon_gb)``, at least 1, and 1
    when available memory cannot be read.

    Parameters
    ----------
    per_recon_gb
        Memory budgeted for one reconstruction, in GiB.
    headroom_fraction
        Fraction of available memory given to reconstructions.
    override
        Returned as is when positive.
    """
    if override is not None and override > 0:
        logging.info("MRD concurrency limit: %d slot(s)  [manual override]", override)
        return override

    avail_gb = _available_ram_gb()
    if avail_gb <= 0.0:
        logging.warning(
            "Could not determine available RAM — defaulting to 1 concurrent recon slot"
        )
        return 1

    slots = max(1, math.floor(avail_gb * headroom_fraction / per_recon_gb))
    logging.info(
        "MRD concurrency limit: %d slot(s)  "
        "(%.1f GiB available x %.0f%% headroom / %.1f GiB per recon)",
        slots,
        avail_gb,
        headroom_fraction * 100,
        per_recon_gb,
    )
    return slots


def gpu_devices() -> tuple[str, ...]:
    """Return the CUDA devices a worker sees, as ``cuda:<index>``.

    ``CUDA_VISIBLE_DEVICES``, when set, names them, numbered from 0 as CUDA
    numbers them; otherwise they are the GPUs ``nvidia-smi`` lists. Empty
    without either, so the proxy never imports a GPU library to find out.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        count = sum(1 for entry in visible.split(",") if entry.strip())
    else:
        count = _listed_gpus()
    return tuple(f"cuda:{index}" for index in range(count))


def slot_devices(slots: int | None = None, per_gpu: int = 1) -> tuple[str | None, ...]:
    """Return the device each slot holds; ``None`` for the host.

    Without a GPU, one ``None`` per slot :func:`compute_max_concurrent` allows.
    With GPUs, ``per_gpu`` slots per device, taken in turn across them, at most
    as many as host memory allows; ``slots``, when given, is the count instead,
    in turn across the devices.
    """
    devices = gpu_devices()
    if not devices:
        return (None,) * compute_max_concurrent(override=slots)
    count = (
        slots
        if slots is not None and slots > 0
        else min(len(devices) * max(1, per_gpu), compute_max_concurrent())
    )
    return tuple(devices[index % len(devices)] for index in range(count))


@dataclass(frozen=True, eq=False)
class Slot:
    """One reconstruction's share of the host, and the device it runs on; ``None`` for the host."""

    device: str | None


class Slots:
    """The slots series take while they reconstruct, each on its device."""

    def __init__(self, devices: Sequence[str | None]) -> None:
        self._free = [Slot(device) for device in devices]
        self._condition = threading.Condition()

    def take(self, *, wait: bool) -> Slot | None:
        """Take a free slot; without ``wait``, ``None`` when every slot is taken."""
        with self._condition:
            while not self._free:
                if not wait:
                    return None
                self._condition.wait()
            return self._free.pop(0)

    def release(self, slot: Slot) -> None:
        """Return a slot :meth:`take` gave."""
        with self._condition:
            self._free.append(slot)
            self._condition.notify()


def _listed_gpus() -> int:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return 0
    try:
        listed = subprocess.run(  # noqa: S603 -- a fixed query, no caller input
            [executable, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in listed.splitlines() if line.strip().isdigit())
