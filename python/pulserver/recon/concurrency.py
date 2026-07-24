"""RAM-based concurrency limit for ISMRMRD reconstruction sessions.

The default estimate of 48 GiB per reconstruction session is sized for modern
iterative / deep-learning MR reconstructions on a GE Premier/UHP-class server.
On a 156 GiB VRE box (~142 GiB available), the formula yields 2 slots:

    max(1, floor(142 x 0.8 / 48)) = 2 -> 96 GiB working set

Both ``per_recon_gb`` and the overall limit are overridable at runtime via CLI
flags (``--per-recon-gb``, ``--max-recon``) or the environment variables
``MRDSERVER_PER_RECON_GB`` / ``MRDSERVER_MAX_RECON``.
"""

__all__ = ["compute_max_concurrent"]

import logging
import math
import os

_DEFAULT_PER_RECON_GB: float = 48.0
_DEFAULT_HEADROOM_FRACTION: float = 0.8


def _available_ram_gb() -> float:
    """Return available RAM in GiB.

    Tries ``psutil`` first; falls back to ``os.sysconf`` (Linux/POSIX).
    Returns 0.0 if neither is available.
    """
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
    """Return the maximum number of simultaneous reconstruction sessions.

    Parameters
    ----------
    per_recon_gb : float
        Estimated RAM (GiB) consumed by one reconstruction session.
        Default: 48 GiB.
    headroom_fraction : float
        Fraction of available RAM allocated to reconstruction.
        Default: 0.8 (80 %).
    override : int or None
        If set and > 0, skip auto-detection and return this value directly.

    Returns
    -------
    int
        Number of concurrent reconstruction sessions allowed (minimum 1).
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
