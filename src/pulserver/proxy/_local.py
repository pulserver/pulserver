"""Reconstruction of a series in this process, as the proxy's workers reconstruct it."""

from __future__ import annotations

__all__ = ["LocalReconstruction"]

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from .._plugins import PluginPath, directories
from ..recon import ReconContext, load_plugin
from ..recon._runtime.application import run_application
from ..recon._runtime.exam import ExamCacheManager
from ..recon._runtime.readers import deserialize_config
from ._designs import DesignCache
from ._enrich import enrich_header, header_fov_offset_m
from ._proxy import _config_plugin, _enriched, _plugin_path


class LocalReconstruction:
    """Reconstructs series in this process, one at a time, as :class:`ReconProxy`'s workers do.

    For a host that starts no processes, such as a browser tab. A series is
    checked and enriched as the proxy checks and enriches a stream, and
    reconstructed on the host by the reconstruction plugin its config names.
    The series of one exam share an :class:`~pulserver.recon.ExamCache` held
    in memory.

    Parameters
    ----------
    store
        Directory of designs, as the design calls write it; read only.
    plugins
        Directories of reconstruction plugin files, ``<plugin>.py``, in search
        order.
    """

    def __init__(self, store: Path | str, plugins: PluginPath) -> None:
        self.designs = DesignCache(store)
        self.plugins = directories(plugins)
        self.exams = ExamCacheManager()

    def run(
        self,
        header: Any,
        items: Iterable[Any],
        send: Callable[[Any], None],
        config: str = "",
    ) -> bool:
        """Reconstruct one series, handing ``send`` what it emits; return whether it was reconstructed.

        ``header`` is the series' parsed MRD header, enriched in place;
        ``items`` are its acquisitions and waveforms in play order; ``config``
        is the config text a client would send. ``send`` receives the
        reconstruction's images, DICOM datasets and texts as they are emitted.
        A series the proxy would refuse ends with ``pulserver:`` and the
        reason, and a reconstruction that fails with ``pulserver: <plugin>
        failed:`` and the error, as the proxy's client receives them.
        """
        try:
            design = self.designs.resolve(header)
            plugin = _config_plugin(config)
            path = _plugin_path(self.plugins, plugin)
            enrich_header(header, design.table)
            reconstruction = load_plugin(path)
        except Exception as error:
            send(f"pulserver: {error}")
            return False
        refused: list[Exception] = []

        def enriched() -> Iterator[Any]:
            try:
                yield from _enriched(items, design, header_fov_offset_m(header))
            except ValueError as error:
                refused.append(error)
                raise

        with self.exams.lease(header) as exam:
            context = ReconContext(
                header=header, exam=exam, config=deserialize_config(config, "default")
            )
            try:
                run_application(reconstruction, _Series(enriched(), send), context)
            except Exception as error:
                send(
                    f"pulserver: {refused[0]}"
                    if refused
                    else f"pulserver: {plugin} failed: {error}"
                )
                return False
        return True


class _Series:
    """A series as :func:`run_application` reads it: its items, and where its outputs go."""

    def __init__(self, items: Iterator[Any], send: Callable[[Any], None]) -> None:
        self._items = items
        self.send = send

    def __iter__(self) -> Iterator[Any]:
        return self._items
