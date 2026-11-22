"""Trace-weighted images and apparent diffusion coefficient maps of a diffusion-weighted series."""

from __future__ import annotations

__all__ = ["ADC_SERIES", "TRACE_SERIES", "DiffusionMaps", "adc_map", "trace_weighted"]

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..mrd import diffusion_table
from .plugin import ReconResult

#: ``series_index`` of the trace-weighted images and of the ADC map.
TRACE_SERIES, ADC_SERIES = 3, 4

#: The ADC map's unit, in mm²/s: its values are in 10⁻⁶ mm²/s.
ADC_UNIT = 1e-6

#: b-values closer than this, in s/mm², make one shell.
SHELL_TOLERANCE = 10.0


def trace_weighted(images: Sequence[np.ndarray]) -> np.ndarray:
    """Return the trace-weighted image of one shell: the geometric mean of its directions' images.

    The geometric mean of the images of directions that sample the shell
    evenly weights the signal by ``exp(-b ADC)``, the trace of the diffusion
    tensor over three, whatever its orientation.
    """
    stack = np.abs(np.stack(images)).astype(np.float64)
    return np.exp(np.log(np.maximum(stack, _floor(stack))).mean(axis=0))


def adc_map(b_values: Sequence[float], shells: Sequence[np.ndarray]) -> np.ndarray:
    """Return the apparent diffusion coefficient, in mm²/s, of the shells' log-linear fit.

    The slope of ``ln S`` against ``b`` over the shells, by least squares,
    with ``b`` in s/mm²; one shell above b = 0 gives ``ln(S0 / Sb) / b``.
    Every voxel is fitted: none is set aside or zeroed.
    """
    b = np.asarray(b_values, dtype=np.float64)
    stack = np.abs(np.stack(shells)).astype(np.float64)
    log = np.log(np.maximum(stack, _floor(stack)))
    centred = b - b.mean()
    slope = np.tensordot(centred, log - log.mean(axis=0), axes=1) / (centred @ centred)
    return -slope


def _floor(stack: np.ndarray) -> float:
    """Return the smallest signal a logarithm is taken of: far below the noise."""
    return max(float(stack.max()) * 1e-9, np.finfo(np.float64).tiny)


class DiffusionMaps:
    """Trace-weighted images and an ADC map of each slice, made once the slice holds every volume of its table.

    Each volume's magnitude image is added as it is made. When a slice holds
    an image of every volume of the diffusion table, its b = 0 volumes are
    averaged, each shell above b = 0 is made one trace-weighted image
    (:func:`trace_weighted`), series ``TRACE_SERIES``, and the shells give
    the ADC map (:func:`adc_map`), series ``ADC_SERIES``, in units of
    ``ADC_UNIT``. Both are magnitude images.

    Parameters
    ----------
    b_values : sequence of float
        b-value of each volume, by its repetition counter (s/mm²).
    """

    def __init__(self, b_values: Sequence[float]) -> None:
        self.b_values = np.asarray(b_values, dtype=np.float64)
        self.held: dict[int, dict[int, tuple[np.ndarray, Any]]] = {}

    @classmethod
    def from_header(cls, header: Any) -> DiffusionMaps | None:
        """Return the maps of the diffusion table an MRD header states; ``None`` when it states none or no b-value above zero."""
        table = diffusion_table(header)
        if table is None:
            return None
        b, _ = table
        if not np.any(b > SHELL_TOLERANCE):
            return None
        return cls(b)

    def shells(self) -> list[tuple[float, list[int]]]:
        """Return each shell's b-value and its volumes, b = 0 first."""
        found: list[tuple[float, list[int]]] = []
        for volume, b in enumerate(self.b_values):
            for shell, (value, volumes) in enumerate(found):
                if abs(b - value) <= SHELL_TOLERANCE:
                    volumes.append(volume)
                    count = len(volumes)
                    found[shell] = (value + (b - value) / count, volumes)
                    break
            else:
                found.append((float(b), [volume]))
        return sorted(found, key=lambda shell: shell[0])

    def add(
        self, location: int, repetition: int, image: np.ndarray, reference: Any
    ) -> list[ReconResult]:
        """Hold one volume's image of a slice; return the slice's maps once it holds every volume.

        ``reference`` is the acquisition the volume's image takes its geometry
        from. A trace-weighted image takes the reference of its shell's first
        volume, whose repetition counter names the shell's b-value, and the
        ADC map that of the first b = 0 volume.
        """
        volumes = len(self.b_values)
        held = self.held.setdefault(location, {})
        held[repetition % volumes] = (np.abs(np.asarray(image)), reference)
        if len(held) < volumes:
            return []
        del self.held[location]
        shells = self.shells()
        if shells[0][0] > SHELL_TOLERANCE:
            return []
        means = [
            np.mean([held[v][0] for v in volumes], axis=0)
            if b <= SHELL_TOLERANCE
            else trace_weighted([held[v][0] for v in volumes])
            for b, volumes in shells
        ]
        history = ["PULSERVER", "PYTHON", "DIFFUSION"]
        results = [
            ReconResult(
                trace.astype(np.float32),
                reference=held[volumes[0]][1],
                series_index=TRACE_SERIES,
                attributes={
                    "ImageProcessingHistory": history,
                    "SeriesDescriptionAdditional": f"_TRACEW_B{round(b)}",
                },
            )
            for (b, volumes), trace in zip(shells[1:], means[1:], strict=True)
        ]
        adc = adc_map([b for b, _ in shells], means) / ADC_UNIT
        results.append(
            ReconResult(
                adc.astype(np.float32),
                reference=held[shells[0][1][0]][1],
                series_index=ADC_SERIES,
                attributes={
                    "ImageProcessingHistory": history,
                    "SeriesDescriptionAdditional": "_ADC",
                },
            )
        )
        return results
