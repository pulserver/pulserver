"""Multi-echo 3D gradient echo: the ``pics`` image of each echo, then T2*, QSM and SWI with fast-qsm once every echo is in."""

from __future__ import annotations

__all__ = ["ANATOMIES", "PLUGIN", "SERIES", "QsmRecon"]

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ...mrd._metadata import user_parameter
from .._buffers import ReconData
from ..plugin import ReconContext, ReconResult
from .pics import IMAGINARY_SERIES, REAL_SERIES, PicsRecon

_log = logging.getLogger(__name__)

#: The anatomies of the ``QsmAnatomy`` recon parameter, by its value.
ANATOMIES = ("phantom", "brain", "body")

#: The series of each derived map, after the echoes' magnitude, real and
#: imaginary series of ``pics``.
SERIES = {
    "rss": 3,
    "t2star": 4,
    "qsm": 5,
    "swi": 6,
    "swi_mip": 7,
    "pcs": 8,
    "dcs": 9,
}

#: Gyromagnetic ratio of 1H over 2 pi, Hz/T.
GAMMA_HZ_T = 42.57747892e6

#: Sign of the phase against the frequency in the images of this
#: reconstruction: 1 when it grows with frequency (fast-qsm's convention).
#: The paramagnetic Gd vials of a phantom map positive when it is right.
PHASE_SIGN = 1

#: Where an installer puts FreeSurfer's SynthStrip weights, under the Python
#: environment's prefix.
SYNTHSTRIP_WEIGHTS = "share/fastqsm/synthstrip.1.pt"

#: Echo time of the SWI echo at 3 T, scaled by 3 T / B0.
SWI_TE_3T = 20e-3


class QsmRecon(PicsRecon):
    """Each echo as ``pics`` makes it, and once the last echo of a volume is in, the maps of the anatomy the sequence states.

    The anatomy is the ``QsmAnatomy`` recon parameter of the header (0
    phantom, 1 brain, 2 body), as ``gre_multiecho3d`` states it. The complex
    image of each echo, the real and imaginary series ``pics`` sends, is kept
    until the last echo of its slice and loop position closes; then, each its
    own series (:data:`SERIES`):

    - every anatomy: the root sum of squares of the echo magnitudes and T2*
      (ms) of the whole object, from the field fit of fast-qsm;
    - phantom: QSM by PDF and MEDI+0 referenced to the water
      (``fastqsm.pipeline.run(anatomy="phantom", phantom_inversion="medi")``);
    - brain: SWI and its minimum intensity projection from the echo nearest
      :data:`SWI_TE_3T` scaled to B0, QSM by V-SHARP and iLSQR inside the
      SynthStrip brain mask, and the DECOMPOSE paramagnetic and diamagnetic
      component susceptibilities;
    - body: QSM by mask-free TFI (``anatomy="body"``).

    The B0 direction in the image axes is that of the unit's reference
    acquisition, B0 is the header's resonance frequency over gamma, the echo
    times are the header's, and the phase sign is :data:`PHASE_SIGN`. A
    partial-Fourier axis leaves ``pics`` a real image without the field's
    phase, so the scan must sample k-space fully. The
    SynthStrip weights are those of :func:`synthstrip_weights`. Without
    fast-qsm the plugin makes the echoes alone and logs why.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._echoes: dict[tuple[int, ...], dict[int, np.ndarray]] = {}

    def spawn(self) -> QsmRecon:
        clone = super().spawn()
        clone._echoes = {}
        return clone

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> list[ReconResult] | ReconResult | None:
        result = super().recon(context, branch, data)
        te = _echo_times(context.header)
        if not isinstance(result, list) or len(te) < 2:
            return result
        parts = {r.series_index: r.data for r in result}
        if REAL_SERIES not in parts or IMAGINARY_SERIES not in parts:
            return result
        image = np.asarray(parts[REAL_SERIES]) + 1j * np.asarray(
            parts[IMAGINARY_SERIES]
        )
        idx = data.data.reference.idx
        echo = int(idx.contrast)
        position = (int(idx.slice), int(idx.phase), int(idx.set), int(idx.repetition))
        held = self._echoes.setdefault(position, {})
        held[echo] = image.astype(np.complex64)
        if len(held) == len(te):
            del self._echoes[position]
            echoes = np.stack([held[e] for e in sorted(held)])
            result.extend(self.maps(context, data, echoes, te))
        return result

    def maps(
        self,
        context: ReconContext,
        data: ReconData,
        echoes: np.ndarray,
        te: list[float],
    ) -> list[ReconResult]:
        """Return the derived series of ``(echoes, z, y, x)`` complex images at echo times ``te`` (s)."""
        try:
            import torch
            from fastqsm import pipeline, swi
        except ImportError as error:
            _log.warning("qsm: fast-qsm is not installed, echoes only (%s)", error)
            return []
        header = context.header
        anatomy = _anatomy(header)
        b0 = _b0(header)
        reference = data.data.reference
        # B0 is the z axis of the patient frame the direction cosines are in.
        b0_dir = (
            float(reference.read_dir[2]),
            float(reference.phase_dir[2]),
            float(reference.slice_dir[2]),
        )
        voxel = _voxel_size(header)
        # fast-qsm takes (echo, x, y, z); the images are (echo, z, y, x).
        xyz = np.ascontiguousarray(np.transpose(echoes, (0, 3, 2, 1)))
        magnitude = np.abs(xyz).astype(np.float32)
        phase = (np.angle(xyz) * PHASE_SIGN).astype(np.float32)
        device = context.device or ("cuda" if torch.cuda.is_available() else "cpu")

        def back(volume: Any) -> np.ndarray:
            array = (
                volume.detach().cpu().numpy() if hasattr(volume, "detach") else volume
            )
            return np.ascontiguousarray(
                np.transpose(np.asarray(array, np.float32), (2, 1, 0))
            )

        def result(name: str, volume: Any, tag: str) -> ReconResult:
            return ReconResult(
                back(volume),
                series_index=SERIES[name],
                attributes={
                    "ImageProcessingHistory": ["PULSERVER", "PYTHON", "FASTQSM"],
                    "SeriesDescriptionAdditional": f"_{tag}",
                },
            )

        rss = np.sqrt((magnitude.astype(np.float64) ** 2).sum(0)).astype(np.float32)
        out = [result("rss", rss, "RSS")]
        out.append(result("t2star", _t2star_ms(magnitude, phase, te), "T2STAR"))
        common = {"b0": b0, "b0_dir": b0_dir, "device": device}
        if anatomy == "phantom":
            run = pipeline.run(
                magnitude,
                phase,
                te,
                voxel,
                anatomy="phantom",
                phantom_inversion="medi",
                **common,
            )
            out.append(result("qsm", run.maps["chi"], "QSM_MEDI0"))
        elif anatomy == "body":
            run = pipeline.run(magnitude, phase, te, voxel, anatomy="body", **common)
            out.append(result("qsm", run.maps["chi"], "QSM"))
        else:
            target = SWI_TE_3T * 3.0 / b0
            nearest = int(np.argmin([abs(t - target) for t in te]))
            sw = swi(xyz[nearest] if PHASE_SIGN == 1 else np.conj(xyz[nearest]), voxel)
            out.append(result("swi", sw.swi, "SWI"))
            out.append(result("swi_mip", sw.mip, "SWI_MIP"))
            run = pipeline.run(
                magnitude,
                phase,
                te,
                voxel,
                anatomy="head",
                synthstrip_weights=synthstrip_weights(),
                **common,
            )
            out.append(result("qsm", run.maps["chi_masked"], "QSM"))
            out.append(result("pcs", run.maps["pcs"], "PCS"))
            out.append(result("dcs", run.maps["dcs"], "DCS"))
        _log.info(
            "qsm: %s maps in %s s on %s", anatomy, run.info.get("total_s"), device
        )
        return out


def synthstrip_weights() -> str | None:
    """Return the SynthStrip weights: ``FASTQSM_SYNTHSTRIP_WEIGHTS``, else :data:`SYNTHSTRIP_WEIGHTS` in the environment's prefix when it is there."""
    named = os.environ.get("FASTQSM_SYNTHSTRIP_WEIGHTS")
    if named:
        return named
    shipped = Path(sys.prefix) / SYNTHSTRIP_WEIGHTS
    return str(shipped) if shipped.is_file() else None


def _echo_times(header: Any) -> list[float]:
    """Return the header's echo times, in s, ascending."""
    parameters = getattr(header, "sequenceParameters", None)
    te = getattr(parameters, "TE", None) or []
    return sorted(float(value) * 1e-3 for value in te)


def _anatomy(header: Any) -> str:
    value = user_parameter(header, "QsmAnatomy", None)
    if value is None:
        return ANATOMIES[0]
    index = int(float(value))
    if not 0 <= index < len(ANATOMIES):
        raise ValueError(
            f"QsmAnatomy {value!r} is not one of 0 to {len(ANATOMIES) - 1}"
        )
    return ANATOMIES[index]


def _b0(header: Any) -> float:
    return float(header.experimentalConditions.H1resonanceFrequency_Hz) / GAMMA_HZ_T


def _voxel_size(header: Any) -> tuple[float, float, float]:
    space = header.encoding[0].reconSpace
    fov, matrix = space.fieldOfView_mm, space.matrixSize
    return (
        float(fov.x) / int(matrix.x),
        float(fov.y) / int(matrix.y),
        float(fov.z) / max(int(matrix.z), 1),
    )


def _t2star_ms(magnitude: np.ndarray, phase: np.ndarray, te: list[float]) -> np.ndarray:
    """Return T2* in ms over the object fast-qsm finds from the magnitude, 0 elsewhere and where R2* <= 0."""
    import fastqsm
    import torch
    from fastqsm.lab.compartments import object_mask
    from fastqsm.lab.fieldfit import core_fit_field, echo_difference

    rss = np.sqrt((magnitude.astype(np.float64) ** 2).sum(0))
    obj = torch.from_numpy(object_mask(rss))
    mag, pha = torch.from_numpy(magnitude), torch.from_numpy(phase)
    dphi = fastqsm.unwrap(echo_difference(mag[:2], pha[:2]), mag[0], obj)
    fit = core_fit_field(mag, pha, dphi, te, mag.device)
    r2 = torch.as_tensor(fit.r2star).float()
    t2 = torch.where((r2 > 0) & obj, 1e3 / r2.clamp_min(1e-6), torch.zeros_like(r2))
    return t2.numpy()


PLUGIN = QsmRecon()
