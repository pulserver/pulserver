"""A beating heart scanned by a 2D Cartesian cine: phantom, heartbeat, readout timeline and the MRD stream of it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import ismrmrd
import numpy as np

#: Time of day the stream's time stamps start at, in ms (10:00).
CLOCK_START_MS = 36_000_000

#: Signal of each compartment relative to the blood pool, for a phantom drawn
#: without a tissue model.
DEFAULT_SIGNALS = {
    "blood": 1.0,
    "myocardium": 0.3,
    "liver": 0.25,
    "muscle": 0.2,
    "fat": 0.8,
    "lung": 0.03,
}


def contraction(since_r: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """Return the left ventricle's contraction, 0 at end diastole and 1 at end systole, ``since_r`` s after an R wave of a beat of ``rr`` s.

    Systole lasts ``0.35 sqrt(rr)`` s (Bazett) whatever the beat, ejecting
    along half a cosine; diastole takes the rest of the beat, its rapid filling
    in the first 40 %, then diastasis, then the atrial contraction in the
    last 15 % of the beat.
    """
    since_r = np.asarray(since_r, dtype=float)
    rr = np.broadcast_to(np.asarray(rr, dtype=float), since_r.shape)
    systole = 0.35 * np.sqrt(rr)
    out = np.zeros(since_r.shape)
    ejecting = since_r < systole
    out[ejecting] = 0.5 - 0.5 * np.cos(np.pi * since_r[ejecting] / systole[ejecting])
    diastole = ~ejecting
    s = (since_r[diastole] - systole[diastole]) / np.maximum(
        rr[diastole] - systole[diastole], 1e-6
    )
    filling = np.clip(s / 0.4, 0.0, 1.0)
    kick_start = 1.0 - 0.15 * rr[diastole] / (rr[diastole] - systole[diastole])
    kick = np.clip((s - kick_start) / np.maximum(1.0 - kick_start, 1e-6), 0.0, 1.0)
    out[diastole] = 0.15 + 0.85 * (0.5 + 0.5 * np.cos(np.pi * filling)) - 0.15 * kick
    out[diastole] = np.clip(out[diastole], 0.0, 1.0)
    return out


def r_waves(
    duration: float,
    rr: float = 0.857,
    variability: float = 0.03,
    ectopic: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return the R-wave times of a heartbeat over ``duration`` s, from a random offset in the first beat.

    Each interval is ``rr`` varied by a normal of ``variability`` of it; beat
    ``ectopic`` is premature, at 60 % of the interval, and the next
    compensates at 140 %.
    """
    generator = np.random.default_rng(seed)
    times = [generator.uniform(0.0, rr)]
    index = 0
    while times[-1] < duration + 2 * rr:
        interval = rr * (1.0 + variability * generator.standard_normal())
        if ectopic is not None and index == ectopic:
            interval = 0.6 * rr
        elif ectopic is not None and index == ectopic + 1:
            interval = 1.4 * rr
        times.append(times[-1] + interval)
        index += 1
    return np.asarray(times)


def heart_state(times: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """Return the contraction at each time of a heart beating at the R waves ``waves``."""
    times = np.asarray(times, dtype=float)
    beat = np.clip(np.searchsorted(waves, times, side="right") - 1, 0, waves.size - 2)
    return contraction(times - waves[beat], waves[beat + 1] - waves[beat])


def _disc(x, y, cx, cy, rx, ry=None, angle=0.0):
    ry = rx if ry is None else ry
    c, s = math.cos(angle), math.sin(angle)
    u = (x - cx) * c + (y - cy) * s
    v = -(x - cx) * s + (y - cy) * c
    return (u / rx) ** 2 + (v / ry) ** 2 < 1.0


def _paint(image: np.ndarray, mask: np.ndarray, value) -> None:
    """Set ``image`` to ``value`` inside ``mask``; an array ``value`` is read where ``mask`` is."""
    image[mask] = value[mask] if np.ndim(value) else value


def short_axis(
    matrix: int,
    state: float,
    signals: dict[str, complex] | None = None,
    oversampling: int = 2,
) -> np.ndarray:
    """Return a short-axis section of a torso at contraction ``state``, ``(matrix, matrix)``, the field of view 1.

    Drawn at ``oversampling`` times the matrix and low-passed to it in
    k-space, so the edges are band-limited as an acquisition sees them. The
    left ventricle's cavity shrinks with ``state`` while its myocardium keeps
    its area and thickens; the right ventricle shrinks with it. A signal is a
    number, or an array on the oversampled grid such as a bSSFP signal under
    a field map.
    """
    signals = {**DEFAULT_SIGNALS, **(signals or {})}
    n = matrix * oversampling
    axis = (np.arange(n) - n // 2) / n
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = np.zeros((n, n), dtype=np.complex64)
    # Anterior is up and the subject's left on the right of the image.
    torso = _disc(x, y, 0.0, 0.02, 0.44, 0.33)
    _paint(image, torso, signals["fat"])
    _paint(image, _disc(x, y, 0.0, 0.02, 0.415, 0.305), signals["muscle"])
    _paint(image, _disc(x, y, -0.2, 0.01, 0.17, 0.24, 0.15), signals["lung"])
    _paint(image, _disc(x, y, 0.2, 0.0, 0.17, 0.24, -0.15), signals["lung"])
    _paint(image, _disc(x, y, -0.15, 0.17, 0.17, 0.11, 0.3), signals["liver"])
    _paint(
        image, _disc(x, y, 0.0, 0.26, 0.045, 0.04), signals["muscle"] * 1.5
    )  # vertebra
    _paint(image, _disc(x, y, 0.06, 0.2, 0.024), signals["blood"])  # descending aorta
    cx, cy = 0.03, -0.03
    cavity = 0.085 * math.sqrt(1.0 - 0.6 * state)
    wall = math.sqrt(cavity**2 + 0.112**2 - 0.085**2)
    rv_scale = 1.0 - 0.35 * state
    rv = (cx - 0.11, cy - 0.015)
    # Epicardial fat, the right ventricle, then the left.
    _paint(
        image, _disc(x, y, cx - 0.04, cy, wall + 0.075, wall + 0.012), signals["fat"]
    )
    _paint(
        image, _disc(x, y, *rv, 0.075 * rv_scale + 0.01, 0.115), signals["myocardium"]
    )
    _paint(image, _disc(x, y, *rv, 0.075 * rv_scale, 0.105), signals["blood"])
    _paint(image, _disc(x, y, cx, cy, wall), signals["myocardium"])
    _paint(image, _disc(x, y, cx, cy, cavity), signals["blood"])
    for angle in (0.7, 2.3):
        px = cx + 0.75 * cavity * math.cos(angle)
        py = cy + 0.75 * cavity * math.sin(angle)
        _paint(image, _disc(x, y, px, py, 0.011 + 0.007 * state), signals["myocardium"])
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    low = n // 2 - matrix // 2
    spectrum = spectrum[low : low + matrix, low : low + matrix] / oversampling**2
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum))).astype(
        np.complex64
    )


def body_array(coils: int, matrix: int, seed: int = 0) -> np.ndarray:
    """Return the sensitivities ``(coils, matrix, matrix)`` of a ring of loops around the torso, half anterior and half posterior."""
    generator = np.random.default_rng(seed)
    axis = (np.arange(matrix) - matrix // 2) / matrix
    y, x = np.meshgrid(axis, axis, indexing="ij")
    maps = []
    for c in range(coils):
        row, col = divmod(c, coils // 2)
        px = -0.3 + 0.6 * col / max(coils // 2 - 1, 1)
        py = -0.38 if row == 0 else 0.42
        distance2 = (x - px) ** 2 + (y - py) ** 2
        magnitude = 1.0 / (1.0 + distance2 / 0.06)
        phase = generator.uniform(0, 2 * np.pi) + 2.0 * (x * px + y * py)
        maps.append(magnitude * np.exp(1j * phase))
    return np.asarray(maps, dtype=np.complex64)


def centred_fft(images: np.ndarray) -> np.ndarray:
    """Return the unitary 2D transform of ``(..., y, x)``, the centre at ``n // 2``."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)


@dataclass
class Timeline:
    """The readouts of a cine train, in play order.

    Attributes
    ----------
    times : ndarray
        Time of each readout's echo, s from the start of the scan.
    lines, phases, segments, slices : ndarray
        The ``LIN``, ``PHS``, ``SEG`` and ``SLC`` labels of each.
    triggers : list of float
        Times at which a trigger was met, s.
    """

    times: np.ndarray
    lines: np.ndarray
    phases: np.ndarray
    segments: np.ndarray
    slices: np.ndarray
    triggers: list = field(default_factory=list)


def timeline(seq, waves: np.ndarray, latency: float = 0.0) -> Timeline:
    """Return the readouts of ``seq`` as played against the R waves ``waves``.

    A block with a trigger waits for the next R wave, plus ``latency``, from
    where the train has got to; every other block follows the one before.
    """
    labels = seq.evaluate_labels(evolution="adc")
    t, times, triggers = 0.0, [], []
    for index in range(1, seq.num_blocks + 1):
        block = seq.get_block(index)
        if getattr(block, "trig", None):
            nxt = waves[np.searchsorted(waves, t)] + latency
            triggers.append(nxt)
            t = nxt
        if block.adc is not None:
            adc = block.adc
            times.append(t + adc.delay + adc.dwell * adc.num_samples / 2)
        t += block.block_duration

    def label(name):
        return np.atleast_1d(labels.get(name, np.zeros(len(times)))).astype(int)

    return Timeline(
        np.asarray(times),
        label("LIN"),
        label("PHS"),
        label("SEG"),
        label("SLC"),
        triggers,
    )


def header(
    coils: int,
    matrix: int,
    phases: int,
    parameters: dict[str, object],
    fov_mm: float = 340.0,
) -> SimpleNamespace:
    """Return a header of one Cartesian encoding space for a cine stream, stating ``parameters`` as recon parameters."""
    voxels = SimpleNamespace(x=matrix, y=matrix, z=1)
    fov = SimpleNamespace(x=fov_mm, y=fov_mm, z=8.0)
    limits = {
        "kspace_encoding_step_1": SimpleNamespace(
            minimum=0, maximum=matrix - 1, center=matrix // 2
        )
    }
    if phases > 1:
        limits["phase"] = SimpleNamespace(minimum=0, maximum=phases - 1, center=0)
    encoding = SimpleNamespace(
        encodedSpace=SimpleNamespace(matrixSize=voxels, fieldOfView_mm=fov),
        reconSpace=SimpleNamespace(matrixSize=voxels, fieldOfView_mm=fov),
        encodingLimits=SimpleNamespace(**limits),
        trajectory="cartesian",
    )
    strings = [
        SimpleNamespace(name=name, value=str(value))
        for name, value in parameters.items()
    ]
    return SimpleNamespace(
        encoding=[encoding],
        acquisitionSystemInformation=SimpleNamespace(
            receiverChannels=coils,
            coilLabel=[
                SimpleNamespace(coilNumber=c, coilName=f"B{c}") for c in range(coils)
            ],
        ),
        measurementInformation=SimpleNamespace(
            measurementID="cine", frameOfReferenceUID="1.2.3"
        ),
        userParameters=SimpleNamespace(
            userParameterLong=[],
            userParameterDouble=[],
            userParameterString=strings,
            userParameterBase64=[],
        ),
    )


def acquisitions(
    played: Timeline,
    waves: np.ndarray,
    kspace_of_state,
    *,
    states: int = 256,
    noise: float = 0.0,
    seed: int = 1,
) -> list:
    """Return the acquisitions of ``played``: each line of the heart's k-space at its echo, stamped with the time and the last R wave.

    ``kspace_of_state(state)`` gives the coils' centred k-space ``(coils, y,
    x)`` at a contraction; the contraction is drawn at ``states`` levels.
    Complex Gaussian noise of standard deviation ``noise`` is added to each
    sample.
    """
    generator = np.random.default_rng(seed)
    levels = np.linspace(0.0, 1.0, states)
    state = heart_state(played.times, waves)
    level = np.rint(state * (states - 1)).astype(int)
    cache: dict[int, np.ndarray] = {}
    out = []
    last = np.searchsorted(waves, played.times, side="right") - 1
    count = len(played.times)
    for r in range(count):
        if level[r] not in cache:
            cache[level[r]] = kspace_of_state(levels[level[r]])
        line = cache[level[r]][:, played.lines[r]]
        if noise:
            line = line + noise * (
                generator.standard_normal(line.shape)
                + 1j * generator.standard_normal(line.shape)
            ) / np.sqrt(2.0)
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(line.shape[1], line.shape[0])
        acquisition.data[:] = line
        acquisition.center_sample = line.shape[1] // 2
        acquisition.idx.kspace_encode_step_1 = int(played.lines[r])
        acquisition.idx.phase = int(played.phases[r])
        acquisition.idx.segment = int(played.segments[r])
        acquisition.idx.slice = int(played.slices[r])
        acquisition.acquisition_time_stamp = int(
            CLOCK_START_MS + round(played.times[r] * 1e3)
        )
        beat = waves[last[r]] if last[r] >= 0 else None
        stamps = [0, 0, 0]
        if beat is not None and beat >= 0:
            stamps[0] = int(CLOCK_START_MS + round(beat * 1e3))
        acquisition.physiology_time_stamp[:] = stamps
        if r == count - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        out.append(acquisition)
    return out


def joined(designed):
    """Return a designed chain's blocks in play order as one sequence; a sequence as it is."""
    import pypulseqpp as pp

    if not isinstance(designed, list):
        return designed
    seq = pp.Sequence(designed[0].system)
    for part in designed:
        for index in range(1, part.num_blocks + 1):
            seq.add_block(part.get_block(index))
    for key, value in designed[-1].definitions.items():
        if key != "TRSize":
            seq.set_definition(key=key, value=value)
    return seq
