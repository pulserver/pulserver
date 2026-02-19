"""Ceq structure definition."""

__all__ = ["Ceq", "PulseqBlock"]

from dataclasses import dataclass
from types import SimpleNamespace

import struct
import numpy as np
import pypulseq as pp

from . import _autoseg

CHANNEL_ENUM = {"osc0": 0, "osc1": 1, "ext1": 2}
SEGMENT_RINGDOWN_TIME = 116 * 1e-6  # TODO: doesn't have to be hardcoded


@dataclass
class PulseqShapeArbitrary:
    n_samples: int
    raster: float
    time: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray | None = None
    amplitude: float | None = None

    def __post_init__(self):
        self.magnitude = np.asarray(self.magnitude, dtype=np.float32)
        self.time = (
            np.asarray(self.time, dtype=np.float32) if self.time is not None else None
        )
        self.phase = (
            np.asarray(self.phase, dtype=np.float32) if self.phase is not None else None
        )

        # determine amplitude and normalize waveform
        self.amplitude = abs(self.magnitude).max()
        self.magnitude /= self.amplitude

    def to_bytes(self, endian=">") -> bytes:
        _bytes = struct.pack(endian + "i", self.n_samples) + struct.pack(
            endian + "f", self.raster
        )

        # add time
        if self.time is not None:
            _bytes += self.time.astype(endian + "f4").tobytes()

        # add magnitude
        _bytes += self.magnitude.astype(endian + "f4").tobytes()

        # add phase
        if self.phase is not None:
            _bytes += self.phase.astype(endian + "f4").tobytes()

        # add amplitude
        _bytes += struct.pack(endian + "f", self.amplitude)

        return _bytes


@dataclass
class PulseqShapeTrap:
    amplitude: float
    rise_time: float
    flat_time: float
    fall_time: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "f", self.amplitude)
            + struct.pack(endian + "f", self.rise_time)
            + struct.pack(endian + "f", self.flat_time)
            + struct.pack(endian + "f", self.fall_time)
        )


@dataclass
class PulseqRF:
    type: int
    complexflag: int
    wav: PulseqShapeArbitrary
    duration: float
    delay: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + struct.pack(endian + "h", self.complexflag)
            + self.wav.to_bytes(endian)
            + struct.pack(endian + "f", self.duration)
            + struct.pack(endian + "f", self.delay)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqRF":
        n_samples = data.signal.shape[0]

        # determine whether wave is arbitrary or extended trap
        dt = np.unique(np.diff(np.round(data.t * 1e6))) / 1e6

        if len(dt) == 1:  # uniform raster -> arbitrary shape
            type = 1
            raster = dt.item()
            time = None
        else:  # non-uniform raster -> extended trapezoid
            type = 2
            raster = 0.0
            time = data.t

        sum_rf_real = sum(abs(data.signal.real))
        sum_rf_imag = sum(abs(data.signal.imag))

        if sum_rf_real > 100 * sum_rf_imag:
            complexflag = 0
            rho = data.signal.real
            theta = None
        else:
            complexflag = 1
            rho = np.abs(data.signal)
            theta = np.angle(data.signal)

        return cls(
            type=type,
            complexflag=complexflag,
            wav=PulseqShapeArbitrary(n_samples, raster, time, rho, theta),
            duration=data.shape_dur,
            delay=data.delay,
        )


@dataclass
class PulseqGrad:
    type: int
    delay: float
    shape: PulseqShapeArbitrary | PulseqShapeTrap

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + struct.pack(endian + "f", self.delay)
            + self.shape.to_bytes(endian)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqGrad":
        if data.type == "trap":
            type = 1
            shape_obj = PulseqShapeTrap(
                data.amplitude, data.rise_time, data.flat_time, data.fall_time
            )
        elif data.type == "grad":
            n_samples = data.waveform.shape[0]

            # determine whether wave is arbitrary or extended trap
            dt = np.unique(np.diff(np.round(data.tt * 1e6))) / 1e6

            if len(dt) == 1:  # uniform raster -> arbitrary shape
                type = 2
                raster = dt.item()
                time = None
            else:  # non-uniform raster -> extended trapezoid
                type = 3
                raster = 0.0
                time = data.tt

            waveform = data.waveform
            shape_obj = PulseqShapeArbitrary(n_samples, raster, time, waveform)

        return cls(type=type, delay=data.delay, shape=shape_obj)


@dataclass
class PulseqADC:
    type: int
    num_samples: int
    dwell: float
    delay: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + struct.pack(endian + "i", self.num_samples)
            + struct.pack(endian + "f", self.dwell)
            + struct.pack(endian + "f", self.delay)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqADC":
        return cls(
            type=1,
            num_samples=data.num_samples,
            dwell=data.dwell,
            delay=data.delay,
        )


@dataclass
class PulseqTrig:
    type: int
    channel: int
    delay: float
    duration: float

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.type)
            + struct.pack(endian + "i", self.channel)
            + struct.pack(endian + "f", self.delay)
            + struct.pack(endian + "f", self.duration)
        )

    @classmethod
    def from_struct(cls, data: SimpleNamespace) -> "PulseqTrig":
        return cls(
            type=1,
            channel=CHANNEL_ENUM[data.channel],
            delay=data.delay,
            duration=data.duration,
        )


class PulseqBlock:
    """Pulseq block structure."""

    def __init__(
        self,
        ID: int,
        rf: SimpleNamespace = None,
        gx: SimpleNamespace = None,
        gy: SimpleNamespace = None,
        gz: SimpleNamespace = None,
        adc: SimpleNamespace = None,
        trig: SimpleNamespace = None,
        delay: SimpleNamespace = None,
    ) -> "PulseqBlock":
        self.ID = ID
        args = [rf, gx, gy, gz, adc, trig, delay]
        args = [arg for arg in args if arg is not None]
        self.duration = pp.calc_duration(*args)
        self.rf = PulseqRF.from_struct(rf) if rf else None
        self.gx = PulseqGrad.from_struct(gx) if gx else None
        self.gy = PulseqGrad.from_struct(gy) if gy else None
        self.gz = PulseqGrad.from_struct(gz) if gz else None
        self.adc = PulseqADC.from_struct(adc) if adc else None
        self.trig = PulseqTrig.from_struct(trig) if trig else None

    def to_bytes(self, endian=">") -> bytes:  # noqa
        bytes_data = struct.pack(endian + "i", self.ID) + struct.pack(
            endian + "f", self.duration
        )

        # RF Event
        if self.rf:
            bytes_data += self.rf.to_bytes(endian)
        else:
            bytes_data += struct.pack(endian + "h", 0)

        # Gradient Events
        for grad in [self.gx, self.gy, self.gz]:
            if grad:
                bytes_data += grad.to_bytes(endian)
            else:
                bytes_data += struct.pack(endian + "h", 0)

        # ADC Event
        if self.adc:
            bytes_data += self.adc.to_bytes(endian)
        else:
            bytes_data += struct.pack(endian + "h", 0)

        # Trigger Event
        if self.trig:
            bytes_data += self.trig.to_bytes(endian)
        else:
            bytes_data += struct.pack(endian + "h", 0)

        return bytes_data


class Segment:
    """Ceq segment."""

    def __init__(self, segment_id: int, block_ids: list[int]):
        self.segment_id = segment_id
        self.n_blocks_in_segment = len(block_ids)
        self.block_ids = np.asarray(block_ids, dtype=np.int16)

    def to_bytes(self, endian=">") -> bytes:
        return (
            struct.pack(endian + "h", self.segment_id)
            + struct.pack(endian + "h", self.n_blocks_in_segment)
            + self.block_ids.astype(endian + "i2").tobytes()
        )


class Ceq:
    """CEQ structure."""

    def __init__(
        self,
        parent_blocks: list[PulseqBlock],
        loop: list[list],
        sections_edges: list[list[int]],
    ):
        loop = np.asarray(loop, dtype=np.float32)
        segments = _build_segments(loop, sections_edges)

        # build CEQ structure
        self.n_parent_blocks = len(parent_blocks)
        self.parent_blocks = parent_blocks
        self.n_segments = len(segments)
        self.segments = segments
        self.n_max = loop.shape[0]
        self.n_columns_in_loop_array = loop.shape[1] - 2  # discard "hasrot", "hasadc"
        self.loop = loop[:, :-2]

        # Safety, RF scaling and scan duration info
        self.max_rf_power = 0.0
        self.max_b1 = _find_b1_max(parent_blocks)
        self.max_grad = _find_grad_max(parent_blocks)
        self.max_slew = _find_slew_max(parent_blocks)
        self.duration = _calc_duration(self.loop[:, 0], self.loop[:, 9])
        self.n_readouts = int(np.sum(loop[:, -2]))
        self.n_gain = 0

    def to_bytes(self, endian=">") -> bytes:  # noqa
        bytes_data = struct.pack(endian + "h", self.n_parent_blocks)
        for block in self.parent_blocks:
            bytes_data += block.to_bytes(endian)
        bytes_data += struct.pack(endian + "h", self.n_segments)
        for segment in self.segments:
            bytes_data += segment.to_bytes(endian)
        bytes_data += struct.pack(endian + "i", self.n_max)
        bytes_data += struct.pack(endian + "h", self.n_columns_in_loop_array)
        bytes_data += self.loop.astype(endian + "f4").tobytes()

        bytes_data += struct.pack(endian + "f", self.max_rf_power)
        bytes_data += struct.pack(endian + "f", self.max_b1)
        bytes_data += struct.pack(endian + "f", self.max_grad)
        bytes_data += struct.pack(endian + "f", self.max_slew)
        bytes_data += struct.pack(endian + "f", self.duration)
        bytes_data += struct.pack(endian + "i", self.n_readouts)
        bytes_data += struct.pack(endian + "i", self.n_gain)

        return bytes_data

    def export(self, dformat="file"):  # noqa
        if dformat == "bytes":
            return self.to_bytes(endian=">")
        elif dformat == "file":
            return self.to_bytes(endian="<")


# %% local subroutines
def _build_segments(loop, sections_edges):
    hasrot = np.ascontiguousarray(loop[:, -1]).astype(int)

    # get rotation matrices
    rotmat = np.ascontiguousarray(loop[:, 10:19])

    # get label for each rotation matrix
    _, arr_ind, arr_inv = np.unique(
        rotmat, return_inverse=True, return_index=True, axis=0
    )
    mapper = np.zeros(len(arr_ind), dtype=int)
    mapper[np.argsort(arr_ind)] = np.arange(len(arr_ind))
    rotlabel = mapper[arr_inv]

    # get parent block ids
    parent_block_id = np.ascontiguousarray(loop[:, 1]).astype(int)

    # build section edges
    if not sections_edges:
        sections_edges = [0]
    sections_edges = np.stack((sections_edges, sections_edges[1:] + [-1]), axis=-1)

    # loop over sections and find segment definitions
    seg_definitions = []

    # fill sections from 0 to n-1
    n_sections = len(sections_edges)
    for n in range(n_sections - 1):
        section_start, section_end = sections_edges[n]
        _seg_definition = _autoseg.find_segment_definitions(
            parent_block_id[section_start:section_end]
        )
        seg_definitions.extend(_seg_definition)

    # fill last section
    section_start = sections_edges[-1][0]
    if parent_block_id[section_start:].size != 0:
        _seg_definition = _autoseg.find_segment_definitions(
            parent_block_id[section_start:]
        )
        seg_definitions.extend(_seg_definition)

    # split rotated segments (segment must be rotated as a whole, find discontinuities in rotations)
    if (hasrot < 0).any():

        # initialize splitted segments
        splitted_segments = []
        for n in range(n_sections - 1):
            section_start, section_end = sections_edges[n]

            _seg_definition = seg_definitions[n]
            _segment_size = len(_seg_definition)
            _rotlabel = rotlabel[section_start:section_end]
            _rotlabel = _rotlabel.reshape(
                len(_rotlabel) // _segment_size, _segment_size
            )
            _rotchange = (np.diff(_rotlabel, axis=1) != 0).astype(int)
            _rotchange = np.unique(_rotchange, axis=0)
            if _rotchange.shape[0] != 1:
                raise ValueError("Irregular rotation pattern!")

            # get sub-segments
            _split_end = np.where(_rotchange.squeeze())[0]
            _split_start = np.concatenate(([0], _split_end + 1))
            _subsegments = []
            for m in range(len(_split_end)):
                _subsegments.append(
                    _seg_definition[_split_start[m] : _split_end[m] + 1]
                )
            _subsegments.append(_seg_definition[_split_start[-1] :])
            splitted_segments.extend(_subsegments)

        # fill last section
        section_start = sections_edges[-1][0]
        _seg_definition = seg_definitions[-1]
        _segment_size = len(_seg_definition)
        _rotlabel = rotlabel[section_start:]
        _rotlabel = _rotlabel.reshape(len(_rotlabel) // _segment_size, _segment_size)
        _rotchange = (np.diff(_rotlabel, axis=1) != 0).astype(int)
        _rotchange = np.unique(_rotchange, axis=0)
        if _rotchange.shape[0] != 1:
            raise ValueError("Irregular rotation pattern!")

        # get sub-segments
        _split_end = np.where(_rotchange.squeeze())[0]
        _split_start = np.concatenate(([0], _split_end + 1))
        _subsegments = []
        for m in range(len(_split_end)):
            _subsegments.append(_seg_definition[_split_start[m] : _split_end[m] + 1])
        _subsegments.append(_seg_definition[_split_start[-1] :])
        splitted_segments.extend(_subsegments)

        # update
        seg_definitions = splitted_segments

    # remove duplicated segments
    seg_definitions = np.unique(np.asarray(seg_definitions, dtype=object)).tolist()
    if isinstance(seg_definitions[0], list) is False:
        seg_definitions = [seg_definitions]

    # for each block, find the segment it belongs to
    segment_id = np.zeros(loop.shape[0], dtype=np.float32)
    for n in range(len(seg_definitions)):
        idx = _autoseg.find_segments(parent_block_id, seg_definitions[n])
        segment_id[idx] = n

    # assign segment label to loop
    loop[:, 0] = segment_id

    # now build segment fields
    n_segments = len(seg_definitions)
    segments = []
    for n in range(n_segments):
        segments.append(Segment(n, seg_definitions[n]))

    return segments


def _find_b1_max(parent_blocks):
    return np.max(
        [block.rf.wav.amplitude for block in parent_blocks if block.rf is not None]
    )


# TODO: implement this
def _find_grad_max(parent_blocks):
    return 0.0


# TODO: implement this
def _find_slew_max(parent_blocks):
    return 0.0


def _calc_duration(segment_id, block_duration):
    block_duration = block_duration.sum()

    # total segment ringdown
    n_seg_boundaries = (np.diff(segment_id + 1) != 0).sum()
    seg_ringdown_duration = SEGMENT_RINGDOWN_TIME * n_seg_boundaries

    return block_duration + seg_ringdown_duration
