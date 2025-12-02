"""Sequence Parameters object."""

__all__ = ["BaseParams", "ParamsParser"]

from dataclasses import dataclass
from dataclasses import asdict as _asdict

import struct
import re

from .._opts import get_opts


class BaseParams:
    """
    Base parameter parser class.

    This is designed to convert data parsed by ParamsParser
    to a format suitable for actual design routines.

    """

    def __init__(
        self,
        dwell: float | None = 4e-6,
        raster: float | None = 2e-6,
        gmax: float | None = None,
        smax: float | None = None,
        b0_field: float | None = None,
        psd_rf_wait: float | None = 0.0,
        psd_grd_wait: float | None = 0.0,
        rf_dead_time: float | None = 100e-6,
        rf_ringdown_time: float | None = 60e-6,
        adc_dead_time: float | None = 40e-6,
    ):
        # Build opts
        if gmax is None:
            raise ValueError("Please provide gmax")
        if smax is None:
            raise ValueError("Please provide smax")
        if b0_field is None:
            raise ValueError("Please provide b0_field")

        _opts_dict = {
            "B0": b0_field,
            "max_grad": gmax,
            "max_slew": smax,
            "rf_raster_time": raster,
            "grad_raster_time": raster,
            "adc_raster_time": dwell,
            "rf_dead_time": rf_dead_time,
            "rf_ringdown_time": max(rf_ringdown_time, psd_rf_wait),
            "adc_dead_time": adc_dead_time,
        }

        self.opts_dict = get_opts(_opts_dict)

    def asdict(self):  # noqa
        return vars(self)


@dataclass
class ParamsParser:
    """
    Python representation of the C SequenceParams struct.

    Attributes
    ----------
    FOVx : Optional[float]
        Field of view in mm (x-direction).
    FOVy : Optional[float]
        Field of view in mm (y-direction).
    Nx : Optional[int]
        Matrix size (x-direction).
    Ny : Optional[int]
        Matrix size (y-direction).
    Nslices : Optional[int]
        Number of slices.
    Nechoes : Optional[int]
        Number of echoes.
    Nphases : Optional[int]
        Number of phases.
    Ndummies : Optional[int]
        Number of dummy scans to reach steady-state.
    Ngain : Optional[int]
        Number of transmit gain calibration scans.
    slice_thickness : Optional[float]
        Thickness of each slice (mm).
    slice_spacing : Optional[float]
        Spacing between slices (mm).
    Rplane : Optional[float]
        In-plane undersampling factor.
    R : Optional[float]
        Additional in-plane undersampling factor.
    Rslice : Optional[float]
        Through-plane undersampling factor.
    Rshift : Optional[int]
        CAIPIRINHA shift.
    PFfactor : Optional[float]
        Partial Fourier acceleration factor.
    ETL : Optional[int]
        Number of k-space shots per readout.
    Nshots : Optional[int]
        Number of k-space segments for EPI.
    Cplane : Optional[float]
        Number of phase encoding calibration lones.
    Cslice : Optional[float]
        Number of slice encoding calibration lones.
    TE : Optional[float]
        Echo time (ms).
    TE0 : Optional[float]
        First echo time (ms) for multiecho.
    TR : Optional[float]
        Repetition time (ms).
    Tprep : Optional[float]
        Preparation time (ms).
    Trecovery : Optional[float]
        Recovery time (ms).
    flip : Optional[float]
        Flip angle in degrees.
    flip2 : Optional[float]
        Second flip angle in degrees.
    refoc_flip : Optional[float]
        Refocusing flip angle in degrees.
    freq_dir : Optional[int]
        Frequency direction (0: A/P; 1: S/L).
    freq_verse : Optional[int]
        Frequency verse (1: normal; -1: swapped).
    phase_verse : Optional[int]
        Phase verse (1: normal; -1: swapped).
    bipolar_echoes : Optional[int]
        Bipolar echoes (0: false, 1: true).
    dwell : Optional[float]
        ADC dwell time (s).
    raster : Optional[float]
        Waveform raster time (s).
    gmax : Optional[float]
        Maximum gradient strength (mT/m).
    smax : Optional[float]
        Maximum gradient slew rate (T/m/s).
    b1_max : Optional[float]
        Maximum RF value (uT).
    b0_field : Optional[float]
        System field strength (T).
    """

    function_name: str
    FOVx: float | None = None
    FOVy: float | None = None
    Nx: int | None = None
    Ny: int | None = None
    Nslices: int | None = None
    Nechoes: int | None = None
    Nphases: int | None = None
    Ndummies: int | None = None
    Ngain: int | None = None
    slice_thickness: float | None = None
    slice_spacing: float | None = None
    Rplane: float | None = None
    R: float | None = None
    Rslice: float | None = None
    PFfactor: float | None = None
    Rshift: int | None = None
    ETL: int | None = None
    Nshots: int | None = None
    Cplane: float | None = None
    Cslice: float | None = None
    TE: float | None = None
    TE0: float | None = None
    TR: float | None = None
    Tprep: float | None = None
    Trecovery: float | None = None
    flip: float | None = None
    flip2: float | None = None
    refoc_flip: float | None = None
    freq_dir: int | None = None
    freq_verse: int | None = None
    phase_verse: int | None = None
    bipolar_echoes: int | None = None
    dwell: float | None = None
    raster: float | None = None
    gmax: float | None = None
    smax: float | None = None
    b0_field: float | None = None
    psd_rf_wait: float | None = None
    psd_grd_wait: float | None = None
    rf_dead_time: float | None = None
    rf_ringdown_time: float | None = None
    adc_dead_time: float | None = None

    def __post_init__(self):  # noqa
        # rounding
        if self.psd_rf_wait is not None:
            self.psd_rf_wait *= 1e-6
            self.psd_rf_wait = round(self.psd_rf_wait * 1e6) / 1e6
        if self.psd_grd_wait is not None:
            self.psd_grd_wait *= 1e-6
            self.psd_grd_wait = round(self.psd_grd_wait * 1e6) / 1e6
        if self.raster is not None:
            self.raster = round(self.raster * 1e6) / 1e6
        if self.dwell is not None:
            self.dwell = round(self.dwell * 1e6) / 1e6
        if self.rf_dead_time is not None:
            self.rf_dead_time = round(self.rf_dead_time * 1e6) / 1e6
        if self.rf_ringdown_time is not None:
            self.rf_ringdown_time = round(self.rf_ringdown_time * 1e6) / 1e6
        if self.adc_dead_time is not None:
            self.adc_dead_time = round(self.adc_dead_time * 1e6) / 1e6

    @classmethod
    def from_file(cls, filename: str) -> "ParamsParser":  # noqa
        with open(filename, "rb") as file:
            return ParamsParser.from_bytes(file.read())

    @classmethod
    def from_bytes(cls, data: bytes) -> "ParamsParser":
        """Deserialize from a byte array into a SequenceParams object."""
        format_string = "2f 7h 6f 5h 8f 4h 10f"

        # Unpack the function name
        function_name = struct.unpack("256s", data[:256])[0]
        function_name = function_name.decode("utf-8").rstrip("\x00")

        # Unpack values
        values = struct.unpack(format_string, data[256:])
        values = [None if x == -1 or x == -1.0 else x for x in values]

        return ParamsParser(function_name, *values)

    def to_bytes(self) -> bytes:  # noqa
        """
        Serialize this dataclass to a byte array.
        """
        format_string = "2f 7h 6f 5h 8f 4h 10f"

        # Pack function name
        function_name = struct.pack("256s", self.function_name.encode("utf-8"))

        # Pack values
        values = list(self.asdict(filt=False).values())
        values = [-1 if x is None else x for x in values]
        values = _convert_values_to_struct(values, format_string)
        values = struct.pack(format_string, *values)

        return function_name + values

    def asdict(self, filt=True) -> dict:
        """
        Return a dictionary of the dataclass, excluding None values.

        Returns
        -------
        dict
            A dictionary of the dataclass fields, excluding None values.
        """
        if filt:
            out = {k: v for k, v in _asdict(self).items() if v is not None}
        else:
            out = _asdict(self)

        out.pop("function_name")
        return out


# %% local subroutines
def _parse_format_string(format_string):
    # Define a mapping from struct format characters to Python types
    type_map = {
        "f": float,  # 4-byte float
        "h": int,  # 2-byte short, int is used in Python for struct compatibility
    }

    # Parse format string to get count and type
    pattern = r"(\d*)([fh])"  # Regex to match numbers followed by 'f' or 'h'
    parsed_format = []

    for match in re.finditer(pattern, format_string):
        count = (
            int(match.group(1)) if match.group(1) else 1
        )  # Default to 1 if no number
        type_char = match.group(2)

        if type_char in type_map:
            parsed_format.extend([type_map[type_char]] * count)

    return parsed_format


def _convert_values_to_struct(values, format_string):
    parsed_format = _parse_format_string(format_string)

    if len(values) != len(parsed_format):
        raise ValueError("Number of values does not match format string.")

    # Convert values based on parsed format types
    converted_values = [parsed_format[i](values[i]) for i in range(len(values))]
    return converted_values
