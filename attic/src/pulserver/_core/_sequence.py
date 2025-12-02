"""Intermediate sequence representation."""

__all__ = ["Sequence"]

import warnings
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from .._safety import compute_max_energy

from ._ceq import Ceq, PulseqBlock
from ._header import SequenceDefinition


class Sequence:
    """
    Intermediate representation of a Pulseq sequence.

    This class manages the creation and handling of Pulseq blocks, which consist of
    one or more PyPulseq events. A PulseqBlock allows for a maximum of one event for
    each board (rf, gx, gy, gz, adc, trig) to be executed at a time.

    Parameters
    ----------
    system : SimpleNamespace
        The hardware system parameters for the sequence (e.g., gradient, RF limits).
    platform : str
        The target platform for the sequence. Acceptable values are 'pulseq' (alias for 'siemens')
        and 'toppe' (alias for 'gehc').

    Attributes
    ----------
    _system : SimpleNamespace
        Holds the system configuration (e.g., gradient and RF settings).
    _format : str
        Indicates the platform type ('siemens' or 'gehc').
    _sequence : pp.Sequence or Ceq
        PyPulseq sequence or Ceq structure depending on the format.
    _block_library : dict
        Stores all defined Pulseq blocks by name.
    _header : SequenceDefinition
        Holds the sequence metadata, including parameters like FOV and shape.
    _section_labels : list
        List of section labels in the sequence.
    _sections_edges : list
        Sequence length markers for different sections.
    """

    def __init__(self, system: SimpleNamespace, platform: str):
        self._system = system

        if platform == "pulseq":
            platform = "siemens"
        elif platform == "toppe":
            platform = "gehc"

        self._format = platform

        if self._format == "siemens":
            self._sequence = pp.Sequence(system=system)
        elif self._format == "gehc":
            self._loop = []
        else:
            raise ValueError(
                f"Accepted platforms are 'siemens'/'pulseq' and 'gehc'/'toppe', found '{platform}'."
            )

        if self._format == "siemens":
            self._block_library = {"delay": {}}
        elif self._format == "gehc":
            self._block_library = {"delay": PulseqBlock(ID=0)}

        self._section_labels = []
        self._sections_edges = []
        self._header = None

    def initialize_header(self, ndims: int):
        """
        Initialize the header with sequence metadata.

        Parameters
        ----------
        ndims : int
            The number of dimensions for the acquisition (e.g., 2D or 3D).
        """
        self._header = SequenceDefinition(ndims)

    def register_block(
        self,
        name: str,
        rf=None,
        gx=None,
        gy=None,
        gz=None,
        adc=None,
        trig=None,
        delay=None,
    ):
        """
        Register a Pulseq block with one or more events.

        Parameters
        ----------
        name : str
            The block name to be registered.
        rf, gx, gy, gz, adc, trig, delay : SimpleNamespace or None
            Individual components of the Pulseq block. These may include RF, gradients
            (gx, gy, gz), ADC event, trigger, or delay. If `None`, the event is ignored.

        Raises
        ------
        AssertionError
            If the sequence format is already defined or if the block contains both
            RF and ADC events.
        """
        # Sanity checks
        if self._format == "siemens":
            assert (
                len(self._sequence.block_events) == 0
            ), "Define all events before building the loop."
        elif self._format == "gehc":
            assert len(self._loop) == 0, "Define all events before building the loop."

        if rf is not None and adc is not None:
            VALID_BLOCK = False
        else:
            VALID_BLOCK = True
        assert VALID_BLOCK, "A block cannot contain both RF and ADC events."

        if gx is not None:
            assert (
                gx.channel == "x"
            ), f"x-gradient waveform is directed towards {gx.channel}"
        if gy is not None:
            assert (
                gy.channel == "y"
            ), f"y-gradient waveform is directed towards {gy.channel}"
        if gz is not None:
            assert (
                gz.channel == "z"
            ), f"z-gradient waveform is directed towards {gz.channel}"

        # Update block library
        if self._format == "siemens":
            self._block_library[name] = {}
            for event, label in zip(
                [rf, gx, gy, gz, adc, trig, delay],
                ["rf", "gx", "gy", "gz", "adc", "trig", "delay"],
            ):
                if event is not None:
                    self._block_library[name][label] = deepcopy(event)
        elif self._format == "gehc":
            ID = len(self._block_library)
            self._block_library[name] = PulseqBlock(
                ID, rf, gx, gy, gz, adc, trig, delay
            )

    def section(self, name: str):
        """
        Define a new section within the sequence.

        Parameters
        ----------
        name : str
            A unique name for the section.

        Raises
        ------
        AssertionError
            If the section name already exists.
        """
        assert name not in self._section_labels, f"Section '{name}' already exists."

        if self._format == "siemens":
            _current_seqlength = len(self._sequence.block_events)
        elif self._format == "gehc":
            _current_seqlength = len(self._loop)

        self._sections_edges.append(_current_seqlength)

        # Update header section
        if self._header is not None:
            self._header.section(name)

    def add_block(
        self,
        name: str,
        gx_amp=1.0,
        gy_amp=1.0,
        gz_amp=1.0,
        rf_amp=1.0,
        rf_phase=0.0,
        rf_freq=0.0,
        adc_phase=0.0,
        delay=None,
        rotmat=None,
    ):
        """
        Add a previously registered block to the sequence.

        Parameters
        ----------
        name : str
            The name of the block to be added.
        gx_amp, gy_amp, gz_amp : float, optional
            Scaling factors for the x, y, and z gradients, respectively.
        rf_amp : float, optional
            Scaling factor for the RF pulse amplitude.
        rf_phase, rf_freq : float, optional
            Phase and frequency modulation for the RF pulse.
        adc_phase : float, optional
            Phase modulation for the ADC event.
        delay : float, optional
            Delay for pure delay blocks.
        rotmat : np.ndarray, optional
            3x3 rotation matrix to apply to gradients.

        Raises
        ------
        AssertionError
            If the block is not registered or gradients have inconsistent lengths.
        ValueError
            If delay is missing for a pure delay block.
        """
        assert name in self._block_library, f"Requested block '{name}' not found!"

        if self._format == "siemens":
            if name == "delay":
                if delay is None:
                    raise ValueError("Missing 'delay' input for pure delay block.")
                self._sequence.add_block(pp.make_delay(delay))
            else:
                current_block = deepcopy(self._block_library[name])
                if delay is not None:
                    warnings.warn("Dynamic delay is ignored for non-delay blocks.")

                # Apply RF modifications
                if "rf" in current_block:
                    current_block["rf"].signal *= rf_amp
                    current_block["rf"].phase_offset = rf_phase
                    current_block["rf"].freq_offset += rf_freq

                # Apply ADC phase
                if "adc" in current_block:
                    current_block["adc"].phase_offset = adc_phase

                # Scale gradients
                for ch, amp in zip(["gx", "gy", "gz"], [gx_amp, gy_amp, gz_amp]):
                    if ch in current_block:
                        current_block[ch] = pp.scale_grad(
                            grad=current_block[ch], scale=amp
                        )

                # Rotate gradients
                if rotmat is not None:
                    # extract gradient waveforms from current event
                    current_grad = {}
                    for ch in ["gx", "gy", "gz"]:
                        if ch in current_block:
                            current_grad[ch] = current_block[ch]

                    # actual rotation
                    current_block = _pp_rotate(current_block, rotmat)

                    # replace rotated gradients in current event
                    for ch in ["gx", "gy", "gz"]:
                        if ch in current_block:
                            current_block[ch] = current_grad[ch]

                self._sequence.add_block(*current_block.values())

        elif self._format == "gehc":
            parent_block_id = self._block_library[name].ID
            if name == "delay":
                if delay is None:
                    raise ValueError("Missing 'delay' input for pure delay block.")
                block_duration = delay
                rotmat = np.eye(3, dtype=np.float32).ravel().tolist()
                hasrot = [1]
                hasadc = [0]
                loop_row = (
                    [
                        -1,
                        parent_block_id,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        block_duration,
                    ]
                    + rotmat
                    + hasadc
                    + hasrot
                )

            else:
                if delay is not None:
                    warnings.warn(
                        "Dynamic delay not allowed except for pure delay blocks - ignoring the specified delay"
                    )
                block_duration = self._block_library[name].duration
                if rotmat is None:
                    rotmat = np.eye(3, dtype=np.float32).ravel().tolist()
                    hasrot = [1]
                else:
                    rotmat = rotmat.ravel().tolist()
                    hasrot = [-1]
                if self._block_library[name].adc is None:
                    hasadc = [0]
                else:
                    hasadc = [1]
                loop_row = (
                    [
                        -1,
                        parent_block_id,
                        rf_amp,
                        rf_phase,
                        rf_freq,
                        gx_amp,
                        gy_amp,
                        gz_amp,
                        adc_phase,
                        block_duration,
                    ]
                    + rotmat
                    + hasadc
                    + hasrot
                )
            self._loop.append(loop_row)

    def set_definition(self, key: str, *args, **kwargs):
        """
        Set a specific sequence parameter in the header.

        Parameters
        ----------
        key : str
            Parameter to be set, e.g., 'fov', 'shape', 'limits', or 'trajectory'.
        *args :
            Positional arguments for the parameter.
        **kwargs :
            Keyword arguments for the parameter.

        Raises
        ------
        KeyError
            If 'fov' is set before 'shape' in 2D acquisitions.
        """
        self._header.set_definition(key, *args, **kwargs)

    def set_label(
        self,
        iy: int = None,
        iz: int = 0,
        islice: int = None,
        icontrast: int = 0,
        iframe: int = 0,
        ishot: int = None,
    ):
        """
        Set the label for the current MRI acquisition by configuring encoding indices and trajectory data.

        Parameters
        ----------
        iy : int, optional
            Cartesian encoding step in the k-space (along the y-axis). Mutually exclusive with `ishot`.
            Default is None.
        iz : int, optional
            Cartesian encoding step in the k-space (along the z-axis, for 3D acquisitions). Mutually
            exclusive with `islice`. Default is 0.
        islice : int, optional
            Slice index for 2D multislice acquisitions. Mutually exclusive with `iz`. Default is None.
        icontrast : int, optional
            Contrast encoding index for multicontrast MRI acquisitions. Mutually exclusive with `iframe`.
            Default is 0.
        iframe : int, optional
            Repetition or dynamic frame index for dynamic MRI acquisitions. Mutually exclusive with
            `icontrast`. Default is 0.
        ishot : int, optional
            Non-Cartesian encoding step (shot number). Mutually exclusive with `iy`. Default is None.

        Raises
        ------
        ValueError
            If both `iy` and `ishot` are provided (Cartesian and Non-Cartesian are mutually exclusive).
            If both `islice` and `iz` are provided (2D and 3D encoding are mutually exclusive).
            If both `iframe` and `icontrast` are provided (dynamic and multicontrast MRI are mutually
            exclusive).

        Notes
        -----
        This function updates the MRI acquisition label by setting k-space encoding indices and, if
        applicable, the corresponding trajectory from the `self._trajectory` attribute. The indices for
        k-space encoding (`kspace_encode_step_1` for the y-axis, `kspace_encode_step_2` for the z-axis or
        slice) and contrast or repetition are updated based on the provided parameters.

        The acquisition trajectory is updated if `ishot` is provided, and it fetches the corresponding
        trajectory point from the `self._trajectory` data. The function also appends the newly created
        acquisition to `self._labels`.

        Examples
        --------
        Set a Cartesian acquisition with y and z encoding steps:

        >>> set_label(iy=10, iz=5)

        Set a 2D multislice acquisition with slice index:

        >>> set_label(islice=3)

        Set a Non-Cartesian shot-based acquisition:

        >>> set_label(ishot=7)
        """
        self._header.set_label(iy, iz, islice, icontrast, iframe, ishot)

    def build(self, ngain=0):
        """
        Build the final sequence.

        Returns
        -------
        sequence : pp.Sequence or Ceq
            The finalized sequence object.
        header : dict or None
            Sequence metadata, if available.
        """
        if self._format != "siemens":
            self._sequence = Ceq(
                list(self._block_library.values()),
                self._loop,
                self._sections_edges,
            )
            P = compute_max_energy(deepcopy(self._sequence), self._system)
            self._sequence.max_rf_power = P
            self._sequence.n_gain = ngain

        if self._header is not None:
            return self._sequence, self._header

        return self._sequence


def _pp_rotate(grad, rot_matrix):
    """
    Apply a rotation matrix to gradient waveforms.

    Parameters
    ----------
    grad : dict
        Dictionary containing the gradient events (gx, gy, gz).
    rot_matrix : np.ndarray
        3x3 rotation matrix.

    Returns
    -------
    grad : dict
        Updated gradient dictionary with rotated waveforms.
    """
    grad_channels = ["gx", "gy", "gz"]
    grad = deepcopy(grad)

    wave_length = [len(grad[ch]) for ch in grad_channels if ch in grad]
    assert (
        len(set(wave_length)) == 1
    ), "All gradient waveforms must have the same length."

    grad_mat = np.stack(
        [grad.get(ch, np.zeros(wave_length[0])).squeeze() for ch in grad_channels],
        axis=0,
    )
    grad_mat = rot_matrix @ grad_mat

    for j, ch in enumerate(grad_channels):
        grad[ch] = grad_mat[j]

    for ch in grad_channels:
        if np.allclose(grad[ch], 0.0):
            grad.pop(ch)

    return grad
