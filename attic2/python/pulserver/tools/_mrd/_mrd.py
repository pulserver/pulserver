"""MRD builder utilities."""

__all__ = ['DUMMY_SYSTEM', 'ISMRMRDBuilder']

import copy
import pathlib
import warnings
from collections.abc import Sequence
from enum import Enum
from types import SimpleNamespace

import ismrmrd as mrd
import ismrmrd.xsd as xsd
import numpy as np
import pypulseq as pp

from ._mrd_file import write_mrd


def __dummy_system__():
    sys = pp.Opts()
    sys.max_grad = np.inf
    sys.max_slew = np.inf
    return sys


DUMMY_SYSTEM = __dummy_system__()
OTHER = xsd.trajectoryType.OTHER


class ISMRMRDBuilder:
    """
    A builder class for ISMRMD Dataset.

    This is used to build a sidecar (ISMR)MRD Dataset containing the Sequence
    details. It can be then merged with dataset from scanner to obtain
    an actual ISMRMRD Dataset to be used for reconstruction.

    Attributes
    ----------
    passthrough : bool
        If ``True``, all the methods are converted into passthrough dummy functions.
        Useful for a quick dry run of sequence creation routine.
    head : ismrmrd.xsd.ismrmrdHeader
        The ISMRMRD XML Header describing the Dataset.
    acquisitions : list[ismrmrd.Acquisition]
        List of ismrmrd Acquisitions included in the Dataset.
        These only contains info known at design time - other data must be
        merged from the actual scan (i.e., data field, actual orientation,
        timestamps, etc).
    acquisitions : list[ismrmrd.Waveform]
        List of ismrmrd Waveforms included in the Dataset.
        hese only contains info known at design time - other data must be
        merged from the actual scan.
    freeWaveformID : int
        First free Waveform ID for custom Waveform types. It is initialized
        to ``1024``, as per MRD specs.
    label_dict : dict
        Dictionary used to keep track of labels evolution. It is used to enable
        ``'SET'`` and ``'INC'`` behaviours as in Pulseq, whereas ISMRMRD
        natively supports ``'SET'`` only.

    """

    def __init__(self, passthrough: bool = False):
        self.passthrough = passthrough

        # MRD attributes
        self.head = xsd.ismrmrdHeader()
        self.head.measurementInformation = xsd.measurementInformationType()
        self.head.experimentalConditions = xsd.experimentalConditionsType()
        self.head.encoding = []
        self.head.sequenceParameters = xsd.sequenceParametersType()
        self.head.userParameters = xsd.userParametersType()
        self.head.waveformInformations = []

        # Arrays
        self.acquisitions = []  # Raw acquisitions
        self.waveforms = []  # Waveforms

        self.freeWaveformID = 1024

        self.label_dict = {
            'kspace_encode_step_1': None,
            'kspace_encode_step_2': None,
            'average': None,
            'slice': None,
            'contrast': None,
            'phase': None,
            'repetition': None,
            'segment': None,
        }

        self.add_encoding()

    def mode_switch(func):
        def wrapper(self, *args, **kwargs):
            if self.passthrough:
                return None
            return func(self, *args, **kwargs)

        return wrapper

    def write(
        self,
        filepath: str | pathlib.Path,
        dataset_name: str = 'dataset',
        overwrite: bool = True,
    ) -> None:
        """
        Write MRD file to disk.

        Parameters
        ----------
        filepath : str
            Path to disk position where to store the file.
        dataset_name : str, optional
            Path within HDF5 file where to store the dataset.
            The default is "dataset".
        overwrite : bool, optional
            If True, overwrite the file if it is exist. The default is False.

        """
        write_mrd(
            filepath,
            dataset_name,
            self.head,
            self.acquisitions,
            self.waveforms,
            overwrite,
        )

    @mode_switch
    def calc_trajectory(self, *events: tuple[SimpleNamespace]):
        """
        Calculate readout header based on input PyPulseq blocks.

        Parameters
        ----------
        *events : tuple[SimpleNamespace]
            List of tuple of events required to get readout.

        Examples
        --------
                * line readout (cartesian): prewind + readout -> calc_trajectory((gx_pre,), (gx, adc))
                * spiral readout: readout only -> calc_trajectory((spiral_x_grad, spiral_y_grad, adc))

        Notes
        -----
        We assume that ADC is designed to cover only the actual readout part
        of the gradient, i.e., discard_pre=discard_post=0.

        """
        # Remove ID
        events = [[copy.deepcopy(ev) for ev in event] for event in events]
        for event in events:
            for ev in event:
                if hasattr(ev, 'id'):
                    delattr(ev, 'id')

        # build sequence based on provided blocks
        seq = pp.Sequence(system=DUMMY_SYSTEM)
        for event in events:
            seq.add_block(*event)

        # calculate k space
        k_traj_adc, _, _, _, t_adc = seq.calculate_kspace()

        # calculate sample time
        sample_time_us = np.unique(np.round(np.diff(t_adc), 12)).item() * 1e6

        # calculate number of samples
        number_of_samples = k_traj_adc.shape[-1]

        # calculate center sample
        k_space_zeros = np.all(k_traj_adc == 0, axis=0)  # True where column = (0,0,0)

        if np.any(k_space_zeros):
            center_sample = np.argmax(k_space_zeros)  # first True
        else:
            distances = np.linalg.norm(k_traj_adc, axis=0)  # Euclidean norm per column
            idx_last = np.argmax(-np.round(distances, 6)[::-1])
            center_sample = len(distances) - 1 - idx_last

        # get non uniform trajectory
        traj = []
        trajectory_dimensions = 3

        dkz = np.unique(np.round(np.diff(k_traj_adc[2]), 2))
        if dkz.size == 1:
            trajectory_dimensions -= 1
        else:
            traj.append(k_traj_adc[2])

        dky = np.unique(np.round(np.diff(k_traj_adc[1]), 2))
        if dky.size == 1:
            trajectory_dimensions -= 1
        else:
            traj.append(k_traj_adc[1])

        dkx = np.unique(np.round(np.diff(k_traj_adc[0]), 2))
        if dkx.size == 1:
            trajectory_dimensions -= 1
        else:
            traj.append(k_traj_adc[0])

        traj = traj[::-1]
        if traj:
            traj = np.stack(traj, axis=1)

        return SimpleNamespace(
            sample_time_us=int(sample_time_us),
            number_of_samples=number_of_samples,
            center_sample=center_sample,
            traj=traj,
            trajectory_dimensions=trajectory_dimensions,
        )

    @mode_switch
    def add_encoding(self):
        """Append a new ``Encoding`` space into ``head``."""
        encoding = xsd.encodingType()
        encoding.encodedSpace = xsd.encodingSpaceType()
        encoding.encodedSpace.matrixSize = xsd.matrixSizeType()
        encoding.encodedSpace.fieldOfView_mm = xsd.fieldOfViewMm()
        encoding.reconSpace = xsd.encodingSpaceType()
        encoding.reconSpace.matrixSize = xsd.matrixSizeType()
        encoding.reconSpace.fieldOfView_mm = xsd.fieldOfViewMm()
        encoding.encodingLimits = xsd.encodingLimitsType()
        encoding.echoTrainLength = 1
        encoding.trajectory = xsd.trajectoryType.CARTESIAN  # default
        encoding.trajectoryDescription = xsd.trajectoryDescriptionType()
        encoding.parallelImaging = xsd.parallelImagingType()

        self.head.encoding.append(encoding)
        self.current_encoding = len(self.head.encoding) - 1

    @mode_switch
    def set_encoding(self, idx: int):
        """
        Set currently active ``Encoding`` space.

        Parameters
        ----------
        idx : int
            Index of desired encoding space in ``head.encoding``.

        Raises
        ------
        IndexError
            If selected index is out of range.

        """
        if idx < 0:
            raise IndexError('Encoding index out of range')
        if idx >= self.current_encoding:
            while idx < len(self.head.encodings):
                self.new_encoding()
        self.current_encoding = idx

    def set_name(self, value):
        self.head.measurementInformation.sequenceName = value

    @mode_switch
    def set_H1resonanceFrequency_Hz(self, gamma_bar: float, B0: float):
        """
        Set resonant frequency based on nucleus and static field strength.

        Parameters
        ----------
        gamma_bar : float
            Active nucleus gyromagnetic factor in ``Hz/T``.
        B0 : float
            Static field strength in ``Hz``.

        """
        self.head.experimentalConditions.H1resonanceFrequency_Hz = int(gamma_bar * B0)

    @mode_switch
    def set_fov(self, dx: float, dy: float, dz: float):
        """
        Set current encoding Field Of View.

        Parameters
        ----------
        dx : float
            Field of View along logical ``x``, in units of ``[mm]``.
        dy : float
            Field of View along logical ``y``, in units of ``[mm]``.
        dz : float
            Field of View along logical ``z``, in units of ``[mm]``.

        Notes
        -----
        For 3D imaging, dz is usually given by ``voxel_size_z * num_voxels_z``.
        For 2D, it is usually ``slice_spacing * num_slices``, with ``slice_spacing``
        being equal to slice thickness for contiguous non-overlapping slices.

        """
        self.head.encoding[self.current_encoding].encodedSpace.fieldOfView_mm.x = dx
        self.head.encoding[self.current_encoding].encodedSpace.fieldOfView_mm.y = dy
        self.head.encoding[self.current_encoding].encodedSpace.fieldOfView_mm.z = dz
        self.head.encoding[self.current_encoding].reconSpace.fieldOfView_mm.x = dx
        self.head.encoding[self.current_encoding].reconSpace.fieldOfView_mm.y = dy
        self.head.encoding[self.current_encoding].reconSpace.fieldOfView_mm.z = dz

    @mode_switch
    def set_matrix(self, nx: int, ny: int, nz: int):
        """
        Set current encoding matrix size.

        Parameters
        ----------
        nx : int
            Number of voxels along logical ``x``.
        ny : int
            Number of voxels along logical ``y``.
        nz : int
            Number of voxels along logical ``z``.

        """
        self.head.encoding[self.current_encoding].encodedSpace.matrixSize.x = nx
        self.head.encoding[self.current_encoding].encodedSpace.matrixSize.y = ny
        self.head.encoding[self.current_encoding].encodedSpace.matrixSize.z = nz
        self.head.encoding[self.current_encoding].reconSpace.matrixSize.x = nx
        self.head.encoding[self.current_encoding].reconSpace.matrixSize.y = ny
        self.head.encoding[self.current_encoding].reconSpace.matrixSize.z = nz

    @mode_switch
    def set_limits(
        self, axis: str, maximum: int, minimum: int = 0, center: int | None = None
    ):
        """
        Set current encoding limits along the specified axis.

        Parameters
        ----------
        axis : str
            Target encoding axis.
        maximum : int
            Maximum encoding index along target size.
        minimum : int, optional
            Minimum encoding index along target size. The default is ``0``.
        center : int | None, optional
            Encoding index corresponding to k-space center along target axis.
            The default is ``None`` (center of specified range).

        """
        key = axis2key(axis)[0]
        limit = getattr(self.head.encoding[self.current_encoding].encodingLimits, key)
        if limit is None:
            limit = xsd.limitType()
        limit.minimum = minimum
        limit.maximum = maximum
        if center is None:
            center = int(np.ceil(maximum / 2).item())
        limit.center = center
        setattr(self.head.encoding[self.current_encoding].encodingLimits, key, limit)

    @mode_switch
    def set_etl(self, value: int):
        """
        Set current encoding Echo Train Length.

        Parameters
        ----------
        value : int
            Echo Train Length factor.

        """
        self.head.encoding[self.current_encoding].echoTrainLength = value

    @mode_switch
    def set_trajectory(
        self,
        traj_type: xsd.trajectoryType = OTHER,
        desc: dict | None = None,
        comment: str = '',
    ):
        """
        Set trajectory metadata for current encoding.

        Parameters
        ----------
        traj_type : xsd.trajectoryType, optional
            Trajectory type. The default is ``OTHER``.
        desc : dict | None, optional
            Trajectory description. The default is ``None``.
        comment : str, optional
            Addictional comment describing trajectory.
            The default is ``''``.

        """
        # Set trajectory type
        if not isinstance(traj_type, Enum):
            raise ValueError('traj_type must be an Enum')
        if traj_type not in xsd.trajectoryType:
            raise ValueError('traj_type must be a valid trajectoryType')
        self.head.encoding[self.current_encoding].trajectory = traj_type

        # Set trajectory description
        if desc is None:
            desc = {}

        # fill appropriate fields
        udouble = []
        ulong = []
        ustring = []
        for k, v in desc.items():
            if isinstance(v, float):
                el = xsd.userParameterDoubleType(name=k, value=v)
                udouble.append(el)
            if isinstance(v, int):
                el = xsd.userParameterLongType(name=k, value=v)
                ulong.append(el)
            if isinstance(v, str):
                el = xsd.userParameterStringType(name=k, value=v)
                ustring.append(el)
        self.head.encoding[
            self.current_encoding
        ].trajectoryDescription.userParameterDouble = udouble
        self.head.encoding[
            self.current_encoding
        ].trajectoryDescription.userParameterLong = ulong
        self.head.encoding[
            self.current_encoding
        ].trajectoryDescription.userParameterString = ustring

        # Set trajectory comment
        self.head.encoding[self.current_encoding].comment = comment

    @mode_switch
    def set_parallel_imaging_info(
        self,
        calibration_type: xsd.calibrationModeType,
        Ry: int = 1,
        Rz: int = 1,
        interleaving_dim: xsd.interleavingDimensionType | None = None,
    ):
        """
        Set parallel imaging header info.

        Parameters
        ----------
        calibration_type : xsd.calibrationModeType
            Type of Parallel Imaging calibration (e.g., EXTERNAL, EMBEDDED).
        Ry : int, optional
            Acceleration along ``y``(phase encoding dir).
            The default is ``1`` (no acceleration).
        Rz : int, optional
            Acceleration along ``z``(partition encoding dir).
            The default is ``1`` (no acceleration).

        """
        if not isinstance(calibration_type, Enum):
            raise ValueError('calibration_type must be an Enum')
        if calibration_type not in xsd.calibrationModeType:
            raise ValueError('calibration_type must be a valid calibrationModeType')
        parallelImaging = xsd.parallelImagingType()
        parallelImaging.accelerationFactor = xsd.accelerationFactorType()
        parallelImaging.accelerationFactor.kspace_encoding_step_1 = Ry
        parallelImaging.accelerationFactor.kspace_encoding_step_2 = Rz
        parallelImaging.calibrationMode = calibration_type
        parallelImaging.interleavingDimension = interleaving_dim
        self.head.encoding[self.current_encoding].parallelImaging = parallelImaging

    @mode_switch
    def set_multiband_info(
        self,
        calibration_type: xsd.multibandCalibrationType,
        mb_factor: int,
        spacing: float | list[float],
        calibration_encoding: int,
        kz_shift: float = 0.0,
    ):
        """
        Set multiband header info.

        Parameters
        ----------
        calibration_type : xsd.multibandCalibrationType
            Type of Multiband calibration (e.g., FULL_3D).
        mb_factor : int
            Multiband acceleration factor.
        spacing : float | list[float]
            Spacing between slices in a single multiband packet.
        calibration_encoding : int
            Index of reference encoding space containing calibration data.
        kz_shift : float, optional
            Caipirinha shift between slizes.
            The default is ``0.0`` (No CAIPI shift).

        """
        if not isinstance(calibration_type, Enum):
            raise ValueError('calibration_type must be an Enum')
        if calibration_type not in xsd.multibandCalibrationType:
            raise ValueError(
                'calibration_type must be a valid multibandCalibrationType'
            )

        if isinstance(spacing, float):
            spacing = [spacing]
        multiband_spacing = [
            xsd.multibandSpacingType((np.arange(mb_factor) * dz).tolist())
            for dz in spacing
        ]

        multiband = xsd.multibandType()
        multiband.spacing = multiband_spacing
        multiband.deltaKz = kz_shift
        multiband.calibration = calibration_type
        multiband.calibration_encoding = calibration_encoding

        if self.head.encoding[self.current_encoding].parallelImaging is None:
            self.head.encoding[
                self.current_encoding
            ].parallelImaging = xsd.parallelImagingType()
        self.head.encoding[self.current_encoding].parallelImaging.multiband = multiband

    @mode_switch
    def set_TR(self, value: float | Sequence[float]):
        """
        Set sequence Repetition Time(s).

        Parameters
        ----------
        value : float | Sequence[float]
            Repetition Time in units of ``[ms]``.

        """
        if np.isscalar(value):
            self.head.sequenceParameters.TR.append(value)
        else:
            for val in value:
                self.head.sequenceParameters.TR.append(val)

    @mode_switch
    def set_TE(self, value: float | Sequence[float]):
        """
        Set sequence Echo Time(s).

        Parameters
        ----------
        value : float | Sequence[float]
            Echo Time in units of ``[ms]``.

        """
        if np.isscalar(value):
            self.head.sequenceParameters.TE.append(value)
        else:
            for val in value:
                self.head.sequenceParameters.TE.append(val)

    @mode_switch
    def set_TI(self, value: float | Sequence[float]):
        """
        Set sequence Inversion Time(s).

        Parameters
        ----------
        value : float | Sequence[float]
            Inversion Time in units of ``[ms]``.

        """
        if np.isscalar(value):
            self.head.sequenceParameters.TI.append(value)
        else:
            for val in value:
                self.head.sequenceParameters.TI.append(val)

    @mode_switch
    def set_flipAngle_deg(self, value: float | Sequence[float]):
        """
        Set sequence Flip Angle(s).

        Parameters
        ----------
        value : float | Sequence[float]
            Flip Angle in units of ``[deg]``.

        """
        if np.isscalar(value):
            self.head.sequenceParameters.flipAngle_deg.append(value)
        else:
            for val in value:
                self.head.sequenceParameters.flipAngle_deg.append(val)

    @mode_switch
    def set_sequence_type(self, value: str):
        """
        Set sequence type.

        Parameters
        ----------
        value : str
            Sequence type.

        """
        self.head.sequenceParameters.sequence_type = value

    @mode_switch
    def set_echo_spacing(self, value: float | Sequence[float]):
        """
        Set sequence Echo Spacing(s).

        Parameters
        ----------
        value : float | Sequence[float]
            Echo Spacing in units of ``[ms]``.

        """
        if np.isscalar(value):
            self.head.sequenceParameters.echo_spacing.append(value)
        else:
            for val in value:
                self.head.sequenceParameters.echo_spacing.append(val)

    @mode_switch
    def set_diffusion(
        self,
        channel: xsd.diffusionDimensionType,
        scheme: str,
        direction: np.ndarray,
        bvalue: float | np.ndarray,
    ):
        """
        Set diffusion parameters for current encoding.

        Parameters
        ----------
        channel : xsd.diffusionDimensionType
            Data axis representing diffusion direction (e.g., SEGMENT).
        scheme : str
            Diffusion encoding scheme (e.g., bipolar).
        direction : np.ndarray
            Array of shape ``(n, 3)`` representing diffusion directions.
        bvalue : float | np.ndarray
            Diffusion b-value or array of b-values of shape ``(n,)``.

        """
        if not isinstance(channel, Enum):
            raise ValueError('channel must be an Enum')
        if channel not in xsd.diffusionDimensionType:
            raise ValueError('channel must be a valid diffusionDimensionType')

        direction = np.atleast_2d(direction)
        bvalue = np.atleast_1d(bvalue)
        if bvalue.shape[0] != 1 and bvalue.shape[0] != direction.shape[0]:
            raise ValueError(
                'Number of bvalues must match number of diffusion directions'
            )
        if bvalue.shape[0] == 1:
            bvalue = np.repeat(bvalue, direction.shape[0])

        gradDir = [
            xsd.gradientDirectionType(rl=el[0], ap=el[1], fh=el[2]) for el in direction
        ]
        diffusionParams = [
            xsd.diffusionType(gradDir[n], bvalue[n]) for n in range(bvalue.shape[0])
        ]

        self.head.sequenceParameters.diffusionDimension = channel
        self.head.sequenceParameters.diffusionScheme = scheme
        self.head.sequenceParameters.diffusion = diffusionParams

    @mode_switch
    def add_user_param(self, name: str, value: float | int | str | object):
        """
        Add User Parameter as ``name-value`` pair.

        Parameters
        ----------
        name : str
            Input parameter name.
        value : float | int | str | object
            Input parameter value. Depending on the type,
            it will be appended to ``userParameterDouble``, ``userParameterLong``,
            ``userParameterString`` or serialized as a waveform.

        Returns
        -------
        None.

        """
        if isinstance(value, float):
            if haskey(name, self.head.userParameters.userParameterDouble):
                setparam(
                    name,
                    value,
                    self.head.userParameters.userParameterDouble,
                )
            else:
                el = xsd.userParameterDoubleType(name=name, value=value)
                self.head.userParameters.userParameterDouble.append(el)
                return
        if isinstance(value, int):
            if haskey(name, self.head.userParameters.userParameterLong):
                setparam(
                    name,
                    value,
                    self.head.userParameters.userParameterLong,
                )
            else:
                el = xsd.userParameterLongType(name=name, value=value)
                self.head.userParameters.userParameterLong.append(el)
                return
        if isinstance(value, str):
            if haskey(name, self.head.userParameters.userParameterString):
                setparam(
                    name,
                    value,
                    self.head.userParameters.userParameterString,
                )
            else:
                el = xsd.userParameterStringType(name=name, value=value)
                self.head.userParameters.userParameterString.append(el)
                return

        # if not scalar, default to waveforms
        self._add_to_waveform(name, value)

    def _add_to_waveform(self, name, value):
        value = np.atleast_2d(value)
        found = False
        for el in self.head.waveformInformations:
            if el.waveformName == name:
                found = True
                waveId = el.waveformType
        if not (found):
            waveId = self.freeWaveformID
            newId = xsd.waveformInformationType(waveformName=name, waveformType=waveId)
            self.head.waveformInformations.append(newId)
            self.freeWaveformID += 1

        numChannels = value.shape[0]
        numSamples = value.shape[1]
        newWaveformHeader = mrd.WaveformHeader(
            waveform_id=waveId, channels=numChannels, number_of_samples=numSamples
        )
        newWaveform = mrd.Waveform(head=newWaveformHeader, data=value)
        self.waveforms.append(newWaveform)

    @mode_switch
    def add_acquisition(
        self,
        trajectory: SimpleNamespace,
        events: SimpleNamespace,
        flags: int | tuple[int] | None = None,
        encoding_idx: int | None = None,
        user_int: list[int] | None = None,
        user_float: list[float] | None = None,
    ):
        """
        Add an acquisition to the Acquisition lists.

        Parameters
        ----------
        trajectory : SimpleNamespace
            Current readout trajectory computed as ISMRMDBuilder.calc_trajectory().
        events : SimpleNamespace
            Pulseq labels and rotations associated with current readout.
        flags : int | tuple[int | None, optional
            MRD flags associated with current readout. The default is None.
        encoding_idx : int | None, optional
            Encoding index corresponding to current readout. The default is None (current encoding).
        user_int : list[int] | None, optional
            Additional user-defined integer params associated with current readout. The default is None.
        user_float : list[float] | None, optional
            Additional user-defined floating point params associated with current readout. The default is None.

        """
        acq = mrd.Acquisition()

        # set flags
        if flags is not None:
            if np.isscalar(flags):
                flags = [flags]
            for flag in flags:
                acq.setFlag(flag)

        # set scan counter
        acq.version = 1
        acq.scan_counter = len(self.acquisitions)

        # resize trajectory
        if trajectory.trajectory_dimensions:
            acq.resize(
                trajectory_dimensions=trajectory.trajectory_dimensions,
                number_of_samples=trajectory.number_of_samples,
            )

        # set center sample
        acq.center_sample = trajectory.center_sample

        # set encoding space index
        if encoding_idx is None:
            encoding_idx = self.current_encoding
        acq.encoding_space_ref = encoding_idx

        # set sampling time
        acq.sample_time_us = trajectory.sample_time_us

        # set labels
        self.set_labels(*events)
        for k, v in self.label_dict.items():
            if v is not None:
                setattr(acq.idx, k, v)

        # set user defined parameters
        if user_int is not None:
            if len(user_int) > len(acq.user_int):
                warnings.warn(
                    'Length of provided user_int is larger than ismrmrd.Acquisition.user_int - ignoring extra entries',
                    stacklevel=2,
                )
            for n in range(len(acq.user_int)):
                acq.user_int = user_int[n]
        if user_float is not None:
            if len(user_float) > len(acq.user_float):
                warnings.warn(
                    'Length of provided user_float is larger than ismrmrd.Acquisition.user_float - ignoring extra entries',
                    stacklevel=2,
                )
            for n in range(len(acq.user_float)):
                acq.user_float = user_float[n]

        # set trajectory
        rotation = None
        for event in events:
            if event.type == 'rot3D':
                rotation = event.rot_quaternion
        if trajectory.traj.size:
            traj = copy.deepcopy(trajectory.traj)
            if rotation is not None:
                traj = rotation.apply(traj)
        else:
            traj = None
        if traj is not None:
            acq.traj[:] = traj

        self.acquisitions.append(acq)

    def set_labels(self, *events: list[SimpleNamespace]):
        """
        Update label dictionary using a Pulseq label event.

        Parameters
        ----------
        *events : TYPE
            List of Pulseq label events.

        """
        for event in events:
            if event.type == 'labelset':
                label = axis2key(event.label)[1]
                if label in self.label_dict:
                    self.label_dict[label] = event.value
            if event.type == 'labelinc':
                label = axis2key(event.label)[1]
                if label in self.label_dict:
                    if self.label_dict[label] is None:
                        self.label_dict[label] = 0
                    self.label_dict[label] += event.value


# %% utils
def axis2key(axis):
    if axis.lower() in _axis2key:
        return _axis2key[axis.lower()]
    else:
        return axis.lower(), axis.lower()


_axis2key = {
    'k0': ('kspace_encoding_step_0', 'kspace_encode_step_0'),
    'acq': ('kspace_encoding_step_0', 'kspace_encode_step_0'),
    'k1': ('kspace_encoding_step_1', 'kspace_encode_step_1'),
    'lin': ('kspace_encoding_step_1', 'kspace_encode_step_1'),
    'k2': ('kspace_encoding_step_2', 'kspace_encode_step_2'),
    'par': ('kspace_encoding_step_2', 'kspace_encode_step_2'),
    'avg': ('average', 'average'),
    'slc': ('slice', 'slice'),
    'eco': ('contrast', 'contrast'),
    'phs': ('phase', 'phase'),
    'rep': ('repetition', 'repetition'),
    'seg': ('segment', 'segment'),
    'user0': ('user_0', 'user0'),
    'user1': ('user_1', 'user1'),
    'user2': ('user_2', 'user2'),
    'user3': ('user_3', 'user3'),
    'user4': ('user_4', 'user4'),
    'user5': ('user_5', 'user5'),
    'user6': ('user_6', 'user6'),
    'user7': ('user_7', 'user7'),
}


def haskey(key, *userparams):
    for user in userparams:
        for el in user:
            if el.name == key:
                return True
    return False


def setparam(key, value, *userparams):
    for user in userparams:
        for el in user:
            if el.name == key:
                el.value = value
