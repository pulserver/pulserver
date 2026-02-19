"""Sequence Definition structure for data labeling (e.g., for recon)."""

__all__ = ["SequenceDefinition"]

import numpy as np
import mrd


class SequenceDefinition:
    """
    A class for defining and managing MRD sequence definitions across multiple sections.

    Parameters
    ----------
    ndims : int
        The number of spatial dimensions for the sequence.

    Attributes
    ----------
    _ndims : int
        Number of spatial dimensions.
    _definition : mrd.Header
        Contains the MRD sequence definition.
    _shape_set : bool
        Flag indicating whether the shape has been set.
    _current_section : int
        The index of the current section being worked on.
    _section_labels : dict
        Dictionary to map section names to their indices.
    """

    def __init__(self, ndims):
        self._ndims = ndims
        self._definition = mrd.Header()
        self._labels = []
        self._trajectory = []
        self._idx = []
        self._dwell = 0.0

        # user parameters
        self._definition.user_parameters = mrd.UserParametersType()

        # sequence parameters
        self._definition.sequence_parameters = mrd.SequenceParametersType()

        # initialize encoding
        self._definition.encoding.append(mrd.EncodingType())

        # initialize section
        self._shape_set = False
        self._current_section = 0
        self._section_labels = {}

        # assign number of spatial dimensions
        self._definition.user_parameters.user_parameter_long.append(
            mrd.UserParameterLongType(name="ndims", value=ndims)
        )

    @property
    def definition(self):  # noqa
        return self._definition

    @property
    def labels(self):  # noqa
        return self._labels

    def section(self, name: str):
        """
        Declare a new section in the sequence.

        Parameters
        ----------
        name : str
            Name of the new section.

        Raises
        ------
        AssertionError
            If a section with the same name already exists.
        """
        assert (
            name not in self._section_labels
        ), f"Section {name} already exists - please use another name."
        if self._section_labels:
            self._shape_set = False
            self._current_section += 1
            self._definition.encoding.append(mrd.EncodingType())
            self._copy_from_section_0()  # Copy fields from section 0

        self._section_labels[name] = self._current_section

    def set_definition(self, key: str, *args, **kwargs):
        """
        Set a specific sequence parameter.

        Parameters
        ----------
        key : str
            The parameter to be set. Valid keys are 'fov', 'shape', 'limits', and 'trajectory'.
        *args :
            Positional arguments for the specific parameter.
        **kwargs :
            Keyword arguments for the specific parameter.

        Raises
        ------
        KeyError
            If 'fov' is set before 'shape' in 2D acquisitions.
        """
        if key.lower() == "fov" and self._ndims == 3:
            self._set_fov_3D(*args, **kwargs)
        elif key.lower() == "fov" and self._ndims == 2:
            if self._shape_set is False:
                raise KeyError("For 2D acquisitions, 'shape' must be set before 'fov'")
            self._set_fov_2D(*args, **kwargs)
        elif key.lower() == "shape":
            self._set_shape(*args, **kwargs)
            self._shape_set = True
        elif key.lower() == "limits":
            self._set_limits(*args, **kwargs)
        elif key.lower() == "dwell":
            self._set_dwell(*args, **kwargs)
        elif key.lower() == "trajectory-type":
            self._set_trajectory_type(*args, **kwargs)
        elif key.lower() == "trajectory":
            self._set_trajectory(*args, **kwargs)
        elif key.lower() == "flip":
            self._set_flip(*args, **kwargs)
        elif key.upper() == "TE":
            self._set_echo_time(*args, **kwargs)
        elif key.upper() == "TR":
            self._set_repetition_time(*args, **kwargs)
        elif key.upper() == "TI" or key.lower() == "prep_time":
            self._set_preparation_time(*args, **kwargs)
        else:
            self._set_parameter(key, args[0])

    def _set_shape(self, nx, ny, nz):
        """
        Set the encoded and reconstruction space matrix size for the current section.

        Parameters
        ----------
        nx : int
            Size in the x-dimension.
        ny : int
            Size in the y-dimension.
        nz : int
            Size in the z-dimension.
        """
        idx = self._current_section

        # set encoded shape size
        self._definition.encoding[idx].encoded_space.matrix_size.x = nx
        self._definition.encoding[idx].encoded_space.matrix_size.y = ny
        self._definition.encoding[idx].encoded_space.matrix_size.z = nz

        # set recon shape size (= encoded shape)
        self._definition.encoding[idx].recon_space.matrix_size.x = nx
        self._definition.encoding[idx].recon_space.matrix_size.y = ny
        self._definition.encoding[idx].recon_space.matrix_size.z = nz
        self._shape_set = True

    def _set_fov_3D(self, fov_x, fov_y, fov_z):
        """
        Set the Field of View (FOV) for 3D acquisitions.

        Parameters
        ----------
        fov_x : float
            FOV in the x-dimension.
        fov_y : float
            FOV in the y-dimension.
        fov_z : float
            FOV in the z-dimension.
        """
        idx = self._current_section

        # set encoded fov
        self._definition.encoding[idx].encoded_space.field_of_view_mm.x = fov_x
        self._definition.encoding[idx].encoded_space.field_of_view_mm.y = fov_y
        self._definition.encoding[idx].encoded_space.field_of_view_mm.z = fov_z

        # set recon fov (= encoded fov)
        self._definition.encoding[idx].recon_space.field_of_view_mm.x = fov_x
        self._definition.encoding[idx].recon_space.field_of_view_mm.y = fov_y
        self._definition.encoding[idx].recon_space.field_of_view_mm.z = fov_z

    def _set_fov_2D(self, fov_x, fov_y, slice_spacing):
        """
        Set the Field of View (FOV) for 2D acquisitions.

        Parameters
        ----------
        fov_x : float
            FOV in the x-dimension.
        fov_y : float
            FOV in the y-dimension.
        slice_spacing : float
            The spacing between slices.
            This is equal to slice thickness + slice gap.
        """
        idx = self._current_section

        # set encoded fov
        nz = self._definition.encoding[idx].encoded_space.matrix_size.z
        self._definition.encoding[idx].encoded_space.field_of_view_mm.x = fov_x
        self._definition.encoding[idx].encoded_space.field_of_view_mm.y = fov_y
        self._definition.encoding[idx].encoded_space.field_of_view_mm.z = (
            nz * slice_spacing
        )

        # set recon fov (= encoded fov)
        nz = self._definition.encoding[idx].recon_space.matrix_size.z

        self._definition.encoding[idx].recon_space.field_of_view_mm.x = fov_x
        self._definition.encoding[idx].recon_space.field_of_view_mm.y = fov_y
        self._definition.encoding[idx].recon_space.field_of_view_mm.z = (
            nz * slice_spacing
        )

    def _set_dwell(self, dwell: float):
        self._dwell = dwell

    def _set_limits(
        self,
        n_views=None,
        n_slices=None,
        n_partitions=None,
        n_contrasts=None,
        n_frames=None,
    ):
        """
        Set the k-space encoding limits for the current section.

        Parameters
        ----------
        n_views : int, optional
            Number of views.
        n_slices : int, optional
            Number of slices.
        n_partitions : int, optional
            Number of partitions.
        n_contrasts : int, optional
            Number of contrasts.
        n_frames : int, optional
            Number of frames.
        """
        idx = self._current_section

        # Assign view limits
        if n_views is not None:
            view_limit = mrd.LimitType()
            view_limit.minimum = 0
            view_limit.maximum = n_views - 1
            view_limit.center = int(n_views / 2)
            self._definition.encoding[idx].encoding_limits.kspace_encoding_step_1 = (
                view_limit
            )

        # Assign partition limits
        if n_partitions is not None:
            partition_limit = mrd.LimitType()
            partition_limit.minimum = 0
            partition_limit.maximum = n_partitions - 1
            partition_limit.center = int(n_partitions / 2)
            self._definition.encoding[idx].encoding_limits.kspace_encoding_step_2 = (
                partition_limit
            )

        # Assign slice limits
        if n_slices is not None:
            slice_limit = mrd.LimitType()
            slice_limit.minimum = 0
            slice_limit.maximum = n_slices - 1
            slice_limit.center = int(n_slices / 2)
            self._definition.encoding[idx].encoding_limits.slice = slice_limit

        # Assign contrast limits
        if n_contrasts is not None:
            contrast_limit = mrd.LimitType()
            contrast_limit.minimum = 0
            contrast_limit.maximum = n_slices - 1
            contrast_limit.center = int(n_slices / 2)
            self._definition.encoding[idx].encoding_limits.contrast = contrast_limit

        # Assign frames limits
        if n_frames is not None:
            frame_limit = mrd.LimitType()
            frame_limit.minimum = 0
            frame_limit.maximum = n_frames - 1
            frame_limit.center = int(n_frames / 2)
            self._definition.encoding[idx].encoding_limits.repetition = frame_limit

    def _set_trajectory_type(self, trajectory_type: str | None = None):
        """
        Set the k-space trajectory type for the current section.

        Parameters
        ----------
        trajectory_type : str, optional
            The type of k-space trajectory. Options are 'cartesian', 'epi', 'radial',
            'goldenangle', 'spiral', or 'other'. Defaults to 'cartesian'.
        """
        idx = self._current_section

        # String to enum map
        _trajectory_map = {
            "cartesian": 0,
            "epi": 1,
            "radial": 2,
            "goldenangle": 3,
            "spiral": 4,
            "other": 5,
        }

        # Assign default
        if trajectory_type is None:
            trajectory_type = "cartesian"

        # Ensure lower case
        trajectory_type = trajectory_type.lower()

        # Get other
        if trajectory_type not in _trajectory_map.keys():
            trajectory_type = "other"

        self._definition.encoding[idx].trajectory = mrd.Trajectory(
            _trajectory_map[trajectory_type]
        )

    def _set_trajectory(
        self,
        trajectory: np.ndarray,
        dcf: np.ndarray | None = None,
        discard_pre: int = 0,
        discard_post: int = 0,
    ):
        """
        Set the k-space trajectory for the current section.

        Parameters
        ----------
        trajectory : np.ndarray
            K-space trajectory of shape ``(..., nsamples, ndims)``.
            Order for leading axes (from outer to inner) is
            ``"contrast/frames", "partitions", "shots"``.
            Assume trajectory shape is unsqueezed.
        dcf : np.ndarray, optional
            Density compensation function of shape ``(..., nsamples)``.
            Order for leading axes (from outer to inner) is
            ``"contrast/frames", "partitions", "shots"``.
            Assume trajectory shape is unsqueezed.
            The default is ``None`` (no compensation).
        discard_pre : int, optional
            Number of samples to be discarded at beginning of readout.
            The default is ``0``.
        discard_post : int, optional
            Number of samples to be discarded at the end of readout.
            The default is ``0``.

        """
        # Put dcf in ndim+1 position along dims axis of trajectory
        if dcf is not None:
            trajectory = np.concatenate((trajectory, dcf[..., None]), axis=-1)

        self._trajectory.append(trajectory)
        self._idx.append((discard_pre, discard_post))

    def _set_flip(self, flip):
        flip = np.asarray(flip)
        self._definition.sequence_parameters.flip_angle_deg = flip.tolist()

    def _set_echo_time(self, TE):
        TE = np.asarray(TE)
        self._definition.sequence_parameters.t_e = TE.tolist()

    def _set_repetition_time(self, TR):
        TR = np.asarray(TR)
        self._definition.sequence_parameters.t_r = TR.tolist()

    def _set_preparation_time(self, TI):
        TI = np.asarray(TI)
        self._definition.sequence_parameters.t_i = TI.tolist()

    def _set_parameter(self, name, value):
        """
        Add field to header structure.

        Parameters
        ----------
        name : str
            Name of the field.
        value : int | float | str | ArrayLike
            Value(s) to be appended.

        """
        # types
        floating = [float, np.float32, np.float64]
        integers = [
            int,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]

        # infer type
        if np.isscalar(value):
            dtype = type(value)
            value = [value]
        else:
            dtype = type(value[0])

        # add
        if dtype == str:
            for val in value:
                el = mrd.UserParameterStringType(name=name, value=val)
                self._definition.user_parameters.user_parameter_string.append(el)
        elif dtype in floating:
            for val in value:
                el = mrd.UserParameterDoubleType(name=name, value=val)
                self._definition.user_parameters.user_parameter_double.append(el)
        elif dtype in integers:
            for val in value:
                el = mrd.UserParameterLongType(name=name, value=val)
                self._definition.user_parameters.user_parameter_long.append(el)

    def _copy_from_section_0(self):
        """Copy undefined fields from section 0 to the current section."""
        current_encoding = self._definition.encoding[self._current_section]
        section_0_encoding = self._definition.encoding[0]

        fields_to_copy = [
            "encoded_space",
            "recon_space",
            "encoding_limits",
            "trajectory",
        ]

        for field in fields_to_copy:
            setattr(current_encoding, field, getattr(section_0_encoding, field))

        # copy trajectory and adc index
        if self._trajectory:
            self._trajectory.append(self._trajectory[0])

        if self._idx:
            self._idx.append(self._idx[0])

    def set_label(
        self,
        iy: int = None,
        iz: int = 0,
        islice: int = 0,
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
        idx = self._current_section
        acq = mrd.Acquisition()
        acq.encoding_space_ref = idx
        acq.sample_time_us = self._dwell * 1e6

        if self._idx:
            acq.discard_pre, acq.discard_post = self._idx[idx]
        else:
            acq.discard_pre, acq.discard_post = 0, 0

        if ishot is not None and iy is not None:
            raise ValueError(
                "Provide either iy (Cartesian) or ishot (Non Cartesian), not both"
            )
        elif ishot is not None:
            acq.idx.kspace_encode_step_1 = ishot
        elif iy is not None:
            acq.idx.kspace_encode_step_1 = iy

        if islice != 0 and iz != 0:
            raise ValueError(
                "Provide either islice (2D multislice) or iz (3D), not both"
            )
        elif iz != 0:
            acq.idx.kspace_encode_step_2 = iz
        elif islice != 0:
            acq.idx.slice = islice

        if iframe != 0 and icontrast != 0:
            raise ValueError(
                "Provide either icontrast (multicontrast MRI) or iframe (dynamic MRI), not both"
            )
        elif icontrast != 0:
            acq.idx.contrast = icontrast
        elif iframe != 0:
            acq.idx.repetition = iframe

        # update trajectory
        if ishot is not None and self._trajectory:
            ishot = min(ishot, self._trajectory[idx].shape[2])
        if self._trajectory:
            it = min(iframe, self._trajectory[idx].shape[0])
            it = min(icontrast, self._trajectory[idx].shape[0])
            iz = min(iz, self._trajectory[idx].shape[1])
            acq.trajectory = self._trajectory[idx][it, iz, ishot]

        # append acquisition
        self._labels.append(mrd.StreamItem.Acquisition(acq))
