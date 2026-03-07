API Reference
=============

This is the API documentation for the `pulserver` package, focusing on development tools for MR sequence design and reconstruction.

.. automodule:: pulserver
    :members:
    :undoc-members:
    :show-inheritance:

Command-Line Interface
----------------------

Server API
----------

Toolbox
-------

The Toolbox provides three abstraction layers:
high-level sequence applications, mid-level design and reconstruction
building blocks, and low-level Pulseq and MRD backends.

High-level Applications
^^^^^^^^^^^^^^^^^^^^^^^

Middle-level Abstractions
^^^^^^^^^^^^^^^^^^^^^^^^

Sequence Design
"""""""""""""""

.. automodule:: pulserver.design
    :members:
    :undoc-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :recursive:

**Magnetization Preparation Kernels**

    pulserver.design.InversionPrep
    pulserver.design.T2Prep
    pulserver.design.T1T2Prep
    pulserver.design.MTPrep
    pulserver.design.FatSaturation

**Imaging Readout Kernels**

    pulserver.design.ArbitraryGRE2D
    pulserver.design.ArbitraryGRE3D
    pulserver.design.EpiGRE2D
    pulserver.design.EpiGRE3D
    pulserver.design.LineGRE2D
    pulserver.design.LineGRE3D
    pulserver.design.WaveGRE3D
    pulserver.design.ArbitraryFSE2D
    pulserver.design.ArbitraryFSE3D
    pulserver.design.EpiFSE2D
    pulserver.design.EpiFSE3D
    pulserver.design.LineFSE2D
    pulserver.design.LineFSE3D
    pulserver.design.WaveFSE3D

**Encoding Tables**

    pulserver.design.Cartesian2DTable
    pulserver.design.Cartesian2DsmsTable
    pulserver.design.Caipirinha3DTable
    pulserver.design.EPI2DTable
    pulserver.design.EPI2DsmsTable
    pulserver.design.EPI3DTable
    pulserver.design.NonCartesian2DTable
    pulserver.design.NonCartesian2DsmsTable
    pulserver.design.NonCartesian3DTable
    pulserver.design.NonCartesian3DStackTable
    pulserver.design.Shuffling3DTable
    pulserver.design.PoissonDisk3DTable

Reconstruction Gadgets
""""""""""""""""""""""

Low-level Backends
^^^^^^^^^^^^^^^^^^

Pulseq
"""""""

.. automodule:: pulserver.pulseq
    :members:
    :undoc-members:
    :show-inheritance:

.. autosummary::
    :toctree: _autosummary
    :recursive:

**Pulseq Core**

    pulserver.pulseq.Sequence
    pulserver.pulseq.Opts

**Pulseq Utilities**

    pulserver.pulseq.DUMMY_OPTS
    pulserver.pulseq.add_gradients
    pulserver.pulseq.align
    pulserver.pulseq.calc_duration
    pulserver.pulseq.calc_kspace_band_jump
    pulserver.pulseq.calc_kspace_line_jump
    pulserver.pulseq.calc_kspace_readout_params
    pulserver.pulseq.calc_rf_bandwidth
    pulserver.pulseq.calc_rf_center
    pulserver.pulseq.convert
    pulserver.pulseq.points_to_waveform
    pulserver.pulseq.scale_waveform
    pulserver.pulseq.split_waveform
    pulserver.pulseq.split_waveform_at
    pulserver.pulseq.traj_to_grad
    pulserver.pulseq.time_revert_waveform

**RF Pulses**

    pulserver.pulseq.make_adiabatic_inversion
    pulserver.pulseq.make_adiabatic_t2prep
    pulserver.pulseq.make_arbitrary_rf
    pulserver.pulseq.make_block_pulse
    pulserver.pulseq.make_fermi_pulse
    pulserver.pulseq.make_slr_pulse
    pulserver.pulseq.make_sms_pulse
    pulserver.pulseq.make_spsp_pulse

**Gradient Waveforms**

    pulserver.pulseq.make_arbitrary_grad
    pulserver.pulseq.make_epi
    pulserver.pulseq.make_extended_trapezoid
    pulserver.pulseq.make_extended_trapezoid_area
    pulserver.pulseq.make_spiral
    pulserver.pulseq.make_trapezoid
    pulserver.pulseq.make_wave

**Sampling Masks**

    pulserver.pulseq.make_caipirinha_sampling
    pulserver.pulseq.make_partial_fourier_sampling
    pulserver.pulseq.make_poisson_disk_sampling
    pulserver.pulseq.make_regular_sampling

**Acquisition Ordering**

    pulserver.pulseq.make_centerout_ordering_1d
    pulserver.pulseq.make_centerout_ordering_2d
    pulserver.pulseq.make_interleaved_ordering_1d
    pulserver.pulseq.make_radial_ordering_2d
    pulserver.pulseq.make_random_ordering_1d
    pulserver.pulseq.make_random_ordering_2d
    pulserver.pulseq.make_spiral_ordering_2d

MRD
"""
