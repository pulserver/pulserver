:orphan:

API object index
================

The stub page of every documented object. The API pages carry the same
object lists as tables, each entry linking the stub this page writes;
writing them here keeps them out of the toctree the sidebar is built
from.

.. currentmodule:: pulserver.design

.. autosummary::
   :toctree: generated
   :nosignatures:

   SequencePlugin
   Evaluation
   Protocol
   ScannerSequence
   load_plugin
   load_exam
   RfLayout
   RfControl
   TimeParam
   FloatParam
   IntParam
   BoolParam
   ChoiceParam
   StringListParam
   ConfigParam
   Description

.. currentmodule:: pulserver.host

.. autosummary::
   :toctree: generated
   :nosignatures:

   call
   DesignStore
   design_identity
   design_id

.. currentmodule:: pulserver.ir

.. autosummary::
   :toctree: generated
   :nosignatures:

   convert
   prescribe
   check
   CheckLimits
   sar_ratios
   SarRatio
   played_rf
   summary
   play
   plan_waves
   WaveBudget
   Grouping
   VendorProfile
   read_vendor
   Quantity
   sample_wave
   playout
   Prescan
   repetition_gradients
   chain
   cache_path

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/enum.rst

   Format

.. currentmodule:: pulserver.mrd

.. autosummary::
   :toctree: generated
   :nosignatures:

   has_acquisition_flag
   acquisition_label
   acquisition_labels
   AcquisitionBucket
   AcquisitionBucketStats
   MrdMetadata
   EncodingSpace
   LOOP_COUNTERS
   user_parameter
   max_stored_value
   coil_combine
   center_crop
   as_numpy
   read_chain
   SequenceDefinitions
   ReadoutTable

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/enum.rst

   AcquisitionFlag

.. currentmodule:: pulserver.protocol

.. autosummary::
   :toctree: generated
   :nosignatures:

   Parameter
   ProtocolKey
   UIParam
   PRESCRIPTION
   FOV_OFFSET
   FOV_ROTATION
   prescribed_offset
   prescribed_rotation
   PROTOCOL_BEGIN
   PROTOCOL_END
   format_listing
   parse_listing
   format_values
   parse_values
   format_prescription
   parse_prescription
   Validation
   format_validation
   parse_validation
   RfDefinitionRecord
   format_rf_definitions
   parse_rf_definitions
   RfLayoutRecord
   format_rf_layout
   parse_rf_layout

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/enum.rst

   Kind
   InputMode
   TEPreset
   TRPreset
   FloatKey
   IntKey
   BoolKey
   EnumKey
   ConfigKey
   UserKey
   UserNameKey
   SequenceType
   ImagingMode
   PreparationType
   TriggerType

.. currentmodule:: pulserver.proxy

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReconProxy
   ReconServer
   LocalReconstruction
   DesignCache
   Design
   DESIGN_PARAMETER
   DesignIntake
   SequenceTable
   TableSpace
   enrich_header
   enrich_acquisition

.. currentmodule:: pulserver.recon

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReconPlugin
   Gadget
   load_plugin
   AsymmetricEcho
   Prewhiten
   RemoveReadoutOversampling
   ReconContext
   ExamCache
   ExamImage
   B0_MAP
   B1_MAP
   COIL_SENSITIVITIES
   NOISE_COVARIANCE
   EXAM_ARTIFACTS
   coil_maps
   CoilSensitivities
   MissingCalibration
   CoilCompression
   ReconData
   ReconBuffer
   ReconResult

.. currentmodule:: pulserver.validate

.. autosummary::
   :toctree: generated
   :nosignatures:

   validate
   Comparison
   ChannelAgreement
   gradient_tolerance_mt_per_m
   read_waveform_xml
   PlayedWaveforms
   VENDORS

.. currentmodule:: pulserver.virtual

.. autosummary::
   :toctree: generated
   :nosignatures:

   trajectory
   acquire
   simulate
   excited
   Slabs
   Isochromats
   Repetitions
   RigidMotion
   Scan
   Chunk
   SAMPLE_RATE
   export
   Phantom
   BrainWeb
   Ellipse
   localizer
   Coil
   COILS
   send
   record
   Console

