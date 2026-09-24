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

   ScannerSequence
   load_plugin
   TimeParam
   FloatParam
   IntParam
   BoolParam
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
   summary
   play
   chain
   cache_path

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
   Validation
   format_validation
   parse_validation

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
   SequenceType
   ImagingMode
   PreparationType
   TriggerType

.. currentmodule:: pulserver.recon

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReconPlugin
   Gadget
   load_plugin
   ReconContext
   ExamCache
   ReconData
   ReconBuffer
   ReconResult

.. currentmodule:: pulserver.vre

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReconProxy
   DesignCache
   Design
   DESIGN_PARAMETER
   SequenceTable
   TableSpace
   enrich_header
   enrich_acquisition

