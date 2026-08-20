"""The sequence description a scan carries, decoded from its MRD waveforms."""

from ._seqdesc import (
    AdcRole,
    EventType,
    RfDefinition,
    RfShape,
    RfUse,
    SequenceDescription,
    SequenceDescriptionCollection,
    SequenceEvent,
    SequenceParameters,
    ShimDefinition,
    decode_sequence_description,
    decompress_shape,
)

__all__ = [
    "AdcRole",
    "EventType",
    "RfDefinition",
    "RfShape",
    "RfUse",
    "SequenceDescription",
    "SequenceDescriptionCollection",
    "SequenceEvent",
    "SequenceParameters",
    "ShimDefinition",
    "decode_sequence_description",
    "decompress_shape",
]
