"""Tests for mrdserver.constants — struct sizes and message identifiers."""

from mrdserver import constants


def test_gadget_message_identifier_size():
    assert constants.GadgetMessageIdentifier.size == 2


def test_gadget_message_length_size():
    assert constants.GadgetMessageLength.size == 4


def test_gadget_message_config_file_size():
    assert constants.GadgetMessageConfigurationFile.size == 1024


def test_identifier_pack_unpack_roundtrip():
    mid = constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
    packed = constants.GadgetMessageIdentifier.pack(mid)
    assert constants.GadgetMessageIdentifier.unpack(packed)[0] == mid


def test_message_id_ranges():
    assert constants.GADGET_MESSAGE_CONFIG == 2
    assert constants.GADGET_MESSAGE_HEADER == 3
    assert constants.GADGET_MESSAGE_CLOSE == 4
    assert constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION == 1008
    assert constants.GADGET_MESSAGE_ISMRMRD_IMAGE == 1022
    assert constants.GADGET_MESSAGE_ISMRMRD_WAVEFORM == 1026
    assert constants.GADGET_MESSAGE_DICOM_WITHNAME == 1018
