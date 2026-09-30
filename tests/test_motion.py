"""The pose file a tracker publishes into and a running scan reads."""

import struct

import pytest

from pulserver.proxy._motion import (
    LAYOUT,
    MAGIC,
    SLOTS,
    MotionWriter,
    _check_of,
)

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
TURNED = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

_HEADER_BYTES = 16
_SLOT_BYTES = 48


def slot_of(path, slot):
    raw = path.read_bytes()
    at = _HEADER_BYTES + slot * _SLOT_BYTES
    return list(struct.unpack("=12I", raw[at : at + _SLOT_BYTES]))


def rotation_of(words):
    return list(struct.unpack("=9f", struct.pack("=9I", *words[3:])))


def test_the_header_says_what_the_file_is(tmp_path):
    with MotionWriter(tmp_path / "motion.buf") as writer:
        magic, layout, slots, slot_bytes = struct.unpack(
            "=4I", writer.path.read_bytes()[:16]
        )
    assert (magic, layout, slots, slot_bytes) == (MAGIC, LAYOUT, SLOTS, _SLOT_BYTES)


def test_a_pose_rings_through_the_slots(tmp_path):
    """Consecutive poses go to different slots, so one is always whole."""
    path = tmp_path / "motion.buf"
    with MotionWriter(path) as writer:
        first = writer.publish(IDENTITY)
        second = writer.publish(TURNED)
    assert (first, second) == (1, 2)
    assert slot_of(path, 0)[0] != slot_of(path, 1)[0]
    assert {slot_of(path, 0)[0], slot_of(path, 1)[0]} == {1, 2}


def test_a_published_pose_carries_its_rotation_and_a_check_that_agrees(tmp_path):
    path = tmp_path / "motion.buf"
    with MotionWriter(path) as writer:
        writer.publish(TURNED, rescan=True)
    words = slot_of(path, 0)
    assert words[0] == 1
    assert words[1] == 1
    assert words[2] == _check_of(words)
    assert rotation_of(words) == pytest.approx(TURNED)


def test_a_slot_never_written_reads_as_empty(tmp_path):
    """A check of zero is what says a slot holds nothing, so none is written."""
    path = tmp_path / "motion.buf"
    with MotionWriter(path) as writer:
        writer.publish(IDENTITY)
    assert slot_of(path, 1) == [0] * 12
    assert _check_of(slot_of(path, 0)) != 0


def test_a_pose_the_scan_has_not_played_is_not_echoed(tmp_path):
    with MotionWriter(tmp_path / "motion.buf") as writer:
        writer.publish(IDENTITY)
        assert writer.applied() is None


def test_the_echo_names_the_pose_the_scan_played(tmp_path):
    """The scan writes the echo; reading it is how the translation follows."""
    path = tmp_path / "motion.buf"
    with MotionWriter(path) as writer:
        writer.publish(IDENTITY)
        writer.publish(TURNED)
        echo = [2, 17, 0]
        echo[2] = _check_of(echo)
        raw = bytearray(path.read_bytes())
        at = _HEADER_BYTES + SLOTS * _SLOT_BYTES
        raw[at : at + 12] = struct.pack("=3I", *echo)
        path.write_bytes(bytes(raw))
        assert writer.applied() == (2, 17)


def test_a_rotation_that_is_not_nine_values_is_refused(tmp_path):
    with (
        MotionWriter(tmp_path / "motion.buf") as writer,
        pytest.raises(ValueError, match="nine values"),
    ):
        writer.publish([1.0, 0.0, 0.0])


def test_creating_replaces_the_poses_of_an_earlier_scan(tmp_path):
    """A scan reads the poses of its own run."""
    path = tmp_path / "motion.buf"
    with MotionWriter(path) as writer:
        writer.publish(TURNED)
    with MotionWriter(path) as writer:
        assert writer.version == 0
        assert slot_of(path, 0) == [0] * 12
