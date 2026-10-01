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


def test_a_pose_states_where_the_object_is_and_what_to_do():
    """A reconstruction states a pose as a waveform the proxy reads."""
    from pulserver.proxy._motion import Pose, pose_of, pose_waveform

    pose = Pose(
        rotation=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        translation_m=(0.01, -0.02, 0.003),
        rescan=True,
    )
    read = pose_of(pose_waveform(pose))
    assert read.rotation == pytest.approx(pose.rotation)
    assert read.translation_m == pytest.approx(pose.translation_m, abs=1e-7)
    assert read.rescan is True


def test_a_waveform_that_is_not_a_pose_is_not_read_as_one():
    """A physiological trace must not be mistaken for a pose."""
    import ismrmrd
    import numpy as np

    from pulserver.proxy._motion import pose_of

    trace = ismrmrd.Waveform.from_array(np.zeros((1, 32), dtype=np.uint32))
    trace.waveform_id = 0
    assert pose_of(trace) is None


def test_a_pose_too_short_to_read_is_passed_over():
    """A malformed pose leaves the last one standing rather than stopping a scan."""
    import ismrmrd
    import numpy as np

    from pulserver.proxy._motion import POSE_WAVEFORM_ID, pose_of

    short = ismrmrd.Waveform.from_array(np.zeros((1, 4), dtype=np.uint32))
    short.waveform_id = POSE_WAVEFORM_ID
    assert pose_of(short) is None


def test_the_translation_does_not_cross_to_the_scan(tmp_path):
    """Only the rotation reaches the sequencer; the translation is the recon's."""
    from pulserver.proxy._motion import Pose

    pose = Pose(rotation=TURNED, translation_m=(0.05, 0.0, 0.0))
    path = tmp_path / "motion.buf"
    with MotionWriter(path) as writer:
        writer.publish(pose.rotation, rescan=pose.rescan)
        words = slot_of(path, 0)
    assert rotation_of(words) == pytest.approx(TURNED)
    # Twelve words of version, rescan, check and nine of rotation: no room for
    # a translation, and none is written.
    assert len(words) == 12


class _Stream:
    """A connection's worth of items, and what was sent on."""

    def __init__(self, items):
        self._items = list(items)
        self.sent = []
        self.unreadable = None

    def __iter__(self):
        return iter(self._items)

    def send(self, item):
        self.sent.append(item)


def test_a_pose_reaches_the_scan_and_not_whoever_asked_for_the_images():
    """A pose is addressed to the sequencer; the client asked for pictures."""
    import ismrmrd
    import numpy as np

    from pulserver.proxy._motion import Pose, pose_waveform
    from pulserver.proxy._proxy import _relay

    pose = Pose(rotation=TURNED)
    trace = ismrmrd.Waveform.from_array(np.zeros((1, 8), dtype=np.uint32))
    trace.waveform_id = 0
    source = _Stream([trace, pose_waveform(pose), trace])
    client = _Stream([])
    taken = []
    _relay(source, client, poses=taken.append)

    assert [p.rotation for p in taken] == [pytest.approx(TURNED)]
    assert len(client.sent) == 2, "the pose was sent on to the client"
    assert all(w.waveform_id == 0 for w in client.sent)


PREAMBLE = """# Pulseq sequence file
[VERSION]
major 1

[DEFINITIONS]
AdcRasterTime 2e-06
{key}GradientRasterTime 4e-06

[BLOCKS]
 1 58 0 0 0 0 0 0
"""


def seqfile(tmp_path, key=""):
    path = tmp_path / "sequence.seq"
    path.write_text(PREAMBLE.format(key=key))
    return path


def test_a_scan_that_asks_to_be_corrected_for_motion_says_so(tmp_path):
    from pulserver.proxy._designs import _asks_for_motion_correction

    assert _asks_for_motion_correction(seqfile(tmp_path, "EnablePmc 1\n"))


@pytest.mark.parametrize("key", ["", "EnablePmc 0\n"])
def test_a_scan_that_does_not_ask_is_not_corrected(tmp_path, key):
    from pulserver.proxy._designs import _asks_for_motion_correction

    assert not _asks_for_motion_correction(seqfile(tmp_path, key))


def test_the_key_is_read_only_inside_the_definitions(tmp_path):
    """A later block naming it is not the scan asking."""
    from pulserver.proxy._designs import _asks_for_motion_correction

    path = tmp_path / "sequence.seq"
    path.write_text(
        "[DEFINITIONS]\nGradientRasterTime 4e-06\n\n[BLOCKS]\nEnablePmc 1\n"
    )
    assert not _asks_for_motion_correction(path)


def test_a_file_that_is_not_there_asks_for_nothing(tmp_path):
    from pulserver.proxy._designs import _asks_for_motion_correction

    assert not _asks_for_motion_correction(tmp_path / "no_such.seq")
