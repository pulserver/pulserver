import pytest

from pulserver.host import SessionKey, SessionStore, revision_hash

KEY = SessionKey(pid=4242, day=20711)
LIMITS = {"max_grad": 40.0, "max_slew": 150.0}


def _generate(session, values, content="seq"):
    digest = revision_hash(session.plugin, session.limits, values)
    found = session.find(digest)
    if found is not None:
        session.select(found)
        return found
    staged = session.stage()
    (staged / "sequence.seq").write_text(content)
    return session.commit(digest, staged)


def test_a_session_key_names_its_bucket_directory(tmp_path):
    session = SessionStore(tmp_path).open(KEY, "gre2d", LIMITS)
    assert session.directory == tmp_path / "bucket" / "4242-20711"
    assert SessionKey.parse(str(KEY)) == KEY


def test_a_session_key_outside_a_float32_slot_is_refused():
    with pytest.raises(ValueError, match="float32"):
        SessionKey(pid=2**24, day=1)


def test_the_same_resolved_protocol_reuses_its_revision(tmp_path):
    session = SessionStore(tmp_path).open(KEY, "gre2d", LIMITS)
    first = _generate(session, {"TE": 2.74, "TR": 6.34})
    again = _generate(session, {"TR": 6.34, "TE": 2.74})
    assert first == again == 1
    assert sorted(p.name for p in (session.directory / "rev").iterdir()) == ["1"]


def test_the_same_protocol_from_other_source_is_another_revision():
    values = {"TE": 2.74}
    assert revision_hash("gre2d", LIMITS, values, "a") != revision_hash(
        "gre2d", LIMITS, values, "b"
    )


def test_current_points_at_the_last_generated_revision(tmp_path):
    session = SessionStore(tmp_path).open(KEY, "gre2d", LIMITS)
    _generate(session, {"TE": 8.0})
    _generate(session, {"TE": 10.0})
    assert (session.directory / "current").readlink().as_posix() == "rev/2"
    _generate(session, {"TE": 8.0})
    assert session.current == 1
    assert (session.directory / "current" / "sequence.seq").read_text() == "seq"


def test_a_restarted_store_recovers_its_sessions(tmp_path):
    session = SessionStore(tmp_path).open(KEY, "gre2d", LIMITS)
    _generate(session, {"TE": 8.0})
    recovered = SessionStore(tmp_path).get(KEY)
    assert recovered.current == 1
    assert recovered.find(revision_hash("gre2d", LIMITS, {"TE": 8.0})) == 1
    assert [s.key for s in SessionStore(tmp_path)] == [KEY]


def test_reopening_a_session_with_another_plugin_is_refused(tmp_path):
    store = SessionStore(tmp_path)
    store.open(KEY, "gre2d", LIMITS)
    assert store.open(KEY, "gre2d", dict(LIMITS)).plugin == "gre2d"
    with pytest.raises(ValueError, match="another plugin"):
        store.open(KEY, "se2d", LIMITS)


def test_a_discarded_stage_leaves_no_revision(tmp_path):
    session = SessionStore(tmp_path).open(KEY, "gre2d", LIMITS)
    staged = session.stage()
    session.discard(staged)
    assert list((session.directory / "rev").iterdir()) == []
    assert session.current is None
