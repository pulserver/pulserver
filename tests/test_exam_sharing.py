"""The caches of one exam, held by the proxies of a host rather than by one of them."""

import pytest
from conftest import exam_header

import pulserver.recon as recon
from pulserver.recon._runtime.exam import DEFAULT_EXAM_DIRECTORY, ExamCacheManager


def test_two_proxies_on_one_exam_read_each_others_maps(tmp_path):
    """The point of a shared location: one proxy per acquisition, one exam between them."""
    first = ExamCacheManager(directory=tmp_path)
    second = ExamCacheManager(directory=tmp_path)
    with first.lease(exam_header("exam-1")) as cache:
        cache[recon.B1_MAP] = [1.0, 0.9, 0.8]
    with second.lease(exam_header("exam-1")) as cache:
        assert cache[recon.B1_MAP] == [1.0, 0.9, 0.8]
    first.close()
    second.close()


def test_a_proxy_moving_on_does_not_take_the_exam_from_one_still_on_it(tmp_path):
    """Retiring an exam removes its artifacts; another proxy's series still wants them."""
    staying = ExamCacheManager(directory=tmp_path)
    moving = ExamCacheManager(directory=tmp_path)
    with moving.lease(exam_header("exam-1")) as cache:
        cache[recon.B0_MAP] = [7.0]

    with staying.lease(exam_header("exam-1")) as held:
        # The other proxy's next series is a different exam, which retires the
        # first -- while this one is still reconstructing under it.
        with moving.lease(exam_header("exam-2")):
            pass
        assert held[recon.B0_MAP] == [7.0]
        directory = held.directory
    assert directory.is_dir()

    staying.close()
    moving.close()
    assert not directory.exists()


def test_the_last_proxy_off_an_exam_takes_its_artifacts_with_it(tmp_path):
    """Nothing holds the exam, so nothing wants what was measured under it."""
    manager = ExamCacheManager(directory=tmp_path)
    with manager.lease(exam_header("exam-1")) as cache:
        cache[recon.B1_MAP] = [1.0]
        directory = cache.directory
    assert directory.is_dir()
    with manager.lease(exam_header("exam-2")):
        pass
    assert not directory.exists()
    manager.close()


def test_an_exam_in_memory_is_nobody_elses(tmp_path):
    """Without a directory there is nothing to share and nothing to lock."""
    manager = ExamCacheManager()
    with manager.lease(exam_header("exam-1")) as cache:
        cache[recon.B1_MAP] = [1.0]
        assert cache.directory is None
    other = ExamCacheManager()
    with other.lease(exam_header("exam-1")) as cache:
        assert recon.B1_MAP not in cache
    manager.close()
    other.close()


def test_the_default_location_is_the_hosts_not_a_process_of_its_own():
    """A directory per proxy is a map measured by a series no other series can read."""
    assert "pulserver-exams" in str(DEFAULT_EXAM_DIRECTORY)


@pytest.mark.parametrize("name", ["B0_MAP", "B1_MAP", "COIL_SENSITIVITIES"])
def test_a_shared_name_survives_the_crossing(tmp_path, name):
    """What one proxy writes under a name, another reads under it."""
    value = [1.5, 2.5]
    writer = ExamCacheManager(directory=tmp_path)
    reader = ExamCacheManager(directory=tmp_path)
    with writer.lease(exam_header("exam-9")) as cache:
        cache[getattr(recon, name)] = value
    with reader.lease(exam_header("exam-9")) as cache:
        assert cache[getattr(recon, name)] == value
    writer.close()
    reader.close()


def test_the_shared_root_is_readable_by_its_owner_alone(tmp_path):
    """An exam's files are unpickled, so writing one is running code in a worker."""
    import stat

    from pulserver.proxy import ReconProxy

    root = tmp_path / "exams"
    store = tmp_path / "store"
    store.mkdir()
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    proxy = ReconProxy(store, plugins, exam_directory=root, spares=0)
    try:
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
    finally:
        proxy.close()


def test_the_default_root_is_named_for_the_user_it_belongs_to(tmp_path):
    """A name another user could take first is a name they could leave a pickle under."""
    import os

    assert str(os.getuid()) in DEFAULT_EXAM_DIRECTORY.name


def test_an_exam_nothing_has_leased_for_a_while_is_flushed(tmp_path):
    """No message ends an exam, so going unused for long enough does."""
    manager = ExamCacheManager(directory=tmp_path)
    with manager.lease(exam_header("exam-1")) as cache:
        cache[recon.B1_MAP] = [1.0]
        directory = cache.directory
    manager.expire()
    assert directory.is_dir(), "an exam between two series was flushed"
    manager.expire(idle=0.0)
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []
    manager.close()


def test_an_exam_is_not_flushed_under_a_series_still_reconstructing(tmp_path):
    manager = ExamCacheManager(directory=tmp_path)
    with manager.lease(exam_header("exam-1")) as cache:
        cache[recon.B0_MAP] = [7.0]
        manager.expire(idle=0.0)
        assert cache.directory.is_dir()
        assert cache[recon.B0_MAP] == [7.0]
    manager.close()


def test_an_idle_exam_is_left_to_the_proxy_still_on_it(tmp_path):
    """Another proxy's idle exam is that proxy's to expire."""
    staying = ExamCacheManager(directory=tmp_path)
    sweeping = ExamCacheManager(directory=tmp_path)
    with staying.lease(exam_header("exam-1")) as cache:
        cache[recon.B1_MAP] = [1.0]
        directory = cache.directory
    sweeping.expire(idle=0.0, now=float("inf"))
    assert directory.is_dir()
    staying.close()
    sweeping.close()


def test_the_exam_of_a_proxy_that_died_is_swept_once_idle(tmp_path):
    """A proxy that died holds no lock, and leaves its exam on disk."""
    import os

    directory = tmp_path / "abandoned"
    directory.mkdir()
    (directory / "map").write_bytes(b"1")
    lock = tmp_path / "abandoned.lock"
    lock.touch()
    manager = ExamCacheManager(directory=tmp_path)
    manager.expire()
    assert directory.is_dir(), "a recent exam was swept"
    for path in (directory, lock):
        os.utime(path, (0, 0))
    manager.expire()
    assert not directory.exists()
    assert not lock.exists()
    manager.close()
