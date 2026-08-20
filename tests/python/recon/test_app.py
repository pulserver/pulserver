"""Tests for the public reconstruction-application contract."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pulserver import (
    AcquisitionBucket,
    ExamCache,
    ReconPlugin,
    ReconContext,
    ReconResult,
)
from pulserver.recon._mrd.exam import ExamCacheManager, resolve_exam_id


class SumRecon(ReconPlugin):
    def startup(self, context):
        del context
        self.lines: list[np.ndarray] = []

    def receive(self, acquisition, context):
        self.lines.append(np.asarray(acquisition.data))
        return super().receive(acquisition, context)

    def recon(self, branch, context):
        del branch
        scale = context.exam.get_or_create("scale", lambda: 2.0)
        return ReconResult(np.stack(self.lines).sum(axis=1) * scale)


def _header(exam_id: str | None = None, study_uid: str | None = None):
    parameters = []
    if exam_id is not None:
        parameters.append(SimpleNamespace(name="ExamID", value=exam_id))
    return SimpleNamespace(
        userParameters=SimpleNamespace(
            userParameterLong=[],
            userParameterDouble=[],
            userParameterString=parameters,
            userParameterBase64=[],
        ),
        studyInformation=SimpleNamespace(
            studyInstanceUID=study_uid,
            studyID=None,
        ),
    )


def test_plugin_runs_unchanged_offline():
    bucket = AcquisitionBucket.from_arrays(
        np.ones((4, 3, 8), dtype=np.complex64),
        labels={"slice": [0, 0, 1, 1]},
    )
    context = ReconContext.offline(exam_id="offline-test")

    result = SumRecon(buffered=False)(bucket, context)

    assert isinstance(result, ReconResult)
    np.testing.assert_array_equal(result.data, np.full((4, 8), 6.0))
    np.testing.assert_array_equal(bucket.labels("slice"), [0, 0, 1, 1])
    assert bucket.headers[0].idx.slice == 0


def test_bucket_preserves_ragged_acquisitions():
    first = SimpleNamespace(
        data=np.zeros((2, 8)), traj=np.zeros((8, 2)), idx=SimpleNamespace()
    )
    second = SimpleNamespace(
        data=np.zeros((2, 12)), traj=np.zeros((12, 2)), idx=SimpleNamespace()
    )
    bucket = AcquisitionBucket(data=(first, second))

    assert isinstance(bucket.kspace(), tuple)
    assert isinstance(bucket.trajectory(), tuple)


def test_exam_cache_factory_runs_once_across_threads():
    cache = ExamCache("exam")
    calls = 0
    barrier = threading.Barrier(8)
    results = []

    def worker() -> None:
        nonlocal calls
        barrier.wait()

        def factory():
            nonlocal calls
            calls += 1
            return object()

        results.append(cache.get_or_create("maps", factory))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert all(result is results[0] for result in results)


def test_exam_cache_disposes_replaced_and_retired_values():
    disposed = []
    cache = ExamCache("exam")
    first = object()
    second = object()
    cache.set("maps", first, cleanup=disposed.append)
    cache.set("maps", second, cleanup=disposed.append)
    cache.close()

    assert disposed == [first, second]
    with pytest.raises(RuntimeError, match="retired"):
        cache["maps"]


def test_exam_generation_is_shared_then_retired_safely():
    manager = ExamCacheManager()
    first_header = _header("100")
    next_header = _header("101")

    with manager.lease(first_header) as first:
        first["maps"] = object()
        with manager.lease(first_header) as shared:
            assert shared is first
        with manager.lease(next_header) as second:
            assert second is not first
            assert not first.closed
            assert manager.current_exam_id == ("exam", "101")
        assert not second.closed
    assert first.closed
    assert not second.closed
    manager.close()
    assert second.closed


def test_exam_id_prefers_vendor_exam_over_standard_study():
    assert resolve_exam_id(_header("100", "1.2.3")) == ("exam", "100")
    assert resolve_exam_id(_header(None, "1.2.3")) == (
        "study-instance",
        "1.2.3",
    )
