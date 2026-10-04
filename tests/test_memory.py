import pytest
from numpy._core import multiarray

from pulserver._memory import without_huge_page_advice


@pytest.fixture
def advising():
    before = multiarray._get_madvise_hugepage()
    multiarray._set_madvise_hugepage(True)
    yield
    multiarray._set_madvise_hugepage(before)


@pytest.mark.parametrize("environment", [None, "1"])
def test_the_services_stop_numpy_advising_huge_pages_unless_the_environment_chooses(
    monkeypatch, advising, environment
):
    if environment is None:
        monkeypatch.delenv("NUMPY_MADVISE_HUGEPAGE", raising=False)
    else:
        monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", environment)

    without_huge_page_advice()

    assert multiarray._get_madvise_hugepage() is (environment == "1")
