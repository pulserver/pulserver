"""The published site: a directory per version, stable, the root and the switcher's list."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "publish_docs.py"
URL = "https://example.org/docs"


@pytest.fixture(scope="module")
def publisher():
    spec = importlib.util.spec_from_file_location("publish_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def built(tmp_path, marker):
    tree = tmp_path / f"build-{marker}"
    tree.mkdir()
    (tree / "index.html").write_text(marker)
    return tree


def root_target(site):
    return (site / "index.html").read_text().split("url=")[1].split("/")[0]


def test_main_alone_is_latest_and_the_root_goes_there(publisher, tmp_path):
    site = tmp_path / "site"
    publisher.publish(built(tmp_path, "main"), "latest", site, URL)

    assert root_target(site) == "latest"
    assert json.loads((site / "versions.json").read_text()) == [
        {"name": "latest", "version": "latest", "url": f"{URL}/latest/"}
    ]
    assert (site / ".nojekyll").exists()


def test_the_newest_release_is_stable_and_preferred(publisher, tmp_path):
    site = tmp_path / "site"
    publisher.publish(built(tmp_path, "main"), "latest", site, URL)
    publisher.publish(built(tmp_path, "one"), "v0.0.1", site, URL)
    entries = publisher.publish(built(tmp_path, "two"), "v0.1.0", site, URL)

    assert (site / "stable" / "index.html").read_text() == "two"
    assert root_target(site) == "stable"
    assert entries == [
        {
            "name": "v0.1.0 (stable)",
            "version": "v0.1.0",
            "url": f"{URL}/stable/",
            "preferred": True,
        },
        {"name": "v0.0.1", "version": "v0.0.1", "url": f"{URL}/v0.0.1/"},
        {"name": "latest", "version": "latest", "url": f"{URL}/latest/"},
    ]


def test_a_patch_tagged_after_a_later_release_does_not_take_stable_back(
    publisher, tmp_path
):
    site = tmp_path / "site"
    publisher.publish(built(tmp_path, "two"), "v0.1.0", site, URL)
    publisher.publish(built(tmp_path, "patch"), "v0.0.2", site, URL)

    assert (site / "stable" / "index.html").read_text() == "two"
    assert (site / "v0.0.2" / "index.html").read_text() == "patch"


def test_a_version_that_is_neither_latest_nor_a_tag_is_refused(publisher, tmp_path):
    with pytest.raises(SystemExit, match="neither"):
        publisher.publish(built(tmp_path, "x"), "stable", tmp_path / "site", URL)


def test_the_site_keeps_only_the_newest_releases_beside_latest_and_stable(
    publisher, tmp_path
):
    """The branch is never pruned, and every version is a whole documentation tree."""
    site = tmp_path / "site"
    for tag in ("v0.1.0", "v0.2.0", "v0.3.0", "v0.4.0", "v0.5.0"):
        publisher.publish(built(tmp_path, tag), tag, site, URL, keep=3)
    publisher.publish(built(tmp_path, "main"), "latest", site, URL, keep=3)

    assert sorted(d.name for d in site.iterdir() if d.is_dir()) == [
        "latest",
        "stable",
        "v0.3.0",
        "v0.4.0",
        "v0.5.0",
    ]


def test_a_release_the_site_no_longer_holds_leaves_the_switcher(publisher, tmp_path):
    site = tmp_path / "site"
    for tag in ("v0.1.0", "v0.2.0", "v0.3.0"):
        entries = publisher.publish(built(tmp_path, tag), tag, site, URL, keep=2)

    assert [entry["version"] for entry in entries] == ["v0.3.0", "v0.2.0"]


def test_stable_survives_a_prune_that_removes_its_own_tag_directory(
    publisher, tmp_path
):
    """stable is a copy of the newest release, not one of the kept directories."""
    site = tmp_path / "site"
    for tag in ("v0.1.0", "v0.2.0"):
        publisher.publish(built(tmp_path, tag), tag, site, URL, keep=0)

    assert (site / "stable" / "index.html").read_text() == "v0.2.0"
    assert not any(d.name.startswith("v") for d in site.iterdir() if d.is_dir())
