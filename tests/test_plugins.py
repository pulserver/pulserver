"""Plugin files found by name along an ordered list of directories."""

import pytest

from pulserver import _plugins

SHIPPED = sorted(path.stem for path in _plugins.SEQUENCES.glob("*.py"))


@pytest.fixture
def path(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "shared.py").write_text("FROM = 'first'\n")
    (second / "shared.py").write_text("FROM = 'second'\n")
    (second / "own.py").write_text("")
    return first, second


def test_a_plugin_is_the_file_of_the_first_directory_that_holds_it(path):
    first, second = path

    assert _plugins.find(path, "shared") == first / "shared.py"
    assert _plugins.find(path, "own") == second / "own.py"
    assert _plugins.find(second, "shared") == second / "shared.py"


def test_the_plugin_names_are_those_of_every_directory_each_once(path):
    assert _plugins.names(path) == sorted({"own", "shared", *SHIPPED})


def test_a_shipped_plugin_is_found_after_every_directory_given(path):
    first = path[0]
    (first / "gre2d.py").write_text("")

    assert _plugins.find(first, "gre2d") == first / "gre2d.py"
    assert _plugins.find(path[1], "gre2d") == _plugins.SEQUENCES / "gre2d.py"
    assert _plugins.find(path[1], "nufft") == _plugins.RECONSTRUCTIONS / "nufft.py"


def test_the_shipped_reconstructions_are_not_listed_as_sequences(path):
    assert "nufft" not in _plugins.names(path)


def test_a_directory_that_does_not_exist_holds_no_plugins(path, tmp_path):
    absent = tmp_path / "absent"

    assert _plugins.names([absent, *path]) == sorted({"own", "shared", *SHIPPED})
    assert _plugins.find([absent, *path], "own") == path[1] / "own.py"


def test_a_link_is_a_plugin_named_after_the_link(path):
    first = path[0]
    (first / "alias.py").symlink_to("../second/own.py")
    (first / "dangling.py").symlink_to("../second/absent.py")

    assert _plugins.names(first) == sorted({"alias", "shared", *SHIPPED})
    assert _plugins.find(first, "alias").read_text() == ""


@pytest.mark.parametrize("name", ["../own", "own.py", "", "a b"])
def test_a_name_outside_the_plugin_alphabet_is_refused(path, name):
    with pytest.raises(ValueError, match="invalid plugin name"):
        _plugins.find(path, name)


def test_a_plugin_no_directory_holds_names_every_directory_searched(path):
    first, second = path

    with pytest.raises(FileNotFoundError) as refused:
        _plugins.find(path, "absent")

    assert str(refused.value) == (
        f"no plugin 'absent' in {first}, {second} or among the shipped plugins"
    )
