"""Loading a reconstruction plugin from its file."""

import pytest

from pulserver.recon import load_plugin
from pulserver.recon.handlers.pics import PicsRecon

PLUGIN_SOURCE = """
from pulserver.recon.handlers.pics import PicsRecon

PLUGIN = PicsRecon()
"""


def test_a_plugin_file_loads_as_the_instance_it_names(tmp_path):
    path = tmp_path / "demo.py"
    path.write_text(PLUGIN_SOURCE)
    assert isinstance(load_plugin(path), PicsRecon)


def test_a_file_without_a_plugin_is_refused(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="no module-level PLUGIN"):
        load_plugin(path)


def test_a_plugin_that_is_not_a_reconstruction_is_refused(tmp_path):
    path = tmp_path / "wrong.py"
    path.write_text("PLUGIN = object()\n")
    with pytest.raises(ValueError, match="no module-level PLUGIN"):
        load_plugin(path)
