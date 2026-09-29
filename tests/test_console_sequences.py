"""The console image's sequences: each designs its default protocol under the console's limits, and images the vials."""

import base64
import io
from pathlib import Path

import numpy as np
import pydicom
import pytest
from _host import PLUGINS, value_block

from pulserver.design import load_plugin
from pulserver.protocol import FOV_OFFSET, FOV_ROTATION
from pulserver.virtual._console import Console

ROOT = Path(__file__).parents[1]
CONSOLE_PLUGINS = ROOT / "docker" / "console" / "plugins"
CONSOLE_RECON = ROOT / "docker" / "console" / "recon"
CONSOLE_LIMITS = ROOT / "docker" / "limits" / "limits.txt"
RECON_PLUGINS = Path(__file__).parent / "recon_plugins"
# The image copies the gradient echo the other tests design beside its own.
SHIPPED = sorted(["gre2d", *(path.stem for path in CONSOLE_PLUGINS.glob("*.py"))])

# A matrix small enough to scan here, at the reference's field of view. The
# balanced SSFP is read at a bandwidth whose readout outlasts the half of its
# excitation that follows the pulse centre.
SMALL = {"nx": 32, "ny": 32, "fov": 220.0, "phase_fov": 220.0}
SMALLER = {
    "bssfp2d": {"bandwidth": 25e3},
    "gre_radial2d": {"ny": None, "phase_fov": None},
    "gre_spiral2d": {"ny": None, "phase_fov": None},
}
# Normalised correlation with the Cartesian gradient echo's image of the same
# vials: below it on a contrast of its own, a regularised NUFFT's residual
# streaks, or both.
AGREEMENT = 0.7


def _prescription(**values):
    return value_block(
        {
            **dict.fromkeys(FOV_OFFSET, 0.0),
            **dict(zip(FOV_ROTATION, np.eye(3).ravel().tolist(), strict=True)),
            **{name: value for name, value in values.items() if value is not None},
        }
    )


def _console(tmp_path, recon):
    return Console(
        plugins=[CONSOLE_PLUGINS, PLUGINS],
        limits=CONSOLE_LIMITS.read_text(),
        store=tmp_path / "designs",
        spacing=2e-3,
        recon_plugins=recon,
    )


def _images(console, name, values):
    reply = console.design("generate", name, _prescription(**values))
    assert "design" in reply, reply
    console.exam("vials")
    messages = []
    status = console.scan(
        reply["design"],
        rotation=np.eye(3),
        centre_mm=(0.0, 0.0, 0.0),
        emit=messages.append,
    )
    assert status == 0, messages[-1:]
    return [
        pydicom.dcmread(io.BytesIO(base64.b64decode(m["dicom"]))).pixel_array
        for m in messages
        if "dicom" in m
    ]


def _correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The Cartesian gradient echo's image of the vials."""
    console = _console(tmp_path_factory.mktemp("reference"), [RECON_PLUGINS])
    (image,) = _images(console, "gre2d", SMALL)
    return image.astype(float)


@pytest.mark.parametrize("name", SHIPPED)
def test_a_console_sequence_designs_its_default_protocol_under_the_console_limits(
    tmp_path, name
):
    reply = _console(tmp_path, [CONSOLE_RECON]).design(
        "generate", name, _prescription()
    )

    assert "design" in reply, reply


@pytest.mark.parametrize("name", SHIPPED)
def test_a_console_sequence_images_the_vials_where_the_cartesian_gradient_echo_does(
    tmp_path, name, reference
):
    plugin = CONSOLE_PLUGINS / f"{name}.py"
    if plugin.exists() and load_plugin(plugin).recon == "nufft":
        pytest.importorskip("bartorch")
    console = _console(tmp_path, [CONSOLE_RECON])

    images = _images(console, name, {**SMALL, **SMALLER.get(name, {})})

    assert images
    for image in images:
        assert _correlation(reference, image.astype(float)) > AGREEMENT


def test_the_cartesian_reconstruction_of_a_gradient_echo_is_the_simple_fft(tmp_path):
    """Lines placed by their counters are the lines in arrival order when they arrive in order."""
    cartesian = tmp_path / "cartesian"
    cartesian.mkdir()
    (cartesian / "gre2d.py").write_text(
        "from pulserver.recon.handlers.cartesian import PLUGIN  # noqa: F401\n"
    )

    placed = _images(_console(tmp_path, [cartesian]), "gre2d", SMALL)
    arrived = _images(_console(tmp_path, [RECON_PLUGINS]), "gre2d", SMALL)

    assert len(placed) == len(arrived) == 1
    np.testing.assert_array_equal(placed[0], arrived[0])
