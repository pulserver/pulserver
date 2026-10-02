"""The shipped sequences: each designs its default protocol under the console's limits, and images the vials with its shipped reconstruction."""

import base64
import io
from pathlib import Path

import numpy as np
import pydicom
import pytest
from _host import PLUGINS, value_block

from pulserver._plugins import RECONSTRUCTIONS, SEQUENCES
from pulserver._zoo import ZOO_PAIRS
from pulserver.host import DesignStore
from pulserver.protocol import FOV_OFFSET, FOV_ROTATION
from pulserver.proxy import _local
from pulserver.virtual._console import Console

ROOT = Path(__file__).parents[1]
CONSOLE_LIMITS = ROOT / "docker" / "limits" / "limits.txt"
RECON_PLUGINS = Path(__file__).parent / "recon_plugins"
SHIPPED = sorted(path.stem for path in SEQUENCES.glob("*.py"))

# A matrix small enough to scan here, at the reference's field of view. The
# balanced SSFP is read at a bandwidth whose readout outlasts the half of its
# excitation that follows the pulse centre. The gradient-echo spiral is read in
# enough interleaves that its image is not limited by undersampling.
SMALL = {"nx": 32, "ny": 32, "fov": 220.0, "phase_fov": 220.0}
NONCARTESIAN = {"ny": None, "phase_fov": None}
SMALLER = {
    "bssfp2d": {"bandwidth": 25e3},
    "gre_radial2d": NONCARTESIAN,
    "gre_spiral2d": {**NONCARTESIAN, "num_shots": 32},
    "se_radial2d": NONCARTESIAN,
    "se_spiral2d": NONCARTESIAN,
    "epi2d": {"bandwidth": 50e3},
    "gre3d": {"nslices": 8},
    "se3d": {"nslices": 8},
    "gre_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "gre_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
    "se_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "se_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
}
# Normalised correlation with the Cartesian gradient echo's image of the same
# vials: below it on a contrast of its own, a regularised NUFFT's residual
# streaks, or both.
AGREEMENT = 0.7
# EPI displaces a vial off water resonance along the phase encode.
AGREEMENTS = {"epi2d": 0.6}


def _prescription(**values):
    return value_block(
        {
            **dict.fromkeys(FOV_OFFSET, 0.0),
            **dict(zip(FOV_ROTATION, np.eye(3).ravel().tolist(), strict=True)),
            **{name: value for name, value in values.items() if value is not None},
        }
    )


def _console(tmp_path, plugins=(), recon=()):
    """A console searching ``plugins`` and ``recon`` before the shipped plugins."""
    return Console(
        plugins=[*plugins, tmp_path / "plugins"],
        limits=CONSOLE_LIMITS.read_text(),
        store=tmp_path / "designs",
        spacing=2e-3,
        recon_plugins=[*recon, tmp_path / "recon"],
    )


def _images(console, name, values, recon=None):
    """The images of a scan of ``name`` reconstructed by ``recon``, or by the plugin shipped with it."""
    reply = console.design("generate", name, _prescription(**values))
    assert "design" in reply, reply
    console.exam("vials")
    messages = []
    status = console.scan(
        reply["design"],
        rotation=np.eye(3),
        centre_mm=(0.0, 0.0, 0.0),
        emit=messages.append,
        recon=recon,
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
    console = _console(tmp_path_factory.mktemp("reference"), [PLUGINS], [RECON_PLUGINS])
    (image,) = _images(console, "gre2d", SMALL, recon="gre2d")
    return image.astype(float)


@pytest.mark.parametrize("name", SHIPPED)
def test_a_console_sequence_designs_its_default_protocol_under_the_console_limits(
    tmp_path, name
):
    reply = _console(tmp_path).design("generate", name, _prescription())

    assert "design" in reply, reply


@pytest.mark.parametrize("name", SHIPPED)
def test_a_console_sequence_images_the_vials_where_the_cartesian_gradient_echo_does(
    tmp_path, name, reference
):
    if ZOO_PAIRS[name] != "cartesian":
        pytest.importorskip("bartorch")
    console = _console(tmp_path)

    images = _images(console, name, {**SMALL, **SMALLER.get(name, {})})

    assert images
    if name.endswith("3d"):
        # The vials lie in the partition at the slab's centre.
        images = [images[len(images) // 2]]
    for image in images:
        assert _correlation(reference, image.astype(float)) > AGREEMENTS.get(
            name, AGREEMENT
        )


def test_the_cartesian_reconstruction_of_a_gradient_echo_is_the_simple_fft(tmp_path):
    """Lines placed by their counters are the lines in arrival order when they arrive in order, to within one stored integer."""
    placed = _images(_console(tmp_path), "gre2d", SMALL, recon="cartesian")
    arrived = _images(
        _console(tmp_path, [PLUGINS], [RECON_PLUGINS]), "gre2d", SMALL, recon="gre2d"
    )

    assert len(placed) == len(arrived) == 1
    np.testing.assert_allclose(
        placed[0].astype(float), arrived[0].astype(float), rtol=0, atol=1
    )


def test_every_shipped_sequence_is_paired_with_a_shipped_reconstruction():
    shipped = {path.stem for path in RECONSTRUCTIONS.glob("*.py")}

    assert set(ZOO_PAIRS) == set(SHIPPED)
    assert set(ZOO_PAIRS.values()) <= shipped


def test_one_design_reconstructs_under_two_reconstructions(tmp_path, monkeypatch):
    loaded = []
    load = _local.load_plugin
    monkeypatch.setattr(
        _local, "load_plugin", lambda path: loaded.append(path.stem) or load(path)
    )
    console = _console(tmp_path)

    cartesian = _images(console, "gre2d", SMALL, recon="cartesian")
    simple = _images(console, "gre2d", SMALL, recon="simplefft")

    assert loaded == ["cartesian", "simplefft"]
    assert len(list(DesignStore(tmp_path / "designs"))) == 1
    assert len(cartesian) == len(simple) == 1


def test_no_shipped_reconstruction_shares_a_name_with_a_shipped_sequence():
    assert not {p.stem for p in RECONSTRUCTIONS.glob("*.py")} & set(SHIPPED)


def test_pics_images_the_vials_from_half_the_phase_encodes(tmp_path, reference):
    pytest.importorskip("bartorch")

    (image,) = _images(_console(tmp_path), "gre2d", {**SMALL, "Ry": 2})

    assert _correlation(reference, image.astype(float)) > AGREEMENT
