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
# enough interleaves that its image is not limited by undersampling. The zero
# echo time sequence is read at a bandwidth at which the dead-time gap after
# each pulse leaves no sample at the centre of k-space unacquired.
SMALL = {"nx": 32, "ny": 32, "fov": 220.0, "phase_fov": 220.0}
NONCARTESIAN = {"ny": None, "phase_fov": None}
SMALLER = {
    "bssfp2d": {"bandwidth": 25e3},
    "bssfp3d": {"nslices": 8},
    "gre_propeller2d": NONCARTESIAN,
    "gre_radial2d": NONCARTESIAN,
    "gre_spiral2d": {**NONCARTESIAN, "num_shots": 32},
    "se_epi_propeller2d": {**NONCARTESIAN, "bandwidth": 50e3},
    "se_propeller2d": NONCARTESIAN,
    "se_radial2d": NONCARTESIAN,
    "se_spiral2d": NONCARTESIAN,
    "epi2d": {"bandwidth": 50e3},
    "epi3d": {"nslices": 8},
    "fse3d": {"nslices": 8},
    "gre3d": {"nslices": 8},
    "gre_multiecho3d": {"nslices": 8, "num_echoes": 2},
    "se3d": {"nslices": 8},
    "gre_stack_of_blades3d": {**NONCARTESIAN, "nslices": 8},
    "gre_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "gre_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
    "mprage3d": {"nslices": 8},
    "se_stack_of_blades3d": {**NONCARTESIAN, "nslices": 8},
    "se_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "se_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
    "mprage_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "mprage_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
    "zte3d": {**NONCARTESIAN, "bandwidth": 25e3},
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
    pytest.importorskip("bartorch")
    console = _console(tmp_path)

    prescribed = {**SMALL, **SMALLER.get(name, {})}
    images = _images(console, name, prescribed)

    assert images
    if name.endswith("3d"):
        # The vials lie in the partition at the slab's centre, of each echo or
        # volume; a sequence without a slices entry images one volume.
        partitions = prescribed.get("nslices", len(images))
        assert len(images) % partitions == 0
        images = images[partitions // 2 :: partitions]
    for image in images:
        assert _correlation(reference, image.astype(float)) > AGREEMENTS.get(
            name, AGREEMENT
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
    console = _console(tmp_path, recon=[RECON_PLUGINS])

    traced = _images(console, "gre2d", SMALL, recon="gre2d")
    counted = _images(console, "gre2d", SMALL, recon="exam")

    assert loaded == ["gre2d", "exam"]
    assert len(list(DesignStore(tmp_path / "designs"))) == 1
    assert len(traced) == len(counted) == 1


def test_no_shipped_reconstruction_shares_a_name_with_a_shipped_sequence():
    assert not {p.stem for p in RECONSTRUCTIONS.glob("*.py")} & set(SHIPPED)


def test_pics_images_the_vials_from_half_the_phase_encodes(tmp_path, reference):
    pytest.importorskip("bartorch")

    (image,) = _images(_console(tmp_path), "gre2d", {**SMALL, "Ry": 2})

    assert _correlation(reference, image.astype(float)) > AGREEMENT


@pytest.mark.parametrize("name", ["gre2d", "gre_radial2d"])
def test_a_multi_slice_2d_scan_returns_one_image_per_slice_along_the_slice_normal(
    tmp_path, name
):
    pytest.importorskip("bartorch")
    console = _console(tmp_path)
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    prescribed = {
        **SMALL,
        **SMALLER.get(name, {}),
        **dict(zip(FOV_ROTATION, rotation.ravel().tolist(), strict=True)),
        "nslices": 3,
        "slice_thickness": 4.0,
    }
    reply = console.design("generate", name, _prescription(**prescribed))
    assert "design" in reply, reply
    console.exam("vials")
    messages = []
    status = console.scan(
        reply["design"],
        rotation=rotation,
        centre_mm=(0.0, 0.0, 10.0),
        emit=messages.append,
    )
    assert status == 0, messages[-1:]
    images = [
        pydicom.dcmread(io.BytesIO(base64.b64decode(m["dicom"])))
        for m in messages
        if "dicom" in m
    ]

    assert len(images) == 3
    for image in images:
        assert image.pixel_array.ndim == 2
    row, column = np.reshape(images[0].ImageOrientationPatient, (2, 3))
    normal = np.cross(row, column)
    heights = sorted(
        float(np.dot(image.ImagePositionPatient, normal)) for image in images
    )
    np.testing.assert_allclose(np.diff(heights), 4.0, atol=1e-3)
    assert np.mean(heights) == pytest.approx(np.dot((0.0, 0.0, 10.0), normal), abs=1e-3)
