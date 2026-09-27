"""The ``pulserver scan`` command: a design scanned on the virtual scanner, headless."""

import json
import shutil
import threading
from pathlib import Path

import ismrmrd
import ismrmrd.hdf5
import numpy as np
import pypulseqpp as pp
import pytest
from _host import ANY_ORIENTATION, FIXTURE_LIMITS, FIXTURES, PLUGINS
from scipy.io import wavfile

from pulserver import _cli, virtual
from pulserver.host import DesignStore
from pulserver.host._blocks import format_limits
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END
from pulserver.proxy import ReconProxy
from pulserver.virtual._command import (
    ORIENTATIONS,
    default_phantom,
    main,
    read_phantom,
)

RECON_PLUGINS = Path(__file__).parent / "recon_plugins"
DEADLINE = 180.0


def _limits(path, limits):
    path.write_text(format_limits(limits))
    return path


def _reply(capsys):
    """The design the command's reply names."""
    return capsys.readouterr().out.split()[1]


def test_a_headless_scan_records_the_series_the_virtual_scanner_acquires(
    tmp_path, capsys
):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    store, mrd, sound = tmp_path / "designs", tmp_path / "raw.h5", tmp_path / "scan.wav"
    status = main(
        [
            "--seq",
            str(tmp_path / "gre_2d_3sl.seq"),
            "--limits",
            str(_limits(tmp_path / "limits.txt", FIXTURE_LIMITS)),
            "--store",
            str(store),
            "--orientation",
            "coronal",
            "--center",
            "10",
            "-5",
            "3",
            "--mrd",
            str(mrd),
            "--sound",
            str(sound),
        ]
    )
    assert status == 0
    sequence = DesignStore(store).directory(_reply(capsys)) / "sequence.seq"
    rotation = ORIENTATIONS["coronal"]
    tissue = default_phantom(4).isochromats(1e-3, field_t=FIXTURE_LIMITS["B0"])
    expected = virtual.simulate(sequence, tissue, rotation=rotation)
    dataset = ismrmrd.hdf5.Dataset(str(mrd), "dataset", create_if_needed=False)
    try:
        acquisitions = [
            dataset.read_acquisition(i) for i in range(dataset.number_of_acquisitions())
        ]
    finally:
        dataset.close()
    assert len(acquisitions) == len(expected)
    for index, (acquisition, readout) in enumerate(
        zip(acquisitions, expected, strict=True)
    ):
        np.testing.assert_array_equal(acquisition.data, readout)
        np.testing.assert_allclose(acquisition.position, (10.0, -5.0, 3.0))
        np.testing.assert_allclose(acquisition.slice_dir, rotation[:, 2])
        assert acquisition.scan_counter == index + 1
        assert acquisition.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT) == (
            index == len(expected) - 1
        )
    rate, audio = wavfile.read(sound)
    designed = pp.Sequence()
    designed.read(str(sequence))
    assert rate == virtual.SAMPLE_RATE
    assert audio.shape == (designed.sound().shape[1], 2)
    assert audio.dtype == np.int16


@pytest.fixture
def proxy(tmp_path):
    running = ReconProxy(tmp_path / "designs", RECON_PLUGINS, slots=1)
    running.bind(0)
    thread = threading.Thread(target=running.serve, daemon=True)
    thread.start()
    yield running
    running.close()
    thread.join(timeout=DEADLINE)


def test_a_headless_scan_streams_a_generated_design_to_a_proxy_and_keeps_its_image(
    proxy, tmp_path, capsys
):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text(f"{PROTOCOL_BEGIN}\nTE: 5000\nnx: 32\nny: 32\n{PROTOCOL_END}\n")
    status = _cli.main(
        [
            "scan",
            "--plugins",
            str(PLUGINS),
            "--plugin",
            "gre2d",
            "--protocol",
            str(protocol),
            "--limits",
            str(_limits(tmp_path / "limits.txt", ANY_ORIENTATION)),
            "--store",
            str(tmp_path / "designs"),
            "--orientation",
            "sagittal",
            "--center",
            "0",
            "12",
            "-8",
            "--spacing",
            "2",
            "--recon",
            f"127.0.0.1:{proxy.port}",
            "--output",
            str(tmp_path / "images"),
        ]
    )
    assert status == 0
    assert capsys.readouterr().out.startswith("GENERATED ")
    dataset = ismrmrd.hdf5.Dataset(
        str(tmp_path / "images" / "images.h5"), "dataset", create_if_needed=False
    )
    try:
        image = dataset.read_image("images", 0)
    finally:
        dataset.close()
    assert np.abs(image.data).max() > 0.0
    rotation = ORIENTATIONS["sagittal"]
    np.testing.assert_allclose(image.position, (0.0, 12.0, -8.0))
    for direction, axis in zip(
        (image.read_dir, image.phase_dir, image.slice_dir), rotation.T, strict=True
    ):
        np.testing.assert_allclose(direction, axis, atol=1e-6)


def test_a_design_the_calls_refuse_ends_the_scan_with_their_error(tmp_path, capsys):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    weak = {**FIXTURE_LIMITS, "max_grad": 1.0}
    status = main(
        [
            "--seq",
            str(tmp_path / "gre_2d_3sl.seq"),
            "--limits",
            str(_limits(tmp_path / "limits.txt", weak)),
            "--mrd",
            str(tmp_path / "raw.h5"),
        ]
    )
    assert status == 1
    assert capsys.readouterr().out.startswith("ERROR ")
    assert not (tmp_path / "raw.h5").exists()


@pytest.mark.parametrize("name", sorted(ORIENTATIONS))
def test_an_orientation_is_a_rotation_that_reads_encodes_and_selects_along_its_axes(
    name,
):
    rotation = ORIENTATIONS[name]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    columns = {
        "axial": ("x", "y", "z"),
        "coronal": ("x", "z", "-y"),
        "sagittal": ("y", "z", "x"),
    }
    for column, axis in zip(rotation.T, columns[name], strict=True):
        expected = np.zeros(3)
        expected["xyz".index(axis[-1])] = -1.0 if axis.startswith("-") else 1.0
        np.testing.assert_array_equal(column, expected)


def test_a_phantom_file_lists_the_fields_of_its_ellipses(tmp_path):
    path = tmp_path / "phantom.json"
    path.write_text(
        json.dumps(
            {
                "ellipses": [
                    {"centre": [0.01, 0.0, 0.0], "semi_axes": [0.05, 0.04], "t2": 0.07},
                    {
                        "centre": [0.0, 0.02, 0.0],
                        "semi_axes": [0.01, 0.01],
                        "shift_ppm": -3.45,
                    },
                ]
            }
        )
    )
    tissue = read_phantom(path, coils=3)
    assert tissue.coils == 3
    assert tissue.ellipses == (
        virtual.Ellipse((0.01, 0.0, 0.0), (0.05, 0.04), t2=0.07),
        virtual.Ellipse((0.0, 0.02, 0.0), (0.01, 0.01), shift_ppm=-3.45),
    )
    path.write_text(json.dumps({"vials": []}))
    with pytest.raises(ValueError, match="ellipses"):
        read_phantom(path)


def test_the_command_without_a_known_subcommand_prints_its_usage(capsys):
    assert _cli.main(["simulate"]) == 2
    assert "pulserver scan" in capsys.readouterr().err
