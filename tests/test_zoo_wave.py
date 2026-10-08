"""Wave-CAIPI scans of a shipped 3D Cartesian sequence, played by the virtual scanner and reconstructed by its paired reconstruction."""

import ismrmrd
import ismrmrd.xsd
import numpy as np
import pytest
from _analytic import acquire
from _host import generate
from conftest import play

from pulserver import _plugins, virtual
from pulserver._zoo import ZOO_PAIRS
from pulserver.host import DesignStore
from pulserver.proxy._designs import DesignCache
from pulserver.proxy._enrich import enrich_header
from pulserver.proxy._proxy import _enriched
from pulserver.recon.handlers.pics import PicsRecon

pytest.importorskip("bartorch")

#: Undersampled by two along both encodes, with a readout long enough that
#: the wave keeps its amplitude below the slew rate; 16 locations of 8 mm make
#: a 128 mm slab.
PROTOCOL = {
    "nx": 32,
    "ny": 32,
    "nslices": 16,
    "slice_thickness": 8.0,
    "Ry": 2,
    "Rz": 2,
    "bandwidth": 20000,
}

#: The wave spreads a voxel over several lines of this small matrix, and the
#: point-spread function holds the phase at the voxel centres alone, so the
#: thin ellipses' edges reconstruct a few per cent from the wave-free scan's.
TOLERANCE = 8e-2


class Images(PicsRecon):
    def __init__(self):
        super().__init__()
        self.results = []

    def recon(self, context, branch, data):
        result = super().recon(context, branch, data)
        self.results.append(result)
        return result


def _slab():
    """Ellipses stacked along z inside the central half of the slab, seen by four coils."""
    ellipses = [
        virtual.Ellipse((0.0, 0.0, z), (0.07, 0.05), 0.3, 1.0 - 4.0 * abs(z))
        for z in np.linspace(-0.03, 0.03, 7)
    ]
    ellipses.append(virtual.Ellipse((0.03, 0.02, 0.0), (0.02, 0.015), 0.0, 0.5))
    return virtual.Phantom(ellipses, coils=4)


def _reconstructed(directory, wave, device):
    """The image the pics reconstruction makes of a gre3d scan of the slab, the wave on or off."""
    plugins = directory / "plugins"
    plugins.mkdir()
    source = (_plugins.SEQUENCES / "gre3d.py").read_text()
    if wave:
        assert "\nWAVE = False\n" in source
        source = source.replace("\nWAVE = False\n", "\nWAVE = True\n")
    (plugins / "gre3d.py").write_text(source)
    store = DesignStore(directory / "designs")
    design = generate(store, "gre3d", PROTOCOL, plugins=plugins)
    acquired = acquire(store.directory(design) / "sequence.seq", _slab())
    virtual.record(directory / "raw.h5", design, acquired)

    dataset = ismrmrd.Dataset(str(directory / "raw.h5"), "dataset")
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    received = [
        dataset.read_acquisition(i) for i in range(dataset.number_of_acquisitions())
    ]
    dataset.close()
    cache = DesignCache(directory / "designs").resolve(header)
    enrich_header(header, cache.table)
    plugin = Images()
    play(plugin, header, list(_enriched(iter(received), cache)), device=device)
    (image,) = [result.data for result in plugin.results if result is not None]
    return np.asarray(image)


def test_a_wave_caipi_gre3d_reconstructs_to_the_image_of_the_wave_free_scan(
    tmp_path, device
):
    assert ZOO_PAIRS["gre3d"] == "pics"
    (tmp_path / "wave").mkdir()
    (tmp_path / "plain").mkdir()

    waved = _reconstructed(tmp_path / "wave", True, device)
    plain = _reconstructed(tmp_path / "plain", False, device)

    assert waved.shape == plain.shape == (16, 32, 32)
    scale = (waved * plain).sum() / (waved * waved).sum()
    assert np.linalg.norm(scale * waved - plain) / np.linalg.norm(plain) < TOLERANCE
