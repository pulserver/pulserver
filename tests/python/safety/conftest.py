"""Fixtures for the C safety-engine tests.

These exercise ``pulseg_safety.c`` through the pybind11 bindings directly.
There is no Python wrapper between the test and the engine any more: the
verdict a scanner reaches at predownload is the verdict asserted here.
"""

from pathlib import Path

import pytest
from pulserver._ext.pulseg import _PulseqCollection

EXPECTED = Path(__file__).resolve().parents[2] / "utils" / "expected"
CORPUS = Path(__file__).resolve().parents[1] / "fixtures"


def build_collection(seq_path, system, *, num_averages: int = 1) -> _PulseqCollection:
    """Load a ``.seq`` file into the C collection the safety engine runs on.

    Loads from raw bytes rather than by path: the path constructor picks up a
    ``.pge`` cache beside the file when one exists, which would silently
    override the requested ``num_averages``.
    """
    return _PulseqCollection(
        [Path(seq_path).read_bytes()],
        float(system.gamma),
        float(system.B0),
        float(system.max_grad),
        float(system.max_slew),
        float(system.rf_raster_time),
        float(system.grad_raster_time),
        float(system.adc_raster_time),
        float(system.block_duration_raster),
        True,
        num_averages,
    )


@pytest.fixture(scope="session")
def expected_data_dir() -> Path:
    return EXPECTED
