"""Shared fixtures for the pulserver pytest suite."""

from pathlib import Path

import matplotlib
import numpy as np
import pypulseq as pp
import pytest

from pge import Opts

matplotlib.use("Agg")


GENERATED_SEQUENCE_FILES = [
    "gre_2d_1sl_1avg.seq",
    "gre_2d_3sl_1avg.seq",
    "epi_2d_1sl_1avg.seq",
    "epi_2d_3sl_1avg.seq",
    "fse_2d_1sl_1avg.seq",
    "fse_2d_3sl_1avg.seq",
    "bssfp_2d_1sl_1avg.seq",
    "bssfp_2d_3sl_1avg.seq",
    "gre_epi_collection_2d_1sl_1avg.seq",
    "mprage_2d_1sl_1avg.seq",
    "mprage_2d_3sl_1avg.seq",
    "mprage_nav_2d_1sl_1avg.seq",
    "mprage_nav_2d_3sl_1avg.seq",
    # Keep noncart userotext0 and skip userotext1 as requested.
    "mprage_noncart_3d_1sl_1avg_userotext0.seq",
    "mprage_noncart_3d_3sl_1avg_userotext0.seq",
]

VALIDATE_PASS_SEQUENCE_FILES = GENERATED_SEQUENCE_FILES.copy()

REPRESENTATIVE_SEQUENCE_FILES = [
    "gre_2d_1sl_1avg.seq",
    "epi_2d_1sl_1avg.seq",
    "fse_2d_1sl_1avg.seq",
    "bssfp_2d_1sl_1avg.seq",
    "gre_epi_collection_2d_1sl_1avg.seq",
    "mprage_2d_1sl_1avg.seq",
    "mprage_nav_2d_1sl_1avg.seq",
    "mprage_noncart_3d_1sl_1avg_userotext0.seq",
]


@pytest.fixture(scope="session")
def expected_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "utils" / "expected"


@pytest.fixture(scope="session")
def generated_seq_files() -> list[str]:
    return GENERATED_SEQUENCE_FILES.copy()


@pytest.fixture(scope="session")
def representative_generated_seq_files() -> list[str]:
    return REPRESENTATIVE_SEQUENCE_FILES.copy()


def _id_seq_name(name: str) -> str:
    return name.replace(".seq", "")


@pytest.fixture(params=GENERATED_SEQUENCE_FILES, ids=_id_seq_name)
def generated_seq_path(expected_data_dir: Path, request) -> Path:
    return expected_data_dir / request.param


@pytest.fixture(params=VALIDATE_PASS_SEQUENCE_FILES, ids=_id_seq_name)
def validate_pass_seq_path(expected_data_dir: Path, request) -> Path:
    return expected_data_dir / request.param


@pytest.fixture(params=[1, 3], ids=lambda n: f"navg_{n}")
def num_averages(request) -> int:
    return request.param


@pytest.fixture(params=REPRESENTATIVE_SEQUENCE_FILES, ids=_id_seq_name)
def representative_generated_seq_path(expected_data_dir: Path, request) -> Path:
    return expected_data_dir / request.param


@pytest.fixture(scope="session")
def relaxed_check_opts() -> Opts:
    """System Opts whose limits all fixture sequences satisfy.

    Uses ``max_grad=50 mT/m`` and ``max_slew=350 T/m/s`` (well above the
    effective peak values in the fixture files, accounting for the
    vectorial safety factor applied by the C-backend) and raster times
    matching the MATLAB ``make_system()`` used in
    ``generate_test_sequences.m``.
    """
    return Opts(
        max_grad=50.0,  # mT/m  — fixture peak ~35 mT/m
        max_slew=350.0,  # T/m/s — effective fixture peak ~270 T/m/s after sqrt(3) factor
        B0=3.0,
        grad_raster_time=20e-6,  # matches make_system() in generate_test_sequences.m
        block_duration_raster=20e-6,  # matches make_system()
    )


@pytest.fixture
def simple_gre_seq():
    """Build a simple 2D GRE sequence with pypulseq."""
    sys = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )
    seq = pp.Sequence(system=sys)

    flip = 15 * np.pi / 180
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip,
        duration=3e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        system=sys,
        return_gz=True,
    )

    gx = pp.make_trapezoid(
        channel="x", flat_area=128 / 0.25, flat_time=3.2e-3, system=sys
    )
    adc = pp.make_adc(
        num_samples=128, duration=gx.flat_time, delay=gx.rise_time, system=sys
    )
    gx_pre = pp.make_trapezoid(
        channel="x", area=-gx.area / 2, duration=1e-3, system=sys
    )
    gz_reph = pp.make_trapezoid(
        channel="z", area=-gz.area / 2, duration=1e-3, system=sys
    )

    n_pe = 16
    pe_areas = np.linspace(-0.5, 0.5, n_pe) * 128 / 0.25

    for i in range(n_pe):
        gy_pre = pp.make_trapezoid(
            channel="y", area=pe_areas[i], duration=1e-3, system=sys
        )
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(gx, adc)

    return seq
