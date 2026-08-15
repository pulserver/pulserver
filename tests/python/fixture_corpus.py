"""The checked-in `.seq` fixture corpus: one entry per zoo slot, plus edges.

Every fixture under ``tests/python/fixtures/`` is written by
``pulserver.pypulseq`` itself, through the builders below, at the same small
protocols the per-slot tests design. The write→read fidelity that justifies
trusting these files is asserted fixture-free in
``test_pypulseq_roundtrip.py``; the checked-in copies exist so that
file-consuming tests read stable, reviewable ground truth and so that a
writer or parser regression shows up as a diff against them.

Regenerate with ``bash scripts/regenerate_fixtures.sh`` (or run
``tests/utils/generate_fixtures.py`` directly). The output is deterministic:
running the generator twice leaves the tree unchanged.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _gre_2d():
    from pulserver.seqzoo import gre_2d

    return gre_2d.main(n_x=32, n_y=32, te=5e-3, tr=30e-3)


def _gre_2d_3sl():
    """The multi-slice scan the k-space and auto_label tests read.

    Three contiguous 5 mm slices put the stack at -5/0/+5 mm, acquired
    slice-inner in position order; 64 symmetric readout samples put the echo
    at sample 32, and 8 fully-sampled lines put the centre line at 4.
    """
    from pulserver.seqzoo import gre_2d

    return gre_2d.main(
        n_x=64,
        n_y=8,
        n_slices=3,
        slice_thickness=5e-3,
        slice_gap=0.0,
        slice_order="sequential",
        n_dummy=5,
        te=5e-3,
        tr=30e-3,
    )


def _gre_3d():
    from pulserver.seqzoo import gre_3d

    return gre_3d.main(
        n_x=16, n_y=16, n_z=8, slab_thickness=64e-3, n_acs=6, n_acs_z=4, n_dummy=2
    )


def _se_2d():
    from pulserver.seqzoo import se_2d

    return se_2d.main(n_x=32, n_y=16, te=20e-3, tr=100e-3, n_acs=6)


def _se_3d():
    from pulserver.seqzoo import se_3d

    return se_3d.main(
        n_x=16,
        n_y=16,
        n_z=8,
        slab_thickness=64e-3,
        te=20e-3,
        tr=60e-3,
        n_acs=6,
        n_acs_z=4,
    )


def _gre_multiecho_2d():
    from pulserver.seqzoo import gre_multiecho_2d

    return gre_multiecho_2d.main(
        n_x=32, n_y=8, n_echoes=3, n_acs=4, n_dummy=1, tr=40e-3
    )


def _gre_multiecho_3d():
    from pulserver.seqzoo import gre_multiecho_3d

    return gre_multiecho_3d.main(
        n_x=32,
        n_y=8,
        n_z=4,
        slab_thickness=32e-3,
        n_echoes=3,
        n_acs=4,
        n_acs_z=2,
        n_dummy=1,
        tr=30e-3,
    )


def _bssfp_2d():
    from pulserver.seqzoo import bssfp_2d

    return bssfp_2d.main(
        n_x=32, n_y=8, n_acs=4, n_dummy=2, readout_bandwidth_hz=25e3
    )


def _bssfp_3d():
    from pulserver.seqzoo import bssfp_3d

    return bssfp_3d.main(
        n_x=32,
        n_y=8,
        n_z=4,
        slab_thickness=32e-3,
        n_acs=4,
        n_acs_z=2,
        n_dummy=2,
        readout_bandwidth_hz=25e3,
    )


def _fse_2d():
    from pulserver.seqzoo import fse_2d

    return fse_2d.main(
        n_x=32,
        n_y=16,
        n_acs=6,
        etl=4,
        te=40e-3,
        tr=300e-3,
        readout_bandwidth_hz=50e3,
    )


def _fse_3d():
    from pulserver.seqzoo import fse_3d

    return fse_3d.main(
        n_x=32,
        n_y=8,
        n_z=4,
        slab_thickness=64e-3,
        n_acs=4,
        n_acs_z=2,
        etl=8,
        te=60e-3,
        tr=250e-3,
        readout_bandwidth_hz=50e3,
        readout_crusher_cycles=1.0,
        elliptical=False,
    )


def _mprage_3d():
    from pulserver.seqzoo import mprage_3d

    return mprage_3d.main(
        n_x=32,
        n_y=8,
        n_z=4,
        slab_thickness=64e-3,
        n_acs=4,
        n_acs_z=2,
        views_per_segment=8,
        ti=200e-3,
        tr_outer=600e-3,
        elliptical=False,
    )


def _gre_radial_2d():
    from pulserver.seqzoo import gre_radial_2d

    return gre_radial_2d.main(
        n_x=32, n_spokes=8, n_dummy=2, tr=20e-3, readout_bandwidth_hz=125e3
    )


def _gre_spiral_2d():
    from pulserver.seqzoo import gre_spiral_2d

    return gre_spiral_2d.main(
        n_x=32, n_arms=4, n_dummy=2, tr=20e-3, readout_bandwidth_hz=125e3
    )


def _gre_stack_of_stars_3d():
    from pulserver.seqzoo import gre_stack_of_stars_3d

    return gre_stack_of_stars_3d.main(
        n_x=32,
        n_z=4,
        slab_thickness=64e-3,
        n_spokes=4,
        n_dummy=2,
        readout_bandwidth_hz=125e3,
    )


def _gre_stack_of_spirals_3d():
    from pulserver.seqzoo import gre_stack_of_spirals_3d

    return gre_stack_of_spirals_3d.main(
        n_x=32,
        n_z=4,
        slab_thickness=64e-3,
        n_arms=4,
        n_dummy=2,
        readout_bandwidth_hz=125e3,
    )


def _se_propeller_2d():
    from pulserver.seqzoo import se_propeller_2d

    return se_propeller_2d.main(
        n_x=32,
        blade_width=4,
        n_blades=4,
        te=60e-3,
        tr=200e-3,
        readout_bandwidth_hz=125e3,
    )


def _mprage_stack_of_spirals_3d():
    from pulserver.seqzoo import mprage_stack_of_spirals_3d

    return mprage_stack_of_spirals_3d.main(
        n_x=32,
        n_z=4,
        slab_thickness=64e-3,
        n_arms=3,
        ti=200e-3,
        tr_outer=500e-3,
        readout_bandwidth_hz=125e3,
    )


def _zte_3d():
    from pulserver.seqzoo import zte_3d

    return zte_3d.main(n_x=32, n_views=16, n_shots=4, readout_bandwidth_hz=125e3)


#: name -> builder for the single-file fixtures.
CORPUS = {
    "gre_2d": _gre_2d,
    "gre_2d_3sl": _gre_2d_3sl,
    "gre_3d": _gre_3d,
    "se_2d": _se_2d,
    "se_3d": _se_3d,
    "gre_multiecho_2d": _gre_multiecho_2d,
    "gre_multiecho_3d": _gre_multiecho_3d,
    "bssfp_2d": _bssfp_2d,
    "bssfp_3d": _bssfp_3d,
    "fse_2d": _fse_2d,
    "fse_3d": _fse_3d,
    "mprage_3d": _mprage_3d,
    "gre_radial_2d": _gre_radial_2d,
    "gre_spiral_2d": _gre_spiral_2d,
    "gre_stack_of_stars_3d": _gre_stack_of_stars_3d,
    "gre_stack_of_spirals_3d": _gre_stack_of_spirals_3d,
    "se_propeller_2d": _se_propeller_2d,
    "mprage_stack_of_spirals_3d": _mprage_stack_of_spirals_3d,
    "zte_3d": _zte_3d,
}

#: Corpus entries that also get a `.bin` twin, one per binary-format feature:
#: boundary labels + partial Fourier (gre_2d), rotation extensions (radial),
#: explicit arbitrary arms (stack of spirals), echo-train labels (propeller,
#: fse), continuous gradients (zte). The binary format's LABELNAMES section
#: makes every label name self-describing, so label-heavy slots roundtrip
#: across processes like any other.
BINARY_TWINS = (
    "gre_2d",
    "fse_3d",
    "gre_radial_2d",
    "mprage_stack_of_spirals_3d",
    "se_propeller_2d",
    "zte_3d",
)


def write_epi_collections(directory: Path) -> list[Path]:
    """Write the EPI NextSequence chains and return every file they created."""
    from pulserver.seqzoo import epi_2d, epi_3d

    before = set(directory.glob("*.seq"))
    epi_2d.main(
        n_x=32,
        n_y=16,
        readout_bandwidth_hz=250e3,
        write_seq=True,
        seq_filename=str(directory / "epi_2d.seq"),
    )
    epi_3d.main(
        n_x=32,
        n_y=16,
        n_z=4,
        slab_thickness=32e-3,
        readout_bandwidth_hz=250e3,
        write_seq=True,
        seq_filename=str(directory / "epi_3d.seq"),
    )
    return sorted(set(directory.glob("*.seq")) - before)


def build_parser_edges():
    """Minimal single-feature files for the parser's edge branches."""
    import numpy as np

    import pulserver.pypulseq as pp

    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")

    adc_only = pp.Sequence(system=system)
    adc_only.add_block(
        pp.make_adc(num_samples=64, duration=640e-6, delay=100e-6, system=system)
    )

    trap_only = pp.Sequence(system=system)
    trap_only.add_block(pp.make_trapezoid(channel="x", area=100.0, system=system))

    ext_only = pp.Sequence(system=system)
    ext_only.add_block(
        pp.make_delay(1e-3), pp.make_label("LIN", "SET", 1)
    )
    ext_only.add_block(
        pp.make_delay(1e-3),
        pp.make_soft_delay(numID=0, hint="TE", offset=0.0, factor=1.0),
    )

    for seq, name in ((adc_only, "adc_only"), (trap_only, "trap_only"), (ext_only, "ext_only")):
        seq.set_definition("Name", name)
    return {"adc_only": adc_only, "trap_only": trap_only, "ext_only": ext_only}
