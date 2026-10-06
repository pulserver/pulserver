"""The virtual scanner over every shipped sequence: played as designed, and scanned where prescribed."""

import numpy as np
import pypulseqpp as pp
import pytest
from _analytic import acquire, trajectory
from _virtual import OFF_RESONANCE_HZ, OFFSET, ORIENTATIONS, phantom, posed, precession
from _zoo import SMALL, designed
from pypulseqpp import sequences

from pulserver import ir
from pulserver.proxy import SequenceTable

# A small fraction of the k-space spacing of every sequence here.
K_TOLERANCE = 1e-3


@pytest.fixture(scope="module", params=sorted(SMALL))
def design(request, tmp_path_factory):
    """The design as its file holds it, which keeps six significant digits of an amplitude."""
    name = request.param
    path = tmp_path_factory.mktemp(name) / "scan.seq"
    sequences.write(path, designed(name))
    sequence = pp.Sequence()
    sequence.read(str(path))
    return name, sequence, path


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_every_shipped_sequence_plays_its_design_turned_as_it_is_checked(
    design, rotation
):
    _, sequence, path = design
    ir.convert(path, pp.Opts())
    played = np.concatenate(trajectory(path, rotation=rotation), axis=1)
    turned = pp.TransformFOV(rotation=rotation).apply_to_sequence(sequence)
    np.testing.assert_allclose(played, turned.calculate_kspace()[0], atol=K_TOLERANCE)


def _centred_residual(sequence, acquired, table, position=None):
    """The samples, corrected by the proxy, against the object at the isocentre."""
    for index, samples in enumerate(acquired):
        modulation = table.readout_phase_modulation(index, position)
        if modulation is not None:
            acquired[index] = samples * np.exp(1j * modulation)
    ideal = phantom(coils=1).kspace(sequence.calculate_kspace()[0])
    ideal = np.split(ideal, np.cumsum([a.shape[1] for a in acquired])[:-1], axis=1)
    excited = [i for i, samples in enumerate(acquired) if np.abs(samples).any()]
    a = np.concatenate([acquired[i] for i in excited], axis=1)
    b = np.concatenate([ideal[i] for i in excited], axis=1)
    factor = np.vdot(b, a) / np.vdot(b, b)
    return np.linalg.norm(a - factor * b) / np.linalg.norm(b)


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_every_shipped_sequence_scans_an_object_posed_as_prescribed_as_at_the_isocentre(
    design, rotation
):
    _, sequence, path = design
    ir.convert(path, pp.Opts(), fov_offset=OFFSET)
    acquired = acquire(path, posed(rotation, coils=1), rotation=rotation)
    table = SequenceTable.read(path, fov_offset_m=OFFSET)
    assert _centred_residual(sequence, acquired, table) < 1e-3


def test_an_object_moved_from_the_converted_shift_is_centred_by_the_proxy_alone(design):
    _, sequence, path = design
    ir.convert(path, pp.Opts())
    acquired = acquire(path, posed(np.eye(3), coils=1))
    table = SequenceTable.read(path)
    assert _centred_residual(sequence, acquired, table, OFFSET) < 1e-3


def test_every_shipped_sequence_accrues_off_resonance_as_its_design_times_it(design):
    _, sequence, path = design
    ir.convert(path, pp.Opts())
    tissue = phantom(coils=1)
    on = np.concatenate(acquire(path, tissue), axis=1)
    off = np.concatenate(
        acquire(path, tissue, off_resonance_hz=OFF_RESONANCE_HZ), axis=1
    )
    accrued = np.nan_to_num(precession(sequence))
    expected = on * np.exp(-2j * np.pi * OFF_RESONANCE_HZ * accrued)
    assert np.linalg.norm(off - expected) / np.linalg.norm(on) < 1e-4
