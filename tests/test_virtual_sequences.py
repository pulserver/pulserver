"""The virtual scanner over every shipped sequence: played as designed, and scanned where prescribed."""

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import OFFSET, ORIENTATIONS, phantom, posed
from _zoo import SMALL
from pypulseqpp import sequences

from pulserver import ir, virtual

# A small fraction of the k-space spacing of every sequence here.
K_TOLERANCE = 1e-3


@pytest.fixture(scope="module", params=sorted(SMALL))
def design(request, tmp_path_factory):
    name = request.param
    sequence = getattr(sequences, name)(**SMALL[name])
    path = tmp_path_factory.mktemp(name) / "scan.seq"
    sequence.write(str(path))
    return name, sequence, path


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_every_shipped_sequence_plays_its_design_turned_as_it_is_checked(
    design, rotation
):
    _, sequence, path = design
    ir.convert(path, pp.Opts())
    played = np.concatenate(virtual.trajectory(path, rotation=rotation), axis=1)
    turned = pp.TransformFOV(rotation=rotation).apply_to_sequence(sequence)
    np.testing.assert_allclose(played, turned.calculate_kspace()[0], atol=K_TOLERANCE)


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_every_shipped_sequence_scans_an_object_posed_as_prescribed_as_at_the_isocentre(
    design, rotation
):
    _, sequence, path = design
    ir.convert(path, pp.Opts(), fov_offset=OFFSET)
    acquired = virtual.acquire(path, posed(rotation, coils=1), rotation=rotation)
    ideal = phantom(coils=1).kspace(sequence.calculate_kspace()[0])
    ideal = np.split(ideal, np.cumsum([a.shape[1] for a in acquired])[:-1], axis=1)
    excited = [i for i, samples in enumerate(acquired) if np.abs(samples).any()]
    a = np.concatenate([acquired[i] for i in excited], axis=1)
    b = np.concatenate([ideal[i] for i in excited], axis=1)
    factor = np.vdot(b, a) / np.vdot(b, b)
    assert np.linalg.norm(a - factor * b) / np.linalg.norm(b) < 1e-3
