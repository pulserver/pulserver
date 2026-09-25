"""The virtual scanner over every shipped sequence: played as designed, and scanned where prescribed."""

import numpy as np
import pypulseqpp as pp
import pytest
from _zoo import SMALL
from pypulseqpp import sequences

from pulserver import ir, virtual

OFFSET = np.array([0.02, -0.012, 0.0])
# A small fraction of the k-space spacing of every sequence here.
K_TOLERANCE = 1e-3


@pytest.fixture(scope="module", params=sorted(SMALL))
def design(request, tmp_path_factory):
    name = request.param
    sequence = getattr(sequences, name)(**SMALL[name])
    path = tmp_path_factory.mktemp(name) / "scan.seq"
    sequence.write(str(path))
    return name, sequence, path


def test_every_shipped_sequence_plays_the_trajectory_it_designs(design):
    _, sequence, path = design
    ir.convert(path, pp.Opts())
    played = np.concatenate(virtual.trajectory(path), axis=1)
    np.testing.assert_allclose(played, sequence.calculate_kspace()[0], atol=K_TOLERANCE)


def test_every_shipped_sequence_scans_an_object_where_it_is_prescribed(design):
    _, sequence, path = design
    ir.convert(path, pp.Opts(), fov_offset=OFFSET)
    phantom = virtual.Phantom(
        [
            virtual.Ellipse(tuple(OFFSET), (0.08, 0.06), 0.3),
            virtual.Ellipse(
                tuple(OFFSET + np.array([0.03, 0.02, 0.0])), (0.02, 0.015), 0.0, 0.5
            ),
        ]
    )
    acquired = virtual.acquire(path, phantom)
    k = sequence.calculate_kspace()[0]
    ideal = phantom.kspace(k) * np.exp(2j * np.pi * (OFFSET @ k))
    ideal = np.split(ideal, np.cumsum([a.shape[1] for a in acquired])[:-1], axis=1)
    excited = [i for i, samples in enumerate(acquired) if np.abs(samples).any()]
    a = np.concatenate([acquired[i] for i in excited], axis=1)
    b = np.concatenate([ideal[i] for i in excited], axis=1)
    factor = np.vdot(b, a) / np.vdot(b, b)
    assert np.linalg.norm(a - factor * b) / np.linalg.norm(b) < 1e-3
