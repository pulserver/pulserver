"""The MRD encoding every zoo file states: its labels, as the proxy enriches them."""

import ismrmrd
import ismrmrd.xsd as xsd
import numpy as np
import pytest
from test_zoo_evaluation import (
    CHANGED,
    OPTIMIZED,
    SCANS,
    SHIPPED,
    SYSTEM,
    TOGGLED,
    WITH_BLOCHSIM,
    _chain,
    _protocol,
    zoo,  # noqa: F401
)

from pulserver._labels import MRD_COUNTERS
from pulserver.proxy._enrich import SequenceTable, enrich_acquisition, enrich_header

#: Counters with an ``encodingLimits`` entry, by label.
LIMITS = {
    "LIN": "kspace_encoding_step_1",
    "PAR": "kspace_encoding_step_2",
    "SLC": "slice",
    "ECO": "contrast",
    "AVG": "average",
    "PHS": "phase",
    "REP": "repetition",
    "SET": "set",
    "SEG": "segment",
}

#: Every shipped and toggled sequence at a small protocol, and the scans that
#: exercise slices, partitions, echoes, frames, phase cycles and shots.
CASES = [
    pytest.param(
        key,
        CHANGED[key.partition("+")[0]],
        id=f"{key}-small",
        marks=WITH_BLOCHSIM if key in OPTIMIZED else (),
    )
    for key in [*SHIPPED, *TOGGLED]
] + [scan for scan in SCANS if scan.values[1]]


def _labels(seq, count):
    """Each label's value at every readout, by name."""
    found = seq.evaluate_labels(evolution="adc")
    return {
        name: np.broadcast_to(np.asarray(values), (count,))
        for name, values in found.items()
        if np.size(values)
    }


def _enriched(path):
    table = SequenceTable.read(path)
    header = xsd.ismrmrdHeader(
        experimentalConditions=xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=297_000_000
        )
    )
    enrich_header(header, table)
    return table, header


@pytest.mark.parametrize(("name", "changes"), CASES)
def test_every_zoo_file_states_in_mrd_the_encoding_its_labels_write(
    zoo,  # noqa: F811
    tmp_path,
    name,
    changes,
):
    plugin = zoo[name]
    designed = _chain(plugin.generate(SYSTEM, _protocol(plugin, changes)))
    for index, seq in enumerate(designed):
        path = tmp_path / f"{index}.seq"
        seq.write(str(path))
        table, header = _enriched(path)
        count = len(table)
        if not count:
            continue
        labels = _labels(seq, count)
        navigator = labels.get("NAV", np.zeros(count)) != 0
        imaging = ~navigator

        for space, encoding in zip(table.spaces, header.encoding, strict=True):
            members = navigator if space.navigator else imaging
            for label, field in LIMITS.items():
                values = labels.get(label, np.zeros(count, dtype=int))[members]
                limit = getattr(encoding.encodingLimits, field)
                assert (limit.minimum, limit.maximum) == (
                    values.min(),
                    values.max(),
                ), (index, space.navigator, label)

        # Counters and flags together: a calibration line acquired again in
        # the image is told apart by its flag, as MRD marks it.
        tuples = np.stack([values[imaging] for values in labels.values()], axis=1)
        assert np.unique(tuples, axis=0).shape[0] == imaging.sum(), index

        for row in np.unique(np.linspace(0, count - 1, 64).astype(int)):
            acquisition = ismrmrd.Acquisition.from_array(
                np.zeros((1, int(table.num_samples[row])), np.complex64)
            )
            enrich_acquisition(acquisition, table, int(row))
            for label, field in MRD_COUNTERS.items():
                if label.startswith("USER"):
                    continue
                expected = labels.get(label, np.zeros(count, dtype=int))[row]
                assert getattr(acquisition.idx, field) == expected, (row, label)


def test_a_cartesian_inversion_volume_encodes_its_lines_and_partitions(
    zoo,  # noqa: F811
    tmp_path,
):
    plugin = zoo["mprage3d"]
    changes = {"nx": 128, "ny": 128, "nslices": 20, "slice_thickness": 5.0}
    seq = _chain(plugin.generate(SYSTEM, _protocol(plugin, changes)))[-1]
    seq.write(str(tmp_path / "mprage.seq"))
    _, header = _enriched(tmp_path / "mprage.seq")
    (encoding,) = header.encoding
    limits = encoding.encodingLimits
    matrix = encoding.reconSpace.matrixSize

    assert (matrix.x, matrix.y, matrix.z) == (128, 128, 20)
    assert (
        limits.kspace_encoding_step_1.minimum,
        limits.kspace_encoding_step_1.maximum,
    ) == (0, 127)
    assert (
        limits.kspace_encoding_step_2.minimum,
        limits.kspace_encoding_step_2.maximum,
    ) == (0, 19)
