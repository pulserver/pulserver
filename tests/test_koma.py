"""What the cache plays, simulated by KomaMRI: the exported file gives the design's signal.

Runs where ``PULSERVER_KOMA_PROJECT`` names a Julia project holding KomaMRI,
such as ``tests/koma`` once instantiated, with ``julia`` on the path.
"""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp
import pytest
from _zoo import SMALL, designed
from pypulseqpp import sequences

from pulserver import ir, virtual

PROJECT = os.environ.get("PULSERVER_KOMA_PROJECT")
KOMA = Path(__file__).parent / "koma"
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SYSTEM = pp.Opts(B0=3.0)
#: Relative to the peak sample of the design's signal. The text format writes a
#: turned gradient's amplitude to six significant figures, and the phase that
#: rounding accrues over a spoiler's moment stays well below this.
TOLERANCE = 1e-3
#: The cases ``PULSERVER_KOMA_CASES`` names, comma-separated; every one without it.
NAMES = [
    name
    for name in sorted(
        p.stem for p in FIXTURES.glob("*.seq") if not p.stem.endswith("_b")
    )
    + sorted(SMALL)
    if not os.environ.get("PULSERVER_KOMA_CASES")
    or name in os.environ["PULSERVER_KOMA_CASES"].split(",")
]

pytestmark = pytest.mark.skipif(
    PROJECT is None, reason="PULSERVER_KOMA_PROJECT names no Julia project"
)


def _phantom():
    """Return spins filling an elliptic cylinder 20 cm long around the isocentre.

    Planes of spins lie 2 mm apart within 3 cm of the isocentre, where the
    slices of the 2D sequences lie, and 10 mm apart beyond, past the edges of
    the slabs. Density, T1, T2 and off-resonance each vary across the object,
    none of them symmetrically, so a turned or mistimed readout, a shifted
    slice or slab and a wrong pulse each change the signal. Positions are in
    m, relaxation times in s and the off-resonance in Hz.
    """
    grid = np.linspace(-0.08, 0.08, 17)
    beyond = np.linspace(0.04, 0.1, 7)
    planes = np.concatenate([-beyond[::-1], np.linspace(-0.03, 0.03, 31), beyond])
    x, y, z = np.meshgrid(grid, grid, planes, indexing="ij")
    inside = (x / 0.09) ** 2 + (y / 0.07) ** 2 <= 1.0
    x, y, z = x[inside], y[inside], z[inside]
    return SimpleNamespace(
        positions=np.column_stack([x, y, z]),
        density=1.0 + 0.5 * (x > 0.02) - 0.3 * (y < -0.03) + 0.2 * (z > 0.0),
        t1=0.6 + 5.0 * (x + 0.08),
        t2=0.04 + 0.5 * (y + 0.08),
        off_resonance=20.0 * y / 0.07 + 10.0 * z / 0.03,
    )


def _write_phantom(spins, path):
    """Write the spins as ``simulate.jl`` reads them: their count, then x, y, z, density, T1, T2 and the off-resonance in rad/s."""
    columns = (
        *spins.positions.T,
        spins.density,
        spins.t1,
        spins.t2,
        2.0 * np.pi * spins.off_resonance,
    )
    with path.open("wb") as file:
        np.array([spins.density.size], dtype="<i8").tofile(file)
        for column in columns:
            column.astype("<f8").tofile(file)


def _design(name, directory):
    """Write the design ``name`` names into ``directory`` and return its first file."""
    if name in SMALL:
        path = directory / "scan.seq"
        sequences.write(path, designed(name))
        return path
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    return directory / f"{name}.seq"


def _as_komamri_reads_it(first, directory):
    """Return the design's files in play order, each as KomaMRI is to read it.

    KomaMRI drops the ppm term of an RF or ADC offset, so a file carrying one
    is given to it as Pulseq 1.4.1, whose writer resolves ppm offsets at the
    field the cache is converted at.
    """
    files = []
    for index, (path, seq) in enumerate(pp.io.read_chain(first)):
        libraries = seq.libraries()
        if np.any(libraries.rf[:, 6:8]) or np.any(libraries.adc[:, 3:5]):
            path = directory / f"resolved_{index}.seq"
            seq.write_v141(str(path), gamma=SYSTEM.gamma, field=SYSTEM.B0)
        files.append(str(path.resolve()))
    return files


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """The signals KomaMRI simulates of each design, by name, and of its exported cache, each without the ADC's offsets."""
    root = tmp_path_factory.mktemp("koma")
    spins = _phantom()
    _write_phantom(spins, root / "phantom.bin")
    cases = []
    for name in NAMES:
        case = root / name
        (case / "design").mkdir(parents=True)
        first = _design(name, case / "design")
        ir.convert(first, SYSTEM)
        virtual.export(first, case / "exported.seq", SYSTEM)
        files = _as_komamri_reads_it(first, case)
        (case / "design.txt").write_text("\n".join(files) + "\n")
        cases.append(case)
    subprocess.run(
        [
            "julia",
            f"--project={PROJECT}",
            str(KOMA / "simulate.jl"),
            str(root / "phantom.bin"),
            *map(str, cases),
        ],
        check=True,
    )
    signals = {}
    for case in cases:
        design, exported = (
            np.fromfile(case / f"{kind}.sig", dtype="<c16")
            for kind in ("design", "exported")
        )
        signals[case.name] = SimpleNamespace(design=design, exported=exported)
    return signals


def _apart(signal, reference):
    """Return the largest difference between two signals, relative to the reference's peak sample."""
    return np.abs(signal - reference).max() / np.abs(reference).max()


@pytest.mark.parametrize("name", NAMES)
def test_komamri_simulates_the_signal_of_the_design_from_the_exported_cache(
    simulated, name
):
    design, exported = simulated[name].design, simulated[name].exported

    assert exported.size == design.size
    print(f"{name}: {_apart(exported, design):.1e} of the design's peak")
    np.testing.assert_allclose(
        exported, design, rtol=0, atol=TOLERANCE * np.abs(design).max()
    )
